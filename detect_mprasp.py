# detect_mp.py
"""
Detector multiprocessado para Raspberry Pi.
- Usa workers que criam seus próprios ObjectDetector.
- Exportadores (CSV/Socket) via export_utils.py
- Imprime no terminal: FPS, frame_id e classes detectadas + infer_time (ms)
"""

import argparse
import time
import signal
from multiprocessing import Process, Queue, Event, cpu_count, set_start_method
import multiprocessing as mp

import cv2
import numpy as np
from tflite_support.task import core, processor, vision

# Visual settings
_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)

def visualize(image, detections):
    """Desenha caixas em image a partir da lista de detections (dicts)."""
    for det in detections:
        start = (det["x_min"], det["y_min"])
        end = (det["x_max"], det["y_max"])
        cv2.rectangle(image, start, end, _TEXT_COLOR, 2)
        label = f"{det['class']} ({det['score']:.2f})"
        loc = (_MARGIN + det["x_min"], _MARGIN + _ROW_SIZE + det["y_min"])
        cv2.putText(image, label, loc, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    return image

def worker_process(worker_id, model_path, input_q: mp.Queue, output_q: mp.Queue, stop_event: mp.Event, num_threads: int, enable_edgetpu: bool, resize_w: int, resize_h: int):
    """Worker: carrega detector e processa frames da input_q. Retorna (frame_id, detections_list)."""
    try:
        base_options = core.BaseOptions(file_name=model_path, use_coral=enable_edgetpu, num_threads=num_threads)
        detection_options = processor.DetectionOptions(max_results=5, score_threshold=0.4)
        opts = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(opts)
        print(f"[W{worker_id}] Modelo carregado.")
    except Exception as e:
        print(f"[W{worker_id}] Erro ao carregar modelo: {e}")
        return

    while not stop_event.is_set():
        try:
            item = input_q.get(timeout=0.5)
        except Exception:
            continue

        if item is None:
            break

        frame_id, frame_bgr = item
        try:
            # detector espera RGB com tamanho correto; assumimos frame já redimensionado pelo main
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = vision.TensorImage.create_from_array(rgb)

            t0 = time.time()
            detection_result = detector.detect(tensor)
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000.0
        except Exception as e:
            print(f"[W{worker_id}] Erro inferência: {e}")
            continue

        out_list = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x_min = int(bbox.origin_x)
            y_min = int(bbox.origin_y)
            x_max = int(bbox.origin_x + bbox.width)
            y_max = int(bbox.origin_y + bbox.height)
            for category in detection.categories:
                out_list.append({
                    "frame_id": frame_id,
                    "class": category.category_name,
                    "score": float(category.score),
                    "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
                    "worker": worker_id,
                    "infer_time_ms": infer_ms
                })

        try:
            output_q.put((frame_id, out_list))
        except Exception as e:
            print(f"[W{worker_id}] Falha ao enviar resultado: {e}")

    print(f"[W{worker_id}] Worker finalizando.")

def run(model, camera_id, frame_width, frame_height, num_workers, worker_num_threads, enable_edgetpu,
        out_csv, send_socket, socket_host, socket_port, resize_w, resize_h):
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    max_queue_size = 4
    input_q = mp.Queue(maxsize=max_queue_size)
    output_q = mp.Queue(maxsize=max_queue_size * 2)
    stop_event = mp.Event()

    workers = []
    for i in range(num_workers):
        p = Process(target=worker_process, args=(i, model, input_q, output_q, stop_event, worker_num_threads, enable_edgetpu, resize_w, resize_h), daemon=True)
        p.start()
        workers.append(p)

    def _handler(sig, frame):
        print("Sinal recebido: encerrando...")
        stop_event.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    frame_id = 0
    frame_buffer = {}
    buffer_max = 8

    counter, fps = 0, 0.0
    start_time = time.time()
    fps_avg_count = 10

    try:
        while True:
            ok, full_frame = cap.read()
            if not ok:
                print("Erro lendo câmera.")
                break

            frame_id += 1
            counter += 1
            full_frame = cv2.flip(full_frame, 1)

            # prepara e envia versão redimensionada para worker
            small = cv2.resize(full_frame, (resize_w, resize_h))

            try:
                if input_q.full():
                    try:
                        input_q.get_nowait()
                    except Exception:
                        pass
                input_q.put_nowait((frame_id, small))
            except Exception:
                pass

            frame_buffer[frame_id] = full_frame
            if len(frame_buffer) > buffer_max:
                keys = sorted(frame_buffer.keys())
                for k in keys[: len(frame_buffer) - buffer_max]:
                    frame_buffer.pop(k, None)

            # processa resultados disponíveis (não bloqueante)
            try:
                while not output_q.empty():
                    fid, detections = output_q.get_nowait()

                    # calcular/atualizar fps a cada fps_avg_count capturas
                    if counter % fps_avg_count == 0:
                        now = time.time()
                        fps = fps_avg_count / (now - start_time) if now - start_time > 0 else 0.0
                        start_time = now

                    # preparar string de classes detectadas e média de tempo de inferência
                    if detections:
                        classes = [d["class"] for d in detections]
                        classes_str = ", ".join(sorted(set(classes)))
                        # avg_infer = sum(d.get("infer_time_ms", 0.0) for d in detections) / len(detections)
                        print(f"[INFO] FPS: {fps:.2f} | frame_id: {fid} | Classes: {classes_str}")
                    else:
                        # nenhum objeto detectado nesse frame
                        print(f"[INFO] FPS: {fps:.2f} | frame_id: {fid} | Nenhum objeto detectado.")

                    # tenta desenhar se tivermos o frame original
                    frame = frame_buffer.get(fid)
                    # if frame is not None:
                        # frame = visualize(frame, detections)
                        # cv2.putText(frame, f"FPS: {fps:.1f}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                        # cv2.imshow("Detector MP", frame)

                    # exporta cada detecção
                    for det in detections:
                        rec = {
                            "timestamp": time.time(),
                            "frame_id": fid,
                            "class": det["class"],
                            "score": det["score"],
                            "x_min": det["x_min"],
                            "y_min": det["y_min"],
                            "x_max": det["x_max"],
                            "y_max": det["y_max"]
                        }
            except Exception:
                pass

            # if cv2.waitKey(1) == 27:
            #     break
            if stop_event.is_set():
                break

    finally:
        print("Encerrando pipeline...")
        stop_event.set()
        for _ in workers:
            try:
                input_q.put_nowait(None)
            except Exception:
                pass

        cap.release()
        # cv2.destroyAllWindows()

        for p in workers:
            p.join(timeout=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientdet_lite0.tflite")
    parser.add_argument("--cameraId", type=int, default=0)
    parser.add_argument("--frameWidth", type=int, default=320)
    parser.add_argument("--frameHeight", type=int, default=240)
    parser.add_argument("--numWorkers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--workerThreads", type=int, default=1, help="num_threads para cada interpreter no worker")
    parser.add_argument("--enableEdgeTPU", action="store_true")
    parser.add_argument("--out_csv", type=str, default="", help="Caminho CSV")
    parser.add_argument("--send_socket", action="store_true")
    parser.add_argument("--socket_host", default="127.0.0.1")
    parser.add_argument("--socket_port", type=int, default=5000)
    parser.add_argument("--resize_w", type=int, default=320, help="largura enviada ao worker")
    parser.add_argument("--resize_h", type=int, default=240, help="altura enviada ao worker")

    args = parser.parse_args()

    run(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numWorkers,args.workerThreads, args.enableEdgeTPU, args.out_csv, args.send_socket, args.socket_host, args.socket_port,args.resize_w, args.resize_h)
