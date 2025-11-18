import argparse
import sys
import time

import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


MODEL_PATH = 'modelos/efficientdet_lite0.tflite'
cAM_ID = 0
FRAME_WIDTH = 160
FRAME_HEIGHT = 120
NUM_THREADS = 4
ENABLE_EDGETPU = False

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + bbox.origin_x,
                     _MARGIN + _ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  
  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.4)
  options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)

  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = visualize(image, detection_result)

    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x_min = bbox.origin_x
        y_min = bbox.origin_y
        x_max = bbox.origin_x + bbox.width
        y_max = bbox.origin_y + bbox.height

        for category in detection.categories:
            print(f"[DETECÇÃO] Classe: {category.category_name} | Confiança: {category.score:.2f}")
            print(f"→ Coordenadas: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            print("-" * 60)

    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()

def main():
  run(
      MODEL_PATH,
      cAM_ID,
      FRAME_WIDTH,
      FRAME_HEIGHT,
      NUM_THREADS,
      ENABLE_EDGETPU)

if __name__ == '__main__':
  main()
