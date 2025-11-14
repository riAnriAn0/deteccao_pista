from tflite_model_maker import object_detector
from tflite_model_maker import model_spec

# Step 1: Choose the model architecture
spec = model_spec.get('efficientdet_lite0')

# Step 2: Load your training data
train_data, validation_data, test_data = object_detector.DataLoader.from_csv("data_set/labels_detecção_pista_2025-11-10-08-32-00.csv")

# Step 3: Train a custom object detector
model = object_detector.create(train_data, model_spec=spec, validation_data=validation_data)

# Step 4: Export the model in the TensorFlow Lite format
model.export(export_dir='.')

# Step 5: Evaluate the TensorFlow Lite model
model.evaluate_tflite('model.tflite', test_data)