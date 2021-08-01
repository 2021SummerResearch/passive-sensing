import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import reading

xlen = 520
catagory = 8

# 读取数据
Swipe_left = reading.read_data('week2data/week2data_zym/swipe_left_coffee.xlsx', 'page_1', 0)
Swipe_right = reading.read_data('week2data/week2data_zym/swipe_right_coffee.xlsx', 'page_1', 1)
Swipe_up = reading.read_data('week2data/week2data_zym/swipe_up_coffee.xlsx', 'page_1', 2)
Swipe_down = reading.read_data('week2data/week2data_zym/swipe_down_coffee.xlsx', 'page_1', 3)
Rotate_Left = reading.read_data('week2data/week2data_zym/rotate_left_coffee.xlsx', 'page_1', 4)
Rotate_Right = reading.read_data('week2data/week2data_zym/rotate_right_coffee.xlsx', 'page_1', 5)
Zoom_in = reading.read_data('week2data/week2data_zym/zoom_in_coffee.xlsx', 'page_1', 6)
Zoom_out = reading.read_data('week2data/week2data_zym/zoom_out_coffee.xlsx', 'page_1', 7)

Swipe_left_1 = reading.read_data('week2data/week2data_yzc/SwipeLeft_week2.xlsx', 'page_1', 0)
Swipe_right_1 = reading.read_data('week2data/week2data_yzc/SwipeRight_week2.xlsx', 'page_1', 1)
Swipe_up_1 = reading.read_data('week2data/week2data_yzc/SwipeUp_week2.xlsx', 'page_1', 2)
Swipe_down_1 = reading.read_data('week2data/week2data_yzc/SwipeDown_week2.xlsx', 'page_1', 3)
Rotate_Left_1 = reading.read_data('week2data/week2data_yzc/RotateLeft_week2.xlsx', 'page_1', 4)
Rotate_Right_1 = reading.read_data('week2data/week2data_yzc/RotateRight_week2.xlsx', 'page_1', 5)
Zoom_in_1 = reading.read_data('week2data/week2data_yzc/ZoomIn_week2.xlsx', 'page_1', 6)
Zoom_out_1 = reading.read_data('week2data/week2data_yzc/ZoomOut_week2.xlsx', 'page_1', 7)

#连接数据集并打乱
x_data = np.vstack((Swipe_left, Swipe_right, Swipe_up, Swipe_down, Rotate_Left, Rotate_Right, Zoom_in, Zoom_out, Swipe_left_1, Swipe_right_1, Swipe_up_1, Swipe_down_1, Rotate_Left_1, Rotate_Right_1, Zoom_in_1, Zoom_out_1))
np.random.seed(1337)
np.random.shuffle(x_data)
x_dataset, y_dataset = np.split(x_data, [xlen], axis=1)

y_dataset = reading.to_vector(y_dataset)
for i in range(len(y_dataset)):
  for j in range(len(y_dataset[i])):
    y_dataset[i, j] = float(y_dataset[i, j])

# 拆分数据，60%做训练集，20%做验证集，20%做测试集
x_train, x_validate, x_test = np.split(x_dataset, [int(0.6 * len(y_dataset)), int(0.8 * len(y_dataset))])
y_train, y_validate, y_test = np.split(y_dataset, [int(0.6 * len(y_dataset)), int(0.8 * len(y_dataset))])

print(x_train.shape,x_validate,x_test)
'''
model = tf.keras.models.load_model('model/categorical_hinge_both.h5')
model.save('model/mymodel')


# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_saved_model('model/mymodel')
model_no_quant_tflite = converter.convert()

# Save the model to disk
open('model/model_no_quant.tflite', "wb").write(model_no_quant_tflite)


def predict_tflite(tflite_model, x_test):
  # Prepare the test data
  x_test_ = x_test.copy()

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # If required, quantize the input layer (from float to integer)
  input_scale, input_zero_point = input_details["quantization"]
  if (input_scale, input_zero_point) != (0.0, 0):
    x_test_ = x_test_ / input_scale + input_zero_point
    x_test_ = x_test_.astype(input_details["dtype"])

  # Invoke the interpreter
  y_pred = np.empty(x_test_.size, dtype=output_details["dtype"])
  for i in range(len(x_test_)):
    interpreter.set_tensor(input_details["index"], [x_test_[i]])
    interpreter.invoke()
    y_pred[i] = interpreter.get_tensor(output_details["index"])[0]

  # If required, dequantized the output layer (from integer to float)
  output_scale, output_zero_point = output_details["quantization"]
  if (output_scale, output_zero_point) != (0.0, 0):
    y_pred = y_pred.astype(np.float32)
    y_pred = (y_pred - output_zero_point) * output_scale

  return y_pred

# Calculate predictions
y_test_pred_tf = model.predict(x_test)
y_test_pred_no_quant_tflite = predict_tflite(model_no_quant_tflite, x_test)
'''