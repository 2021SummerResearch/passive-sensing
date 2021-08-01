import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import reading

xlen = 520
catagory = 8

#读取前文训练好的模型
model = tf.keras.models.load_model('model/categorical_hinge_try2.h5')

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
'''
print(len(Swipe_left))
print(len(Swipe_right))
print(len(Swipe_up))
print(len(Swipe_down))
print(len(Rotate_Left))
print(len(Rotate_Right))
print(len(Zoom_in))
print(len(Zoom_out))
print(len(Swipe_left_1))
print(len(Swipe_right_1))
print(len(Swipe_up_1))
print(len(Swipe_down_1))
print(len(Rotate_Left_1))
print(len(Rotate_Right_1))
print(len(Zoom_in_1))
print(len(Zoom_out_1))


# 连接数据集并打乱
x_data = np.vstack((Swipe_left, Swipe_right, Swipe_up, Swipe_down, Rotate_Left, Rotate_Right, Zoom_in, Zoom_out,
                    Swipe_left_1, Swipe_right_1, Swipe_up_1, Swipe_down_1, Rotate_Left_1, Rotate_Right_1, Zoom_in_1,
                    Zoom_out_1))

np.random.seed(1337)
np.random.shuffle(x_data)
x_dataset, y_dataset = np.split(x_data, [xlen], axis=1)

y_dataset = reading.to_vector(y_dataset)

# 拆分数据，60%做训练集，20%做验证集，20%做测试集
x_train, x_validate, x_test = np.split(x_dataset, [int(0.6 * len(y_dataset)), int(0.8 * len(y_dataset))])
y_train, y_validate, y_test = np.split(y_dataset, [int(0.6 * len(y_dataset)), int(0.8 * len(y_dataset))])

'''
data_test = np.vstack((Swipe_left, Swipe_right, Swipe_up, Swipe_down, Rotate_Left, Rotate_Right, Zoom_in, Zoom_out))
data_train = np.vstack((Swipe_left_1, Swipe_right_1, Swipe_up_1, Swipe_down_1, Rotate_Left_1, Rotate_Right_1, Zoom_in_1,
                    Zoom_out_1))
x_train, y_train = np.split(data_train, [xlen], axis=1)
x_test, y_test = np.split(data_test, [xlen], axis=1)
y_test = reading.to_vector(y_test)
y_train = reading.to_vector(y_train)


#生成预测的结果数据，以one-hot形式编码，即对应类别的向量
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

#选取概率最大的类别作为预测结果，即令最大处为1,其他位置为0
y_train_p = np.zeros((len(y_train_pred),len(y_train_pred[0])))
y_test_p = np.zeros((len(y_test_pred),len(y_test_pred[0])))
for i in range(len(y_train_pred)):
  index = np.argmax(y_train_pred[i])
  y_train_p[i,index] = 1
for i in range(len(y_test_pred)):
  index = np.argmax(y_test_pred[i])
  y_test_p[i,index] = 1

#计算最终的准确率
#将预测结果与真是答案的矩阵对应位置相乘，再返回每行的最大值，若该行最大值为1，即说明预测正确，若为0，则说明预测错误
result_train = np.max(np.multiply(y_train_p, y_train), axis = 1)
result_test = np.max(np.multiply(y_test_p, y_test), axis = 1)
#将最大值矩阵所有元素之和除以数据集中元素总个数，得到准确率
accuracy_train = np.sum(result_train)/len(y_train)
accuracy_test = np.sum(result_test)/len(y_test)

matrx_train = np.dot(y_train_p.T, y_train)
matrx_test = np.dot(y_test_p.T, y_test)
sum_train = np.sum(matrx_train, axis=0)
sum_test = np.sum(matrx_test, axis=0)

for i in range(catagory):
    tmp = sum_train[i]
    for j in range(catagory):
        matrx_train[j,i] = matrx_train[j,i] / tmp

for i in range(catagory):
    tmp = sum_test[i]
    for j in range(catagory):
        matrx_test[j,i] = matrx_test[j,i] / tmp


#打印最终准确率结果
print('The accuracy on train dataset is: {}'.format(accuracy_train))
print('The accuracy on test dataset is: {}'.format(accuracy_test))
