import openpyxl as ox
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

xlen = 520
catagory = 8

def read_data(xlsx_name, sheet_name, label):
    # 读取左滑手势，共184组，每组长度为520，前260帧为传感器1，后260帧为传感器2
    workbook = ox.load_workbook(xlsx_name)
    worksheet = workbook.get_sheet_by_name(sheet_name)
    left = [item.value for item in list(worksheet.columns)[1]]
    left = np.reshape(left, (int(len(left) / xlen), xlen))
    left = np.insert(left, xlen, values=label * np.ones(shape=(len(left)), dtype=int), axis=1)  # 给每一行最后一列打标签[label]
    return left

def to_vector(y_dataset):
    # 将y矩阵转化成向量形式
    y = np.zeros((len(y_dataset), catagory), dtype=int)
    for i in range(len(y_dataset)):
        y[i, int(y_dataset[i, 0])] = 1
    return y

def draw_curve(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g.', label='Training loss')
    plt.plot(epochs, val_loss, 'b.', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 读取数据

    Swipe_left = read_data('week2data/week2data_zym/swipe_left_coffee.xlsx', 'page_1', 0)
    Swipe_right = read_data('week2data/week2data_zym/swipe_right_coffee.xlsx', 'page_1', 1)
    Swipe_up = read_data('week2data/week2data_zym/swipe_up_coffee.xlsx', 'page_1', 2)
    Swipe_down = read_data('week2data/week2data_zym/swipe_down_coffee.xlsx', 'page_1', 3)
    Rotate_Left = read_data('week2data/week2data_zym/rotate_left_coffee.xlsx', 'page_1', 4)
    Rotate_Right = read_data('week2data/week2data_zym/rotate_right_coffee.xlsx', 'page_1', 5)
    Zoom_in = read_data('week2data/week2data_zym/zoom_in_coffee.xlsx', 'page_1', 6)
    Zoom_out = read_data('week2data/week2data_zym/zoom_out_coffee.xlsx', 'page_1', 7)


    Swipe_left_1 = read_data('week2data/week2data_yzc/SwipeLeft_week2.xlsx', 'page_1', 0)
    Swipe_right_1 = read_data('week2data/week2data_yzc/SwipeRight_week2.xlsx', 'page_1', 1)
    Swipe_up_1 = read_data('week2data/week2data_yzc/SwipeUp_week2.xlsx', 'page_1', 2)
    Swipe_down_1 = read_data('week2data/week2data_yzc/SwipeDown_week2.xlsx', 'page_1', 3)
    Rotate_Left_1 = read_data('week2data/week2data_yzc/RotateLeft_week2.xlsx', 'page_1', 4)
    Rotate_Right_1 = read_data('week2data/week2data_yzc/RotateRight_week2.xlsx', 'page_1', 5)
    Zoom_in_1 = read_data('week2data/week2data_yzc/ZoomIn_week2.xlsx', 'page_1', 6)
    Zoom_out_1 = read_data('week2data/week2data_yzc/ZoomOut_week2.xlsx', 'page_1', 7)

    #连接数据集并打乱
    x_data = np.vstack((Swipe_left, Swipe_right, Swipe_up, Swipe_down, Rotate_Left, Rotate_Right, Zoom_in, Zoom_out, Swipe_left_1, Swipe_right_1, Swipe_up_1, Swipe_down_1, Rotate_Left_1, Rotate_Right_1, Zoom_in_1, Zoom_out_1))
    np.random.seed(1337)
    np.random.shuffle(x_data)
    x_dataset, y_dataset = np.split(x_data, [xlen], axis=1)

    y_dataset = to_vector(y_dataset)

    # 拆分数据，60%做训练集，20%做验证集，20%做测试集
    x_train, x_validate, x_test = np.split(x_dataset, [int(0.6 * len(y_dataset)), int(1.0 * len(y_dataset))])
    y_train, y_validate, y_test = np.split(y_dataset, [int(0.6 * len(y_dataset)), int(1.0 * len(y_dataset))])


    # 定义模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(xlen,)))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(8))
    model.compile(optimizer='rmsprop', loss='categorical_hinge', metrics=['mae'])
    model.summary()

    # 训练模型
    history = model.fit(x_train, y_train, epochs=3000, batch_size=128,
                        validation_data=(x_validate, y_validate), verbose=2)
    model.save('')
    #draw_curve(history)