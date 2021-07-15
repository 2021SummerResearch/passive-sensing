import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
flag_RawData = 1            # 需要显示原始数据时置1
flag_ProcessedData = 1      # 需要显示处理过的数据时置1
flag_GenExcel = 0           # 需要将处理过的数据生成excel文件时置1
data = pd.read_csv("./FIR_another_hightlight.csv", encoding="big5")  # 从CSV文件导入

raw_data = data.to_numpy()
print(len(raw_data))  # 原始数据个数

num_2CH = 1000  # 双通道每组数据的个数
num_show = 9    # 需要画图的数据

dataSplit = {}
for i in range(int(len(raw_data) / num_2CH)):
    sample = np.empty([num_2CH])
    sample = raw_data[i * num_2CH:(i + 1) * num_2CH]
    dataSplit[i] = sample  # dataSplit[i]代表第i组数据
print(len(dataSplit))  # 共有数据184组
print(len(dataSplit[0]))  # 每组数据1000个元素，前500个代表通道1，后500个代表通道2
# 归一化方法：线性归一化: x'=(x-x_min)/(x_max-x_min)
for i in range(int(len(dataSplit))):
    dataSplit[i] = (dataSplit[i] - min(dataSplit[i])) / (max(dataSplit[i]) - min(dataSplit[i]))
data_ch1 = dataSplit[0][0:int(num_2CH / 2), :]  # 1通道数据
data_ch2 = dataSplit[0][int(num_2CH / 2):num_2CH, :]  # 2通道数据
# 将数据分成通道1和通道2各184组
dataChannel_1 = {}
dataChannel_2 = {}
for i in range(int(len(dataSplit))):
    dataChannel_1[i] = dataSplit[i][0:int(num_2CH / 2), :]
    dataChannel_2[i] = dataSplit[i][int(num_2CH / 2):num_2CH, :]

if flag_RawData == 1:
    plt.figure()
    for i in range(5):
        plt.figure(i)
        plt.plot(dataChannel_1[i])
        plt.plot(dataChannel_2[i])
    plt.show()

# 尝试在x方向归一化一组数据
min_loc_ch1 = np.argmin(data_ch1)  # 1通道数据最小值的位置
min_loc_ch2 = np.argmin(data_ch2)  # 2通道数据最小值的位置
print(min_loc_ch1)
print(min_loc_ch2)
min_loc_avg = int((min_loc_ch1 + min_loc_ch2) / 2)  # 两通道的平均位置
print(min_loc_avg)

# 在x方向归一化全部数据 操作对象：dataChannel_1[i]和dataChannel_2[i]
# 得到的1通道数据和2通道数据分别为data_ch1_xnorm[i]和data_ch2_xnorm[i]
width = 130  # x归一化时，最低点位置平均值两边的宽度
data_ch1_xnorm = {}
data_ch2_xnorm = {}
for i in range(int(len(dataChannel_1))):

    min_loc_ch1 = np.argmin(dataChannel_1[i])  # 1通道数据最小值的位置
    min_loc_ch2 = np.argmin(dataChannel_2[i])  # 2通道数据最小值的位置
    min_loc_avg = int((min_loc_ch1 + min_loc_ch2) / 2)  # 12通道平均位置
    if not min(len(dataChannel_1[i]) - min_loc_avg, min_loc_avg) <= width:
        data_ch1_xnorm[i] = dataChannel_1[i][min_loc_avg - width:min_loc_avg + width]
        data_ch2_xnorm[i] = dataChannel_2[i][min_loc_avg - width:min_loc_avg + width]
    elif min_loc_avg <= width:
        array_to_fix_1 = np.linspace(start=dataChannel_1[i][0], stop=dataChannel_1[i][0], num=width - min_loc_avg)
        data_ch1_xnorm[i] = dataChannel_1[i][0:min_loc_avg + width]
        data_ch1_xnorm[i] = np.concatenate((array_to_fix_1, data_ch1_xnorm[i]))
        data_ch2_xnorm[i] = dataChannel_2[i][0:min_loc_avg + width]
        data_ch2_xnorm[i] = np.concatenate((array_to_fix_1, data_ch2_xnorm[i]))
    else:
        array_to_fix_2 = np.linspace(start=dataChannel_1[i][-1], stop=dataChannel_1[i][-1]
                                     , num=width - len(dataChannel_1[i]) + min_loc_avg)
        data_ch1_xnorm[i] = dataChannel_1[i][min_loc_avg - width:]
        data_ch1_xnorm[i] = np.concatenate((data_ch1_xnorm[i], array_to_fix_2))
        data_ch2_xnorm[i] = dataChannel_2[i][min_loc_avg - width:]
        data_ch2_xnorm[i] = np.concatenate((data_ch2_xnorm[i], array_to_fix_2))

print(len(data_ch1_xnorm))
print(len(data_ch2_xnorm))

if flag_ProcessedData == 1:
    for i in range(5):
        plt.figure(i)
        plt.plot(data_ch1_xnorm[i])
        plt.plot(data_ch2_xnorm[i])
    plt.show()

# 转化成excel文件
if flag_GenExcel == 1:
    data_sum = pd.Series()
    for i in range(int(len(data_ch1_xnorm))):
        a = data_ch1_xnorm[i]
        a = a.flatten()
        a = pd.Series(a)
        b = data_ch2_xnorm[i]
        b = b.flatten()
        b = pd.Series(b)
        data_sum = pd.concat([data_sum, a])
        data_sum = pd.concat([data_sum, b])

    writer = pd.ExcelWriter("SwipeUp_week2.xlsx")
    data_sum.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
