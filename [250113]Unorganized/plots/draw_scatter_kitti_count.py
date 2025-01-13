import matplotlib.pyplot as plt
import numpy as np

# 定义类别和对应的数量
categories = ['car', 'bicycle', 'motorcycle', 'truck','other-vehicle','person','bicyclist','motorcyclist','road','parking',
              'sidewalk','other-ground','building','fence','vegetation','trunk','terrain','pole','traffic-sign']
values_x = [9.61662000e+04, 3.69700000e+03, 9.84500000e+02, 
            1.40400000e+03, 1.09980000e+04, 1.19180000e+04, 3.71000000e+03,
            9.12600000e+02, 8.16005000e+04, 1.45500000e+04, 6.09377000e+04, 
            7.39680000e+03, 5.45580000e+04, 2.00740000e+04, 7.15833000e+04,
            4.65410000e+04, 6.01452000e+04, 2.19250000e+04,  5.44100000e+03]
values_y = [1.30287591e+08, 6.39562000e+05, 1.28189900e+06,
            5.09479400e+06, 7.67815600e+06, 1.29375900e+06, 6.05464000e+05,
            1.10516000e+05, 5.55049215e+08, 4.05254810e+07, 3.98464216e+08,
            9.63162600e+06, 3.68681416e+08, 1.82644338e+08, 7.73132761e+08,
            1.96926930e+07, 2.47714395e+08, 8.37871200e+06, 1.82343000e+06]

# 设置颜色和透明度
colors = [(100, 150, 245), (100, 230, 245), (30, 60, 15),(80, 30, 180),(100, 80, 250),(255, 30, 30),(255,40,200),(150, 30, 90),
          (255, 0, 250),(255, 150, 255),(75, 0, 75),(175, 0, 75),(255, 200, 0),(255, 120, 50),(0, 175, 0),(135, 60, 0),(150, 240, 80),(255, 240, 150),(255, 0, 0)]
alpha_x = [1.0, 1.0, 1.0,1.0,1.0,1.0, 1.0, 1.0,1.0,1.0,1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0]
alpha_y = [0.5, 0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5]


plt.figure(figsize=(20, 4))
# 处理其他类别
for i in range(0, len(categories)):
    
    plt.bar(categories[i], values_x[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha_x[i], width=0.4,label='数据集 X')
    plt.bar(categories[i], values_y[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha_y[i], width=0.4,label='数据集 Y')

# 设置y轴为对数尺度
plt.yscale('log')
plt.gca().set_ylim([1, max(values_y) * 10])

# 添加标题和标签
# plt.title('柱状图')
# plt.xlabel('类别')
# plt.ylabel('数量')
# plt.legend() # 添加图例

# 显示图形
plt.show()


plt.savefig('/data1/nemo/projects/NexusAD/figure_4a.png', format='png', dpi=300)
# plt.show()
