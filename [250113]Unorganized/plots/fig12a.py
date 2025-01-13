import matplotlib.pyplot as plt
import numpy as np


categories = [
    "A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09",
    "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20"
][1:]

values_x = [
    1269581, 49938, 41603, 31501, 4064, 94, 2279, 735, 15745, 7533, 
    1053, 7969, 9477, 4711, 1172, 14639, 5398, 592, 7146, 4175, 605
][1:] # 低空

values_y = [
    1213985, 70657, 21056, 6712, 4035, 0, 1866, 2167, 40362, 10484,
    79, 1861, 16655, 893, 0, 3473, 1990, 1609, 2616, 1683, 0
][1:] # 高空


# 设置颜色和透明度
colors = [(100, 150, 245), (100, 230, 245), (30, 60, 15),(80, 30, 180),(100, 80, 250),(255, 30, 30),(255,40,200),(150, 30, 90),
          (255, 0, 250),(255, 150, 255),(75, 0, 75),(175, 0, 75),(255, 200, 0),(255, 120, 50),(0, 175, 0),(135, 60, 0),(150, 240, 80),(255, 240, 150),(255, 0, 0),(50, 100, 200), (200, 100, 50)][:len(categories)]
alpha_x = [0.5]*len(categories)
alpha_y = [1.0]*len(categories)

plt.figure(figsize=(20, 4))
# 处理其他类别
for i in range(0, len(categories)):
    plt.bar(categories[i], values_x[i]+ values_y[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha_x[i], width=0.4,label='SUM')
    plt.bar(categories[i], values_y[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha_y[i], width=0.4,label='High Y')

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


plt.savefig('/data1/nemo/projects/NexusAD/plots/figure_4a.svg', format='png', dpi=300)
# plt.show()
