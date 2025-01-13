import matplotlib.pyplot as plt
import numpy as np


gp_scores = {'barriers': [0.04797229750909939, 0.05684858132922128], 'traffic_cones': [0.026902711717726388, 0.03187970412712177], 'vehicles': [0.26790350537924923, 0.2599964467551983], 'traffic_lights': [0.014834615330975538, 0.018706321541039538], 'vulnerable_road_users': [0.060451992478404826, 0.07206047389388272], 'other_objects': [0.03576309647557619, 0.04184758346256172], 'traffic_signs': [0.028664344936058614, 0.03616832506388454]}

ds_scores = {'vulnerable_road_users': [0.07109707426037278, 0.07849221602672529], 'barriers': [0.05806462736181902, 0.06328091492742395], 'other_objects': [0.04277051756215176, 0.050813028682394984], 'vehicles': [0.2228743027042895, 0.22404527639472085], 'traffic_signs': [0.03375904307174764, 0.03875099114819076], 'traffic_lights': [0.020178739369395914, 0.023896398289763882], 'traffic_cones': [0.035430411557115245, 0.03654645864388841]}


def extract_lists(data):
    # 按指定顺序排序的类别列表
    order = ['vehicles', 'vulnerable_road_users', 'traffic_signs', 'traffic_lights', 'traffic_cones', 'barriers', 'other_objects']
    
    # 根据顺序提取数据
    categories = order
    first_numbers = [data[category][0] for category in order]
    second_numbers = [data[category][1] for category in order]
    
    return categories, first_numbers, second_numbers


categories, values_x, values_y= extract_lists(gp_scores)

categories[1] = 'vru'

# 设置颜色和透明度
colors = [(100, 150, 245), (100, 230, 245), (30, 60, 15),(80, 30, 180),(100, 80, 250),(255, 30, 30),(255,40,200),(150, 30, 90),
          (255, 0, 250),(255, 150, 255),(75, 0, 75),(175, 0, 75),(255, 200, 0),(255, 120, 50),(0, 175, 0),(135, 60, 0),(150, 240, 80),(255, 240, 150),(255, 0, 0),(50, 100, 200), (200, 100, 50)][:len(categories)]
alpha_x = [0.5]*len(categories)
alpha_y = [1.0]*len(categories)

plt.figure(figsize=(10, 4))
# 处理其他类别
for i in range(0, len(categories)):
    plt.bar(categories[i], values_x[i]+values_y[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha_x[i], width=0.4,label='SUM')
    plt.bar(categories[i], values_y[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha_y[i], width=0.4,label='High Y')

# 设置y轴为对数尺度
# plt.yscale('log')
# plt.gca().set_ylim([1, max(values_y) * 10])

# 添加标题和标签
# plt.title('柱状图')
# plt.xlabel('类别')
# plt.ylabel('数量')
# plt.legend() # 添加图例

# 显示图形
plt.show()


plt.savefig('/data1/nemo/projects/NexusAD/plots/figure_4b.png', format='png', dpi=300)
# plt.show()
