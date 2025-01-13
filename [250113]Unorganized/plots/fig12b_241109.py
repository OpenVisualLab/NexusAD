import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import os

from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt

# 设置字体路径和属性
font_path = "/data1/nemo/projects/NexusAD/plots/Times_New_Roman.ttf"
font_prop = font_manager.FontProperties(fname=font_path)

plt.rc('font',family='Times New Roman') 

# 数据

# gp_scores = {'other_objects': [0.09879308418667455, 0.11560105928884452], 'traffic_signs': [0.09491504945714774, 0.11976266577445212], 'vehicles': [0.28806828535403145, 0.2795660717797831], 'traffic_lights': [0.09759615349326012, 0.1230679048752601], 'vulnerable_road_users': [0.12972530574764984, 0.1546362100727097], 'traffic_cones': [0.10427407642529607, 0.12356474467876657], 'barriers': [0.10520240681820042, 0.12466794151145016]}
gp_scores = {'other_objects': [0.09879308418667455, 0.11560105928884452, 181], 'barriers': [0.10520240681820042, 0.12466794151145016, 228], 'vehicles': [0.28806828535403145, 0.2795660717797831, 465], 'traffic_cones': [0.10427407642529607, 0.12356474467876657, 129], 'traffic_signs': [0.09491504945714774, 0.11976266577445212, 151], 'traffic_lights': [0.09759615349326012, 0.1230679048752601, 76], 'vulnerable_road_users': [0.12972530574764984, 0.1546362100727097, 233]}

# ds_scores = {'traffic_lights': [0.12304109371582876, 0.145709745669292], 'traffic_signs': [0.11178491083360148, 0.12831454022579722], 'other_objects': [0.1162242325058472, 0.13807888228911686], 'traffic_cones': [0.13523057846227193, 0.13949030016751307], 'barriers': [0.1262274507865631, 0.13756720636396516], 'vulnerable_road_users': [0.15256882888492015, 0.1684382318170071], 'vehicles': [0.23964978785407479, 0.24090889934916226]}

ds_scores = {'other_objects': [0.1162242325058472, 0.13807888228911686, 184], 'traffic_signs': [0.11178491083360148, 0.12831454022579722, 151], 'traffic_cones': [0.13523057846227193, 0.13949030016751307, 131], 'barriers': [0.1262274507865631, 0.13756720636396516, 230], 'vehicles': [0.23964978785407479, 0.24090889934916226, 465], 'traffic_lights': [0.12304109371582876, 0.145709745669292, 82], 'vulnerable_road_users': [0.15256882888492015, 0.1684382318170071, 233]}

# 提取类别和数据
def extract_lists(data):
    order = ['vehicles', 'vulnerable_road_users', 'traffic_signs', 'traffic_lights', 'traffic_cones', 'barriers', 'other_objects']
    categories = order
    first_numbers = [data[category][0] for category in order]
    second_numbers = [data[category][1] for category in order]
    number_numbers = [data[category][2] for category in order]
    return categories, first_numbers, second_numbers, number_numbers

categories, gp_values_x, gp_values_y, nnnnn = extract_lists(gp_scores)
_, ds_values_x, ds_values_y, _ = extract_lists(ds_scores)

gp_r = (sum(gp_values_x)+sum(gp_values_y))/ (sum(gp_values_x)+sum(gp_values_y)+sum(ds_values_x)+sum(ds_values_y))
ds_r = 1-gp_r
gp_values_x = np.array(gp_values_x)/sum(gp_values_x)*gp_r
gp_values_y = np.array(gp_values_y)/sum(gp_values_y)*gp_r

nnnnn = np.array(nnnnn)/sum(nnnnn)

ds_values_x = np.array(ds_values_x)/sum(ds_values_x)*ds_r
ds_values_y = np.array(ds_values_y)/sum(ds_values_y)*ds_r

categories = ['vehicles', 'vru', 'traffic signs', 'traffic lights', 'traffic cones', 'barriers', 'miscellaneous']

# 设置颜色
colors = [(100, 150, 245), (100, 230, 245), (30, 60, 15), (80, 30, 180), (100, 80, 250), (255, 30, 30), (255, 40, 200)]
width = 0.2  # 柱子宽度

x = np.arange(len(categories))  # X轴的位置
gap = 0.05  # 类别之间的间距

# 创建图形
plt.figure(figsize=(8,4))

# 绘制数量占比的新柱子
for i in range(len(categories)):
    plt.bar(x[i] - (2*width - gap), nnnnn[i], width, color='gray', alpha=0.6, label='relative frequency' if i == 0 else "")


# 绘制 gp_scores 和 ds_scores 的柱状图
for i in range(len(categories)):
    plt.bar(x[i] - (width + gap) / 2, gp_values_x[i] + gp_values_y[i], width, color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=0.8)
    plt.bar(x[i] - (width + gap) / 2, gp_values_y[i], width, color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=1.0, label='contribution to general perception' if i == 0 else "")

    plt.bar(x[i] + (width + gap) / 2, ds_values_x[i] + ds_values_y[i], width, color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=0.4, hatch='//', edgecolor='black')
    plt.bar(x[i] + (width + gap) / 2, ds_values_y[i], width, color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=0.6, hatch='//', edgecolor='black', label='contribution to driving suggestion' if i == 0 else "")

font_size = 18
# 设置图例
plt.legend(loc='upper right', fontsize=18)

# 添加标题和标签
# plt.title('Comparison of GP and DS Scores by Category', fontsize=16, fontweight='bold')
# plt.xlabel('categories', fontsize=font_size, fontproperties=font_prop)
plt.ylabel('proportion', fontsize=font_size)
plt.xticks(x, categories, fontsize=font_size, rotation=20)

# 去掉图形边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 调整刻度线样式
plt.tick_params(axis='both', which='major', labelsize=font_size)

# 网格线，只显示在 y 轴上，符合学术风格
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 保存并显示图形
plt.tight_layout()
plt.savefig('/data1/nemo/projects/NexusAD/plots/figure_4b_comparison_cvpr_style.png', format='png', dpi=300)
plt.show()
