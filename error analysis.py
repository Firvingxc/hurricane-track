import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
# 加载数据
data_file = 'test20202_cleaned.txt'  # 替换为你的文件路径
truth_file = 'coodi_cnntruth24.txt'  # 替换为你的文件路径
pred_file = 'coodi_cnnpred24.txt'  # 替换为你的文件路径

# 加载数据文件
data = pd.read_csv(data_file, names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
# truth = pd.read_csv(truth_file, names=['lat', 'lon'])
# pred = pd.read_csv(pred_file, names=['lat', 'lon'])

# Ensure that the 'direct' column is of numeric type
data['direct'] = pd.to_numeric(data['direct'], errors='coerce')

# Calculate the difference between the current and previous 'direct' values
data['direct_diff'] = data['direct'].diff()
print(data['direct_diff'])
# Handle the wrap-around to keep the differences within the range -180 to 180 degrees
data['direct_diff'] = ((data['direct_diff'] + 180) % 360) - 180
print(data['direct_diff'])
# # Display the first few rows to verify
# print(data[['direct', 'direct_diff']].head())

# truth['lat'] = truth['lat'].astype(float)
# truth['lon'] = truth['lon'].astype(float)
data1 =  np.loadtxt('coodi_cnntruth24.txt')
data2 =  np.loadtxt('coodi_cnnpred24.txt')


import numpy as np
from haversine import haversine
# 计算每一对经纬度之间的距离，并将结果保存到一个列表中
distances = []
for coord1, coord2 in zip(data1, data2):
    distance = haversine((coord1[0], coord1[1]), (coord2[0], coord2[1]))
    distances.append(distance)


# 计算每个点的误差
errors = distances
data['error'] = errors
# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Extract the time part in HH:MM:SS format
data['time'] = data['date'].dt.strftime('%H:%M:%S')
#
# # Define times of interest
# times_of_interest = ['06:00:00', '12:00:00', '18:00:00', '00:00:00']
#
# # Loop over each time to create plots
# for time in times_of_interest:
#     # Filter data for the specific time
#     time_data = data[data['time'] == time]
#
#     # Calculate mean error by wind speed and pressure
#     ws_error = time_data.groupby('ws')['error'].mean().reset_index()
#     p_error = time_data.groupby('p')['error'].mean().reset_index()
#     dire_diff_error = data.groupby('direct_diff')['error'].mean().reset_index()
#
#     # Plot Error vs Wind Speed for the specific time
#     plt.figure(figsize=(12, 6))
#     plt.plot(ws_error['ws'], ws_error['error'], marker='o', linestyle='-', label=f'{time} - Wind Speed')
#     plt.title(f'Error vs Wind Speed at {time}')
#     plt.xlabel('Wind Speed (ws)')
#     plt.ylabel('Mean Error (km)')
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#
#     # Plot Error vs Pressure for the specific time
#     plt.figure(figsize=(12, 6))
#     plt.plot(p_error['p'], p_error['error'], marker='o', linestyle='-', label=f'{time} - Pressure', color='r')
#     plt.title(f'Error vs Pressure at {time}')
#     plt.xlabel('Pressure (p)')
#     plt.ylabel('Mean Error (km)')
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#
#
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(dire_diff_error['direct_diff'], dire_diff_error['error'], marker='o', linestyle='-', color='r')
#     plt.title('Error vs Pressure')
#     plt.xlabel('Pressure (p)')
#     plt.ylabel('Mean Error (km)')
#     plt.grid(True)
#     plt.show()
# 1. 将风速每10单位分一组
data['ws_bin'] = pd.cut(data['ws'], bins=range(0, int(data['ws'].max()) + 10, 10), right=False)

# 2. 将气压每50单位分一组
data['p_bin'] = pd.cut(data['p'], bins=range(900, int(data['p'].max()) + 10, 10), right=False)

# 3. 方向差异每45度分一组 (-180到180分成8个区间)
data['direct_diff_bin'] = pd.cut(data['direct_diff'], bins=range(-180, 180, 30), right=False)

# 2. 将气压每50单位分一组
data['speed_bin'] = pd.cut(data['speed'], bins=range(0, int(data['speed'].max())+2 , 5), right=False)

# 4. 按分组计算平均误差
ws_error = data.groupby('ws_bin')['error'].mean().reset_index()
p_error = data.groupby('p_bin')['error'].mean().reset_index()
dire_diff_error = data.groupby('direct_diff_bin')['error'].mean().reset_index()
speed_error = data.groupby('speed_bin')['error'].mean().reset_index()


import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局绘图风格
mpl.rcParams.update({
    'font.size': 11,              # 设置字体大小
    'axes.labelsize': 11,         # 设置坐标轴标签字体大小
    'axes.titlesize': 11,         # 设置标题字体大小
    'xtick.labelsize': 11,        # 设置X轴刻度字体大小
    'ytick.labelsize': 11,        # 设置Y轴刻度字体大小
    'legend.fontsize': 11,        # 设置图例字体大小
    'figure.figsize': (4, 3),     # 设置图表大小
    'axes.linewidth': 1,        # 设置坐标轴线条宽度
    'grid.linewidth': 0.4,        # 设置网格线宽度
    'lines.linewidth': 1,         # 设置曲线线条宽度
    'lines.markersize': 1,        # 设置标记点大小
    'xtick.major.size': 1,        # X轴主刻度大小
    'ytick.major.size': 1,        # Y轴主刻度大小
    'grid.alpha': 0.6,            # 设置网格透明度
    'legend.frameon': False,      # 图例去掉边框
})

# 绘制风速分组后的平均误差
# plt.figure()
# plt.plot(ws_error['ws_bin'].astype(str), ws_error['error'], marker='o', linestyle='-')
#
# # plt.title('Error vs Wind Speed (Grouped)', fontsize=11)
# plt.xlabel('Wind speed (kt)', fontsize=11)
# plt.ylabel('Mean error (km)', fontsize=11)
# plt.grid(True, linestyle='--', alpha=0.7)
#
# plt.xticks(ticks=range(0, len(ws_error['ws_bin']), 4), labels=ws_error['ws_bin'].astype(str)[::4])
#
# plt.tight_layout()  # 自动调整子图参数，适应图表布局
#
# plt.savefig("Wind speed.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
# plt.show()

# 绘制气压分组后的平均误差
plt.figure()
plt.plot(p_error['p_bin'].astype(str), p_error['error'], marker='o', linestyle='-')

# plt.title('Error vs Pressure (Grouped)', fontsize=11)
plt.xlabel('Pressure (hPa)', fontsize=11)
plt.ylabel('Mean error (km)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

plt.xticks(ticks=range(0, len(p_error['p_bin']), 2), labels=p_error['p_bin'].astype(str)[::2])
plt.tight_layout()

plt.savefig("Pressure.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
plt.show()

# 绘制方向差异分组后的平均误差
# plt.figure()
# plt.plot(dire_diff_error['direct_diff_bin'].astype(str), dire_diff_error['error'], marker='o', linestyle='-')
#
# # plt.title('Error vs Direction Difference (Grouped)', fontsize=16)
# plt.xlabel('Direction difference (°)', fontsize=11)
# plt.ylabel('Mean error (km)', fontsize=11)
# plt.grid(True, linestyle='--', alpha=0.7)
#
#
# plt.xticks(ticks=range(0, len(dire_diff_error['direct_diff_bin']), 3), labels=dire_diff_error['direct_diff_bin'].astype(str)[::3])
# plt.tight_layout()
#
# plt.savefig("Direction difference.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
# plt.show()


plt.figure()
plt.plot(speed_error['speed_bin'].astype(str), speed_error['error'], marker='o', linestyle='-')

# plt.title('Error vs Speed (Grouped)', fontsize=16)
plt.xlabel('Hurriance speed (kt)', fontsize=11)
plt.ylabel('Mean error (km)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

plt.xticks(ticks=range(0, len(speed_error['speed_bin']), 2), labels=speed_error['speed_bin'].astype(str)[::2])


plt.tight_layout()

plt.savefig("Hurriance speed.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
plt.show()



# 绘制风速分组后的平均误差
# plt.figure(figsize=(12, 6))
# plt.plot(ws_error['ws_bin'].astype(str), ws_error['error'], marker='o', linestyle='-')
#
# plt.title('Error vs Wind Speed (Grouped)')
# plt.xlabel('Wind Speed (Grouped)')
# plt.ylabel('Mean Error (km)')
# plt.grid(True)
# plt.show()
#
#
# plt.figure(figsize=(12, 6))
# plt.plot(p_error['p_bin'].astype(str), p_error['error'], marker='o', linestyle='-')
#
# plt.title('Error vs Wind Speed (Grouped)')
# plt.xlabel('Wind Speed (Grouped)')
# plt.ylabel('Mean Error (km)')
# plt.grid(True)
# plt.show()
#
#
# plt.figure(figsize=(12, 6))
# plt.plot(dire_diff_error['direct_diff_bin'].astype(str), dire_diff_error['error'], marker='o', linestyle='-')
#
# plt.title('Error vs Wind Speed (Grouped)')
# plt.xlabel('Wind Speed (Grouped)')
# plt.ylabel('Mean Error (km)')
# plt.grid(True)
# plt.show()
# # 1. 计算风速与误差的平均值关系
# ws_error = data.groupby('ws')['error'].mean().reset_index()
#
# # 2. 计算气压与误差的平均值关系
# p_error = data.groupby('p')['error'].mean().reset_index()
#
# dire_diff_error = data.groupby('direct_diff')['error'].mean().reset_index()

# 绘制误差随风速变化的曲线图
# plt.figure(figsize=(12, 6))
# plt.plot(ws_error['ws_bin'], ws_error['error'], marker='o', linestyle='-', color='b')
# plt.title('Error vs Wind Speed')
# plt.xlabel('Wind Speed (ws)')
# plt.ylabel('Mean Error (km)')
# plt.grid(True)
# plt.show()
#
# # 绘制误差随气压变化的曲线图
# plt.figure(figsize=(12, 6))
# plt.plot(p_error['p_bin'], p_error['error'], marker='o', linestyle='-', color='r')
# plt.title('Error vs Pressure')
# plt.xlabel('Pressure (p)')
# plt.ylabel('Mean Error (km)')
# plt.grid(True)
# plt.show()
#
#
# plt.figure(figsize=(12, 6))
# plt.plot(dire_diff_error['direct_diff_bin'], dire_diff_error['error'], marker='o', linestyle='-', color='r')
# plt.title('Error vs Pressure')
# plt.xlabel('Pressure (p)')
# plt.ylabel('Mean Error (km)')
# plt.grid(True)
# plt.show()


