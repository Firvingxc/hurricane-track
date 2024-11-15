#2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap




data1 =  np.loadtxt('coodi_cnntruth24.txt')
# data1 = data1[273:330][::2]
data1 = data1[119:177][::2] #IRMA
#data1 = data1[627:644]
# data1 = data1[331:384][::2] #MARIA,2017
#data1 = data1[238:262]
# data1 = data1[1263:1299][::2] #laura
# data1 = data1[1058:1093][::2] # Jerry 2019
data2 =  np.loadtxt('coodi_cnnpred24.txt')
# data2 = data2[273:330][::2]
# data2 = data2[119:177] #IRMA
#data2 = data2[627:644]
# data2 = data2[331:384][::2]
#data2 = data2[238:262]
# data2 = data2[1263:1299][::2] #LAURA
data2  = data2[1058:1093][::2]
print(data1)

import numpy as np
from haversine import haversine
# 计算每一对经纬度之间的距离，并将结果保存到一个列表中
distances = []
for coord1, coord2 in zip(data1, data2):
    distance = haversine((coord1[0], coord1[1]), (coord2[0], coord2[1]))
    distances.append(distance)
print(max(distances))
# 计算平均距离
average_distance = np.mean(distances)

print(f"平均误差距离为: {average_distance} 公里")

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

# 设置地图
plt.figure(figsize=(10/1.2, 6/1.2))
m = Basemap(projection='merc', llcrnrlat=-10, urcrnrlat=50, llcrnrlon=-70, urcrnrlon=-15, resolution='l')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='lightgray', lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

# 绘制第一个台风路径和标记点
x1, y1 = m(data1[:,1], data1[:,0])
m.plot(x1, y1, 'r-', markersize=3, linewidth=1.5, label='Observation values')
for i in range(len(x1)):
    m.plot(x1[i], y1[i], 'ro', markersize=3)  # 每个点做红色圆点标记

# 绘制第二个台风路径和标记点
x2, y2 = m(data2[:,1], data2[:,0])
m.plot(x2, y2, 'b-', markersize=3, linewidth=1.5, label='24-hour prediction values')
for i in range(len(x2)):
    m.plot(x2[i], y2[i], 'bo', markersize=3)  # 每个点做蓝色圆点标记

# 添加图例和标题
plt.legend(loc='lower left', fontsize=11)
# plt.title('Comparison of Two Typhoon Paths with Marked Points', fontsize=11)
#plt.savefig("48trophoonIRMA2.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
# plt.savefig("48trophoonLAURA.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
#plt.savefig("24trophoonMARIA.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
plt.savefig("24trophoonMELISSA.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
# plt.savefig("48trophoonIRMA.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
# 显示图像
plt.show()