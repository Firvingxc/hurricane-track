import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import griddata
import numpy as np
def datetime64(date_time):
    datetime = np.datetime64(date_time, 's')
    datetime_64 = pd.to_datetime(datetime)
    date = datetime_64.strftime('%Y%m%d%H')
    date = int(date)
    return date


def datetime642(date_time):
    datetime = np.datetime64(date_time, 's')
    datetime_64 = pd.to_datetime(datetime)
    date = datetime_64.strftime('%Y-%m-%d %H:%M:%S')
    # date = int(date)
    return date

# your_data = xr.open_dataset('D:\latlon/downloadswhbb1980-2021_file.nc')
# data_track = pd.read_csv(r'./IBTrACS_droplatlong_big2.txt', sep=',', header=None,
#                          names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
# index_i = []
#
# # 找到包含'66666'的行的索引
# for i in range(len(data_track)):
#     if data_track.name[i] == '66666':
#         index_i.append(i)
#
# # 去掉不在海洋区域的
# drop_index = []
# for i in range(len(index_i)):
#     m = index_i[i] + 1
#     if i == len(index_i) - 1:
#         n = len(data_track)
#     else:
#         n = index_i[i + 1]
#
#     num = 0
#
#     # 检查经纬度范围
#     for j in range(m, n):
#         lat = data_track.lat[j]
#         lon = data_track.lon[j]
#         hurricane_time = datetime642(data_track.date[j])
#         swh_data = your_data['swh'].sel(time=hurricane_time).values
#         swh_data = np.nan_to_num(swh_data, nan=0)
#         lat_idx = round((your_data['latitude'].values[0] - lat) )
#         lon_idx = round((lon - your_data['longitude'].values[0]) )
#
#
#         if swh_data[lat_idx, lon_idx] ==0:
#             num = num+1
#             drop_index.append(j)
#
#
#     # 如果这一段范围内的所有行都不在指定范围内，将起始索引记录到drop_index列表中
#     if num == n - m:
#         drop_index.append(m - 1)
#
# # 删除不在指定经纬度范围内的行
# for i in range(len(drop_index)):
#     data_track = data_track.drop(drop_index[i])
#
# data_track.to_csv(r'./IBTrACS_drawswh.txt', index=False, header=False)
#
#
#
#
# data_track=pd.read_csv(r'./IBTrACS_drawswh.txt',sep=',',header=None,names=['name','date','lat','lon','ws','p','speed','direct'])
# index_i=[]
# for i in range(len(data_track)):
#     if data_track.name[i]=='66666':
#         index_i.append(i)
#
#
#
# #去掉时间上不连续的
# i_remove=[]
# for i in range(len(index_i)):
#     m=index_i[i]+1
#     if i ==len(index_i)-1:
#         n=len(data_track)
#     else:
#         n=index_i[i+1]
#     sel_data=[]
#     length_mid=int((n-m)/2)
#     for l in range(m,n-1):
#         date_time0=data_track.date[l]
#         date_time1=data_track.date[l+1]
#         date0=datetime64(date_time0)
#         date1=datetime64(date_time1)
#         datetime0=pd.to_datetime(date0,format='%Y%m%d%H')
#         datetime1=pd.to_datetime(date0,format='%Y%m%d%H')
#         date0=datetime64(date_time0)
#         date_int0=int(date0)
#         date_int1=int(date1)
#         if date_int1-date_int0==3 or date_int1-date_int0==79 or date_int1-date_int0==7079 or date_int1-date_int0==7179 or date_int1-date_int0==6979 or date_int1-date_int0==886979:
#             j=i+1
#         else:
#             if l<=length_mid:
#                 i_remove=i_remove+[h for h in range(m,l+2)]
#             else:
#                 i_remove=i_remove+[h for h in range(l,n)]
# i_remove_list=list(set(i_remove))
# for i in range(len(i_remove_list)):
#     data_track=data_track.drop(i_remove_list[i])
# data_track.to_csv(r'./IBTrACS_drawswh.txt',index=False,header=False)
# # #
# # #
# # # #
# data_track=pd.read_csv(r'./IBTrACS_drawswh.txt',sep=',',header=None,names=['name','date','lat','lon','ws','p','speed','direct'])
# index_i=[]
# for i in range(len(data_track)):
#     if data_track.name[i]=='66666':
#         index_i.append(i)
#
#
# #去掉生命史小于96个小时
# remove=[]
# for i in range(len(index_i)):
#     m=index_i[i]+1
#     if i==len(index_i)-1:
#         n=len(data_track)
#     else:
#         n=index_i[i+1]
#     if n-m<32:
#         for j in range(m-1,n):
#             remove.append(j)
# for i in range(len(remove)):
#     data_track=data_track.drop(remove[i])
# data_track.to_csv(r'./IBTrACS_drawswh.txt',index=False,header=False)
# # #
# data_track=pd.read_csv(r'./IBTrACS_drawswh.txt',sep=',',header=None,names=['name','date','lat','lon','ws','p','speed','direct'])
# index_i=[]
# for i in range(len(data_track)):
#     if data_track.name[i]=='66666':
#         index_i.append(i)
#
# i_remove=[]
# for i in range(len(index_i)):
#     m=index_i[i]+1
#     if i ==len(index_i)-1:
#         n=len(data_track)
#     else:
#         n=index_i[i+1]
#     sel_data=[]
#     for l in range(m,n):
#         lat=data_track.lat[l]
#         lon=data_track.lon[l]
#         date_time=data_track.date[l]
#         date=datetime64(date_time)
#         datetime=pd.to_datetime(date,format='%Y%m%d%H')
#         date_str=str(date)
#         year=datetime.year
#         # if date_str[8:10] == '06' or date_str[8:10] == '12' or date_str[8:10] == '18' or date_str[8:10] == '00' or date_str[8:10] == '03' or date_str[8:10] == '09' or date_str[
#         #     8:10] == '15' or date_str[8:10] == '21':
#         if date_str[8:10]=='06'or date_str[8:10]=='12'or date_str[8:10]=='18'or date_str[8:10]=='00':
#             j=i+1
#         else:
#             i_remove.append(l)
# for i in range(len(i_remove)):
#     data_track=data_track.drop(i_remove[i])
# data_track.to_csv(r'./IBTrACS_drawswh.txt',index=False,header=False)
#



# 读取数据
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#
# # 读取数据
# swh_data = xr.open_dataset('D:/latlon/downloadswhbb1980-2021_file.nc')
# data_track = pd.read_csv(r'./IBTrACS_dropswh2.txt', sep=',', header=None,
#                          names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
#
# # 找到包含'66666'的行的索引
# index_i = data_track.index[data_track['name'] == '66666'].tolist()
#
# # 初始化总SWH数据
# total_swh = np.zeros((11, 11))
# count = 0
#
# # 遍历每个台风轨迹点
# for idx in index_i:
#     start = idx + 1
#     if idx == index_i[-1]:
#         end = len(data_track)
#     else:
#         end = index_i[index_i.index(idx) + 1]
#
#     for j in range(start, end):
#         lat = data_track.loc[j, 'lat']
#         lon = data_track.loc[j, 'lon']
#         date = datetime642(data_track.date[j])
#
#         # 提取5个经纬度范围内的SWH数据
#         lat_min = lat - 5
#         lat_max = lat + 5
#         lon_min = lon - 5
#         lon_max = lon + 5
#
#         swh_region = swh_data.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
#         swh_time = swh_region.sel(time=date, method='nearest')
#
#         # 获取SWH数据
#         swh_values = swh_time['swh'].values
#         swh_values = np.nan_to_num(swh_values, nan=0)
#
#         # 检查SWH数据是否为11x11的形状
#         if swh_values.shape == (11, 11):
#             total_swh += swh_values
#             print(swh_values)
#             count += 1
#
# # 计算平均SWH
# if count > 0:
#     mean_swh = total_swh / count
#
#     # 获取经纬度坐标
#     lats = np.linspace(-5, 5, 11)
#     lons = np.linspace(-5, 5, 11)
#
#     # 绘制平均SWH波高分布图
#     plt.figure(figsize=(10, 8))
#     plt.contourf(lons, lats, mean_swh, levels=100, cmap='viridis')
#     plt.colorbar(label='SWH (m)')
#     plt.title('Mean SWH Distribution around Typhoon Center')
#     plt.xlabel('Longitude (relative to center)')
#     plt.ylabel('Latitude (relative to center)')
#     plt.grid(True)
#     plt.show()
# else:
#     print("没有符合条件的数据块")

#
# import pandas as pd
# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取数据
swh_data = xr.open_dataset('D:/latlon/downloadswhbb1980-2021_file.nc')
data_track = pd.read_csv(r'./2000-2020.txt', sep=',', header=None,
                         names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])

# 找到包含'66666'的行的索引
index_i = data_track.index[data_track['name'] == '66666'].tolist()

# 初始化总SWH数据
total_swh = np.zeros((11, 11))
count = 0

# 遍历每个台风轨迹点
for idx in index_i:
    start = idx + 1
    if idx == index_i[-1]:
        end = len(data_track)
    else:
        end = index_i[index_i.index(idx) + 1]

    for j in range(start, end):
        lat = data_track.loc[j, 'lat']
        lon = data_track.loc[j, 'lon']
        date = datetime642(data_track.date[j])

        # 提取5个经纬度范围内的SWH数据
        lat_idx = round((swh_data['latitude'].values[0] - lat))
        lon_idx = round((lon - swh_data['longitude'].values[0]))

        lat_min = swh_data['latitude'].values[lat_idx] - 5
        lat_max = swh_data['latitude'].values[lat_idx] + 5
        lon_min = swh_data['longitude'].values[lon_idx] - 5
        lon_max = swh_data['longitude'].values[lon_idx] + 5
        print(lat_max)
        print(lat_min)

        swh_region = swh_data.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
        swh_time = swh_region.sel(time=date, method='nearest')

        n = (lat_max - swh_time['latitude'].values[0])
        m = (swh_time['longitude'].values[0] - lon_min)
        print(swh_time)

        # 获取SWH数据
        swh_values = swh_time['swh'].values
        swh_values = np.nan_to_num(swh_values)


        swh_values = abs(swh_values)

        n = int(n)
        m = int(m)
        print(n)

        # 确保SWH数据为11x11的形状
        if swh_values.shape != (11, 11):
            new_data = np.zeros((11, 11))

            # # 计算要填充的行和列的范围
            # row_min = (17 - data.shape[0]) // 2
            # row_max = row_min + data.shape[0]
            # col_min = (17 - data.shape[1]) // 2
            # col_max = col_min + data.shape[1]

            # 将原始数据复制到新数组中
            new_data[n:swh_values.shape[0] + n, m:swh_values.shape[1] + m] = swh_values
            swh_values = new_data

            # lats = np.linspace(5, -5, 11)
            # lons = np.linspace(-5, 5, 11)
            # # 将小于2米的值设置为NaN
            # # mean_swh[mean_swh < 2.5] = np.nan
            #
            # # 绘制平均SWH波高分布图
            # plt.figure(figsize=(10, 8))
            # contour = plt.contourf(lons, lats, swh_values, levels=1000, cmap='coolwarm', extend='both')
            # # plt.colorbar(contour, label='swh (m)')
            # # # plt.title('Mean SWH Distribution around Typhoon Center')
            # # plt.xlabel('Lon (°)')
            # # plt.ylabel('Lat (°)')
            # # plt.grid(True)
            # plt.axis('off')
            # plt.show()

        total_swh += np.nan_to_num(swh_values)
        count += 1

# 计算平均SWH
mean_swh = total_swh / count

# 获取经纬度坐标
lats = np.linspace(5, -5, 11)
lons = np.linspace(-5, 5, 11)
# 将小于2米的值设置为NaN
# mean_swh[mean_swh < 2.5] = np.nan
vmin = 1.5
vmax = 3.6
levels = np.linspace(vmin, vmax, 500)
# 绘制平均SWH波高分布图
plt.figure(figsize=(4.135, 4.135))
plt.rc('font', family='Times New Roman', size=11)
contour = plt.contourf(lons, lats, mean_swh, levels=levels, cmap='coolwarm', extend='both')
cbar = plt.colorbar(contour)
cbar.set_label('swh (m)')
plt.savefig('2000-2020tu.tif', dpi=600,format='tif', bbox_inches='tight')
# plt.colorbar(contour, label='swh (m)')
# # plt.title('Mean SWH Distribution around Typhoon Center')
# plt.xlabel('Lon (°)')
# plt.ylabel('Lat (°)')
# plt.grid(True)
plt.show()



# import pandas as pd
# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
#
# 读取数据
# swh_data = xr.open_dataset('D:/latlon/downloadswhbb1980-2021_file.nc')
# data_track = pd.read_csv(r'./IBTrACS_dropswh2.txt', sep=',', header=None,
#                          names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
#
# # 找到包含'66666'的行的索引
# index_i = data_track.index[data_track['name'] == '66666'].tolist()
#
# # 初始化总SWH数据
# total_swh = np.zeros((11, 11))
# count = 0
#
# # 定义统一的经纬度网格
# lat_grid = np.linspace(-5, 5, 11)
# lon_grid = np.linspace(-5, 5, 11)
#
# # 遍历每个台风轨迹点
# for idx in index_i:
#     start = idx + 1
#     if idx == index_i[-1]:
#         end = len(data_track)
#     else:
#         end = index_i[index_i.index(idx) + 1]
#
#     for j in range(start, end):
#         lat = data_track.loc[j, 'lat']
#         lon = data_track.loc[j, 'lon']
#         date = datetime642(data_track.date[j])
#
#         # 提取5个经纬度范围内的SWH数据
#         lat_idx = round((swh_data['latitude'].values[0] - lat))
#         lon_idx = round((lon - swh_data['longitude'].values[0]))
#
#         lat_min = swh_data['latitude'].values[lat_idx] - 5
#         lat_max = swh_data['latitude'].values[lat_idx] + 5
#         lon_min = swh_data['longitude'].values[lon_idx] - 5
#         lon_max = swh_data['longitude'].values[lon_idx] + 5
#
#         swh_region = swh_data.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
#
#         swh_time = swh_region.sel(time=date, method='nearest')
#
#         # 创建目标网格
#         target_lat = np.linspace(lat_max, lat_min, 11)
#
#         target_lon = np.linspace(lon_min, lon_max, 11)
#
#         # 重新索引SWH数据到目标网格
#         swh_reindexed = swh_time.interp(latitude=target_lat, longitude=target_lon)
#
#
#         # 获取重新索引后的SWH数据
#         swh_values = swh_reindexed['swh'].values
#
#         swh_values = np.nan_to_num(swh_values)
#
#         total_swh += swh_values
#         count += 1
#
# # 计算平均SWH
# if count > 0:
#     mean_swh = total_swh / count
#
#     # 绘制平均SWH波高分布图
#     plt.figure(figsize=(10, 8))
#     plt.contourf(lon_grid, lat_grid, mean_swh, levels=20, cmap='viridis')
#     plt.colorbar(label='SWH (m)')
#     plt.title('Mean SWH Distribution around Typhoon Center')
#     plt.xlabel('Longitude (relative to center)')
#     plt.ylabel('Latitude (relative to center)')
#     plt.grid(True)
#     plt.show()
# else:
#     print("没有符合条件的数据块")
