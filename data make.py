
import xarray as xr
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc

def vort_function(longitude, latitude, u, v):
    MAX=1000 # maximum iteration (corresponding eps: 1e-7)
    epsilon=1e-5 # precision
    sor_index=0.2
    N=len(longitude)
    M=len(latitude)
    chi=np.zeros((M,N)) #initialization
    Res=np.ones((M,N))*(-9999)
    dx,dy=mpcalc.lat_lon_grid_deltas(longitude, latitude)
#     print(dx.shape,dy.shape)
    divh=mpcalc.divergence(u, v,dx=dx,dy=dy)
    divh=np.array(divh)
    dxx=np.array(dx)
    dyy=np.array(dy)
    for k in range(1000):
        for i in range(1,M-1):
            for j in range(1,N-1):
                Res[i, j]=(chi[i+1, j]+chi[i-1, j]-2*chi[i, j])/(dxx[i, j-1]*dxx[i, j])+(chi[i, j+1]+chi[i, j-1]-2*chi[i, j])/(dyy[i-1, j]*dyy[i, j])+divh[i, j]
                chi[i, j]=chi[i, j]+(1+sor_index)*Res[i, j]/(2/(dxx[i, j-1]*dxx[i, j])+2/(dyy[i-1, j]*dyy[i, j]))
#         print(k)
        if np.max(np.max(Res))<epsilon:
            break   #Terminate the loop
    chi=chi*units.meters*units.meters/units.seconds
    grad = mpcalc.gradient(chi,deltas=(dy,dx))
    Upsi=np.array(-grad[1])
    Vpsi=np.array(-grad[0])
    return Upsi,Vpsi


def stream_function(longitude, latitude, u, v):
    MAX=1000 # maximum iteration (corresponding eps: 1e-7)
    epsilon=1e-5 # precision
    sor_index=0.2
    N=len(longitude)
    M=len(latitude)
    psi=np.zeros((M,N)) #initialization
    Res=np.ones((M,N))*(-9999)
#     Res=Res/units.second
    dx,dy=mpcalc.lat_lon_grid_deltas(longitude, latitude)
    curlz=mpcalc.vorticity(u, v,dx=dx,dy=dy)
#     curlz_absolute=mpcalc.absolute_vorticity(u, v,dx=dx,dy=dy,latitude=latitude)
#     curlz_relative=curlz+curlz_absolute
    curlz=np.array(curlz)
    dxx=np.array(dx)
    dyy=np.array(dy)
    for k in range(1000):
        for i in range(1,M-1):
            for j in range(1,N-1):
                Res[i, j]=(psi[i+1, j]+psi[i-1, j]-2*psi[i, j])/(dxx[i, j-1]*dxx[i, j])+(psi[i, j+1]+psi[i, j-1]-2*psi[i, j])/(dyy[i-1, j]*dyy[i, j])-curlz[i, j]
                psi[i, j]=psi[i, j]+(1+sor_index)*Res[i, j]/(2/(dxx[i, j-1]*dxx[i, j])+2/(dyy[i-1, j]*dyy[i, j]))
#         print(k)
        if np.max(np.max(Res))<epsilon:
            break   #Terminate the loop
    #vorticity wind
    psi=psi*units.meters*units.meters/units.seconds
    grad = mpcalc.gradient(psi,deltas=(dy,dx))
    Vpsi=np.array(grad[1])
    Upsi=np.array(-grad[0])
    return Upsi,Vpsi




# 加载数据
your_data = xr.open_dataset('D:\latlon\downloadswhbb1980-2021_file.nc')
lat_swh = your_data['latitude'].values
lon_swh = your_data['longitude'].values
ds_u10 = xr.open_dataset('D:\latlon/downloaduvbb1980-2021_file.nc')
# ds_mslp = xr.open_dataset('E:/MSLP1980-2021_file.nc')
# print(ds_mslp)



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
#
#
# import xarray as xr
# from scipy.interpolate import griddata
# import numpy as np
#
# 加载数据
your_data = xr.open_dataset('D:\latlon/downloadswhbb1980-2021_file.nc')
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
# data_track.to_csv(r'./IBTrACS_dropswh0.txt', index=False, header=False)
#
#
#
#
# data_track=pd.read_csv(r'./IBTrACS_dropswh0.txt',sep=',',header=None,names=['name','date','lat','lon','ws','p','speed','direct'])
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
# data_track.to_csv(r'./IBTrACS_dropswh0.txt',index=False,header=False)
# # #
# # #
# # # #
# data_track=pd.read_csv(r'./IBTrACS_dropswh0.txt',sep=',',header=None,names=['name','date','lat','lon','ws','p','speed','direct'])
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
# data_track.to_csv(r'./IBTrACS_dropswh1.txt',index=False,header=False)
# # #
# data_track=pd.read_csv(r'./IBTrACS_dropswh1.txt',sep=',',header=None,names=['name','date','lat','lon','ws','p','speed','direct'])
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
# data_track.to_csv(r'./IBTrACS_dropswh2.txt',index=False,header=False)

#
#代码

import numpy as np
import pandas as pd
import xarray as xr

# 读取 xarray 数据集
import numpy as np
import pandas as pd
import xarray as xr

# 读取 xarray 数据集
# your_data = xr.open_dataset('final.nc')

# your_data = xr.open_dataset('E:\swh2000-2018_file.nc')
# ds_u10 = xr.open_dataset('E:/uv2000-2018_file.nc')
data_track = pd.read_csv(r'./IBTrACS_dropswh2.txt', sep=',', header=None,
                         names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
index_i = []

# 找到包含'66666'的行的索引
for i in range(len(data_track)):
    if data_track.name[i] == '66666':
        index_i.append(i)

# 去掉不在指定经纬度范围内的行
drop_index = []

X = []
X2 = []
y = []
for i in range(len(index_i)):
    m = index_i[i] + 1
    if i == len(index_i) - 1:
        n = len(data_track)
    else:
        n = index_i[i + 1]

    num = 0

    # 检查经纬度范围
    for j in range(m, n-2):
        lat = data_track.lat[j]
        lon = data_track.lon[j]
        lat1 = data_track.lat[j+2]
        lon1 = data_track.lon[j+2]
        hurricane_time = datetime642(data_track.date[j+2])

        #
        lat_idx = round((your_data['latitude'].values[0] - lat) )
        lon_idx = round((lon - your_data['longitude'].values[0]) )

        lat_min =your_data['latitude'].values[lat_idx]-8
        lat_max = your_data['latitude'].values[lat_idx]+8
        lon_min = your_data['longitude'].values[lon_idx]-12
        lon_max = your_data['longitude'].values[lon_idx]+12
        #new
        # lat_min =lat-2
        # lat_max = lat+2
        # lon_min = lon-2
        # lon_max = lon+2
        #

        swh_data1 = your_data.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
        n =( lat_max-swh_data1['latitude'].values[0] )
        m = (swh_data1['longitude'].values[0]-lon_min)


        lat1_idx = (lat_max - lat1)
        if lat1>swh_data1['latitude'].values[0] or lat1<swh_data1['latitude'].values[-1] :
            print('lat1',lat1,'lat',lat)
            print('false')
        lon1_idx = (lon1 - lon_min)
        if lon1<swh_data1['longitude'].values[0] or lon1>swh_data1['longitude'].values[-1] :
            print('lon1', lon1, 'lat', lon)
            print('false')
        swh_data = swh_data1['swh'].sel(time=hurricane_time).values


        u10 = ds_u10['u10'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
        u10_data = u10.sel(time=hurricane_time).fillna(0)

        v10 = ds_u10['v10'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
        v10_data = v10.sel(time=hurricane_time).fillna(0)



        ud, vd = vort_function(u10.longitude.values, u10.latitude.values, u10_data.values, v10_data.values)
        ur, vr = stream_function(u10.longitude, u10.latitude, u10_data, v10_data)
        ug = u10_data.values - ud - ur
        vg = v10_data.values - vd - vr

        swh_data = np.nan_to_num(swh_data, nan=0)





        # mslp_data = ds_mslp['msl'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
        # mslp_data = mslp_data.sel(time=hurricane_time).values

        # mslp_data = np.nan_to_num(mslp_data, nan=0)
        rows = len(swh_data)
        cols = len(swh_data[0])

        # for i in range(rows):
        #     for j in range(cols):
        #         if swh_data[i][j] == 0:
        #             ug[i][j] = 0
        #             vg[i][j]=0
        #             # mslp_data[i][j] = 0
        data = np.stack((ug, vg, swh_data), axis=2)
        n = int(n)
        m = int(m)

        if data.shape[0] != 17 or data.shape[1] != 25:
            # 创建一个新的17x17数组，用0填充
            new_data = np.zeros((17, 25, data.shape[2]))

            # # 计算要填充的行和列的范围
            # row_min = (17 - data.shape[0]) // 2
            # row_max = row_min + data.shape[0]
            # col_min = (17 - data.shape[1]) // 2
            # col_max = col_min + data.shape[1]

            # 将原始数据复制到新数组中
            new_data[n:data.shape[0]+n, m:data.shape[1]+m, :] = data

            # 更新data为新填充的数组
            data = new_data
        if data.shape[0] == 17 and data.shape[1] == 25:
            X.append(data)
            X2.append((lat_max,lon_min))
            y.append((lat1_idx, lon1_idx))







# 将 X 和 y 转换为 numpy 数组
# X = np.array(X)
# y = np.array(y)
# print(X.shape)
# 现在，X 包含了波高数据，y 是对应的经纬度在网格中的索引
# 这只是一个示例代码，需要根据实际数据和需求进行调整


# 现在，X 包含了波高数据，y 是对应的经纬度在网格中的索引
# 这只是一个示例代码，需要根据实际数据和需求进行调整

#
np.save('X_data_big.npy', X)
np.save('y_data_big.npy', y)
np.save('origan_col.npy',X2)