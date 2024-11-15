import xarray as xr
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

height = 17
width = 25
num_classes = 2

def datetime642(date_time):
    datetime = np.datetime64(date_time, 's')
    datetime_64 = pd.to_datetime(datetime)
    date = datetime_64.strftime('%Y-%m-%d %H:%M:%S')
    # date = int(date)
    return date


def datetime64(date_time):
    datetime = np.datetime64(date_time, 's')
    datetime_64 = pd.to_datetime(datetime)
    date = datetime_64.strftime('%Y%m%d%H')
    date = int(date)
    return date


def vort_function(longitude, latitude, u, v):
    MAX = 1000  # maximum iteration (corresponding eps: 1e-7)
    epsilon = 1e-5  # precision
    sor_index = 0.2
    N = len(longitude)
    M = len(latitude)
    chi = np.zeros((M, N))  # initialization
    Res = np.ones((M, N)) * (-9999)
    dx, dy = mpcalc.lat_lon_grid_deltas(longitude, latitude)
    #     print(dx.shape,dy.shape)
    divh = mpcalc.divergence(u, v, dx=dx, dy=dy)
    divh = np.array(divh)
    dxx = np.array(dx)
    dyy = np.array(dy)
    for k in range(1000):
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                Res[i, j] = (chi[i + 1, j] + chi[i - 1, j] - 2 * chi[i, j]) / (dxx[i, j - 1] * dxx[i, j]) + (
                            chi[i, j + 1] + chi[i, j - 1] - 2 * chi[i, j]) / (dyy[i - 1, j] * dyy[i, j]) + divh[i, j]
                chi[i, j] = chi[i, j] + (1 + sor_index) * Res[i, j] / (
                            2 / (dxx[i, j - 1] * dxx[i, j]) + 2 / (dyy[i - 1, j] * dyy[i, j]))
        #         print(k)
        if np.max(np.max(Res)) < epsilon:
            break  # Terminate the loop
    chi = chi * units.meters * units.meters / units.seconds
    grad = mpcalc.gradient(chi, deltas=(dy, dx))
    Upsi = np.array(-grad[1])
    Vpsi = np.array(-grad[0])
    return Upsi, Vpsi


def stream_function(longitude, latitude, u, v):
    MAX = 1000  # maximum iteration (corresponding eps: 1e-7)
    epsilon = 1e-5  # precision
    sor_index = 0.2
    N = len(longitude)
    M = len(latitude)
    psi = np.zeros((M, N))  # initialization
    Res = np.ones((M, N)) * (-9999)
    #     Res=Res/units.second
    dx, dy = mpcalc.lat_lon_grid_deltas(longitude, latitude)
    curlz = mpcalc.vorticity(u, v, dx=dx, dy=dy)
    #     curlz_absolute=mpcalc.absolute_vorticity(u, v,dx=dx,dy=dy,latitude=latitude)
    #     curlz_relative=curlz+curlz_absolute
    curlz = np.array(curlz)
    dxx = np.array(dx)
    dyy = np.array(dy)
    for k in range(1000):
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                Res[i, j] = (psi[i + 1, j] + psi[i - 1, j] - 2 * psi[i, j]) / (dxx[i, j - 1] * dxx[i, j]) + (
                            psi[i, j + 1] + psi[i, j - 1] - 2 * psi[i, j]) / (dyy[i - 1, j] * dyy[i, j]) - curlz[i, j]
                psi[i, j] = psi[i, j] + (1 + sor_index) * Res[i, j] / (
                            2 / (dxx[i, j - 1] * dxx[i, j]) + 2 / (dyy[i - 1, j] * dyy[i, j]))
        #         print(k)
        if np.max(np.max(Res)) < epsilon:
            break  # Terminate the loop
    # vorticity wind
    psi = psi * units.meters * units.meters / units.seconds
    grad = mpcalc.gradient(psi, deltas=(dy, dx))
    Vpsi = np.array(grad[1])
    Upsi = np.array(-grad[0])
    return Upsi, Vpsi


#Unet
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # Use bilinear upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust size for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 64)
        self.up1 = Up(192, 32, bilinear)  # Adjust in_channels to match concatenated channels
        self.up2 = Up(96, 12, bilinear)
        self.up3 = Up(44, 6, bilinear)
        self.outc = nn.Conv2d(6, n_classes, kernel_size=1)
        self.fc = nn.Linear(n_classes * 17 * 25, 2)  # Output 2 coordinates

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)

        output = self.fc(logits.view(logits.size(0), -1))
        return output


# 假设已经加载和预处理了X和y
# 归一化坐标
def normalize_coordinates(y, width, height):
    y_normalized = np.zeros_like(y, dtype=np.float32)
    y_normalized[:, 0] = y[:, 0] / (width - 1)
    y_normalized[:, 1] = y[:, 1] / (height - 1)
    return y_normalized


# 反归一化坐标
def denormalize_coordinates(y_normalized, width, height):
    y_denormalized = np.zeros_like(y_normalized, dtype=np.float32)
    y_denormalized[:, 0] = y_normalized[:, 0] * (width - 1)
    y_denormalized[:, 1] = y_normalized[:, 1] * (height - 1)
    return y_denormalized

# 环境数据 - 替换成预测数据
#
# your_data = xr.open_dataset('downloadswhbb1980-2021_file.nc')
# # # # lat_swh = your_data['latitude'].values
# # # # lon_swh = your_data['longitude'].values
# # # # print(lat_swh)
# # # # print(lon_swh)
# ds_u10 = xr.open_dataset('downloaduvbb1980-2021_file.nc')

# #
your_data = xr.open_dataset('du1_swh2017-2020.nc')
lat_swh = your_data['latitude'].values
print(lat_swh)
lon_swh = your_data['longitude'].values
print(lon_swh)
ds_u10 = xr.open_dataset('du1_uv2017-2020_24z.nc').sel(latitude = lat_swh,longitude = lon_swh)

data_track=pd.read_csv(r'./testall22.txt',sep=',',header=None,names=['name','date','lat','lon','ws','p','speed','direct'])
index_i=[]
for i in range(len(data_track)):
    if data_track.name[i]=='66666':
        index_i.append(i)

i_remove=[]
for i in range(len(index_i)):
    m=index_i[i]+1
    if i ==len(index_i)-1:
        n=len(data_track)
    else:
        n=index_i[i+1]
    sel_data=[]
    for l in range(m,n):
        lat=data_track.lat[l]
        lon=data_track.lon[l]
        date_time=data_track.date[l]
        date=datetime64(date_time)
        datetime=pd.to_datetime(date,format='%Y%m%d%H')
        date_str=str(date)
        year=datetime.year
        # if date_str[8:10] == '06' or date_str[8:10] == '12' or date_str[8:10] == '18' or date_str[8:10] == '00' or date_str[8:10] == '03' or date_str[8:10] == '09' or date_str[
        #     8:10] == '15' or date_str[8:10] == '21':
        if  date_str[8:10]=='12'or  date_str[8:10]=='00':
            j=i+1
        else:
            i_remove.append(l)
for i in range(len(i_remove)):
    data_track=data_track.drop(i_remove[i])
data_track.to_csv(r'./testall22.txt',index=False,header=False)
#load model

# class DoubleConv(nn.Module):
#     """(Convolution => [BN] => ReLU) * 2"""
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
#
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
# class Up(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#         # Use bilinear upsampling
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # Adjust size for concatenation
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 32)
#         self.down1 = Down(32, 64)
#         self.down2 = Down(64, 128)
#         self.down3 = Down(128, 64)
#         self.up1 = Up(192, 32, bilinear)  # Adjust in_channels to match concatenated channels
#         self.up2 = Up(96, 12, bilinear)
#         self.up3 = Up(44, 6, bilinear)
#         self.outc = nn.Conv2d(6, n_classes, kernel_size=1)
#         self.fc = nn.Linear(n_classes * 17 * 25, 2)  # Output 2 coordinates
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         logits = self.outc(x)
#
#         output = self.fc(logits.view(logits.size(0), -1))
#         return output
#
# #Unet
# class DoubleConv(nn.Module):
#     """(Convolution => [BN] => ReLU) * 2"""
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
#
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
# class Up(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#         # Use bilinear upsampling
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # Adjust size for concatenation
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 32)
#         self.down1 = Down(32, 64)
#         self.down2 = Down(64, 128)
#         self.down3 = Down(128, 128)
#         self.up1 = Up(256, 64, bilinear)  # Adjust in_channels to match concatenated channels
#         self.up2 = Up(128, 32, bilinear)
#         self.up3 = Up(64, 32, bilinear)
#         self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
#         self.fc = nn.Linear(n_classes * 17 * 25, 2)  # Output 2 coordinates
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         logits = self.outc(x)
#
#         output = self.fc(logits.view(logits.size(0), -1))
#         return output
#
# model = UNet(3,2)
# model.load_state_dict(torch.load('model_test.pth'))
#Unet
class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # Use bilinear upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):



        x1 = self.up(x1)



        # Adjust size for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.up1 = Up(384, 128, bilinear)  # Adjust in_channels to match concatenated channels
        self.up2 = Up(192, 64, bilinear)
        self.up3 = Up(96, 1, bilinear)
        self.outc = nn.Conv2d(1, n_classes, kernel_size=1)
        self.fc = nn.Linear(n_classes * 17 * 25, 2)  # Output 2 coordinates

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) #64
        x4 = self.down3(x3) #128

        x = self.up1(x4, x3)

        x = self.up2(x, x2)

        x = self.up3(x, x1)

        logits = self.outc(x)



        output = self.fc(logits.view(logits.size(0), -1))
        return output

model = UNet(3,1)
model.load_state_dict(torch.load('model_test3.pth'))


model.eval()
#预报

data_track = pd.read_csv(r'./testall22.txt', sep=',', header=None,
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

result = []
oooku = []
for i in range(len(index_i)):
    start = index_i[i] + 1
    print(start)
    if i == len(index_i) - 1:
        end1 = len(data_track)
    else:
        end1 = index_i[i + 1]

    num = 0
    count = 2
    print(end1)



    j = start

    while j <=end1-2:

        your_data = xr.open_dataset('du1_swh2017-2020.nc')


        ds_u10 = xr.open_dataset('du1_uv2017-2020_24znew_6.nc').sel(latitude=lat_swh, longitude=lon_swh)

        if count == 4:
            your_data = xr.open_dataset('du1_swh2017-2020_48.nc')
            ds_u10 = xr.open_dataset('du1_uv2017-2020_48.nc').sel(latitude=lat_swh,
                                                                  longitude=lon_swh)

        if count == 6:
            your_data = xr.open_dataset('du1_swh2017-2020_72.nc')
            ds_u10 = xr.open_dataset('du1_uv2017-2020_72.nc').sel(latitude=lat_swh, longitude=lon_swh)




        lat = data_track.lat[j]
        lon = data_track.lon[j]
        lat1 = data_track.lat[j + 1]
        lon1 = data_track.lon[j + 1]
        hurricane_time = datetime642(data_track.date[j + 1])

        #
        # lat_idx = round((your_data['latitude'].values[0] - lat) )
        # lon_idx = round((lon - your_data['longitude'].values[0]))
        lat_idx = np.abs(your_data['latitude'].values - lat).argmin()
        lon_idx = np.abs(your_data['longitude'].values - lon).argmin()

        lat_min = your_data['latitude'].values[lat_idx] - 8
        lat_max = your_data['latitude'].values[lat_idx] + 8
        lon_min = your_data['longitude'].values[lon_idx] - 12
        lon_max = your_data['longitude'].values[lon_idx] + 12
        # new
        # lat_min =lat-2
        # lat_max = lat+2
        # lon_min = lon-2
        # lon_max = lon+2
        #

        swh_data1 = your_data.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
        n = (lat_max - swh_data1['latitude'].values[0])
        m = (swh_data1['longitude'].values[0] - lon_min)

        lat1_idx = (lat_max - lat1)
        if lat1 > swh_data1['latitude'].values[0] or lat1 < swh_data1['latitude'].values[-1]:
            print('false')
        lon1_idx = (lon1 - lon_min)
        if lon1 < swh_data1['longitude'].values[0] or lon1 > swh_data1['longitude'].values[-1]:
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

        rows = len(swh_data)
        cols = len(swh_data[0])

        data = np.stack((ug, vg, swh_data), axis=2)
        n = int(n)
        m = int(m)


        if data.shape[0] != 17 or data.shape[1] != 25:
            # 创建一个新的17x17数组，用0填充
            new_data = np.zeros((17, 25, data.shape[2]))

            new_data[n:data.shape[0] + n, m:data.shape[1] + m, :] = data

            data = new_data
        if data.shape[0] == 17 and data.shape[1] == 25:
            X.append(data)
            # X2.append((lat_max, lon_min))
            y.append((lat1_idx, lon1_idx))

        origantest = np.array((lat_max, lon_min))
        One_p = data
        scaler = joblib.load('scaler.pkl')

        One_p = One_p.reshape(-1, 3)
        One_p = scaler.transform(One_p)
        One_p = One_p.reshape(1, 17, 25, 3)
        One_p = One_p.transpose(0, 3, 1, 2)
        One_p = torch.Tensor(One_p)




        for u in range(count):
            if j < end1 - 1:
                print(u)
                print(j)
                X2.append((data_track.lat[j + 1], data_track.lon[j + 1]))
                with torch.no_grad():
                    outputs = model(One_p)

                your_data = xr.open_dataset('du1_swh2017-2020.nc')
                ds_u10 = xr.open_dataset('du1_uv2017-2020_24znew_6.nc').sel(latitude=lat_swh,
                                                                      longitude=lon_swh)



                if count == 4:
                    your_data = xr.open_dataset('du1_swh2017-2020_48.nc')
                    ds_u10 = xr.open_dataset('du1_uv2017-2020_48.nc').sel(latitude=lat_swh,
                                                                          longitude=lon_swh)

                if count == 6:
                    your_data = xr.open_dataset('du1_swh2017-2020_72.nc')
                    ds_u10 = xr.open_dataset('du1_uv2017-2020_72.nc').sel(latitude=lat_swh, longitude=lon_swh)
                # # your_data = xr.open_dataset('E:\downloads\modified_three_hourly_swh2020.nc')
                # lat_swh = your_data['latitude'].values
                #
                # lon_swh = your_data['longitude'].values
                #
                # ds_u10 = xr.open_dataset('E:/downloads\modified_three_hourly_uv2020.nc').sel(latitude=lat_swh,
                #                                                                              longitude=lon_swh)
                outputs = outputs.numpy()

                result.append(outputs[0])
                oooku.append(origantest)
                predictions_denorm = denormalize_coordinates(outputs, width, height)

                predictions_denorm[0, 0] = origantest[0] - predictions_denorm[0, 0]
                predictions_denorm[0, 1] = origantest[1] + predictions_denorm[0, 1]

                predictions_denorm = np.array(predictions_denorm)
                predictions_denorm = np.array(predictions_denorm)

                lat = predictions_denorm[0, 0]
                lon = predictions_denorm[0, 1]

                j = j + 1
            # print(u)
            # print(j)
            # X2.append((data_track.lat[j+1], data_track.lon[j+1]))
            # with torch.no_grad():
            #     outputs = model(One_p)
            #
            # if count==4:
            #     your_data = xr.open_dataset('du1_swh2017-2020_48.nc')
            #     ds_u10 = xr.open_dataset('du1_uv2017-2020_48.nc').sel(latitude=lat_swh,
            #                                                           longitude=lon_swh)
            #
            # if count==6:
            #     your_data = xr.open_dataset('du1_swh2017-2020_72.nc')
            #     ds_u10 = xr.open_dataset('du1_uv2017-2020_72.nc').sel(latitude=lat_swh, longitude=lon_swh)
            # # # your_data = xr.open_dataset('E:\downloads\modified_three_hourly_swh2020.nc')
            # # lat_swh = your_data['latitude'].values
            # #
            # # lon_swh = your_data['longitude'].values
            # #
            # # ds_u10 = xr.open_dataset('E:/downloads\modified_three_hourly_uv2020.nc').sel(latitude=lat_swh,
            # #                                                                              longitude=lon_swh)
            # outputs = outputs.numpy()
            #
            #
            # result.append(outputs[0])
            # oooku.append(origantest)
            # predictions_denorm = denormalize_coordinates(outputs, width, height)
            #
            # predictions_denorm[0, 0] = origantest[0] - predictions_denorm[0, 0]
            # predictions_denorm[0, 1] = origantest[1] + predictions_denorm[0, 1]
            #
            # predictions_denorm = np.array(predictions_denorm)
            # predictions_denorm = np.array(predictions_denorm)
            #
            # lat = predictions_denorm[0, 0]
            # lon = predictions_denorm[0, 1]
            #
            # j= j+1

            if j < end1 -1:
                print(end1-1)
                print(j)
                print(data_track.date[j + 1])

                time = datetime642(data_track.date[j + 1])
                # lat_idx = round((your_data['latitude'].values[0] - lat))
                # lon_idx = round((lon - your_data['longitude'].values[0]))
                lat_idx = np.abs(your_data['latitude'].values - lat).argmin()
                lon_idx = np.abs(your_data['longitude'].values - lon).argmin()

                lat_min = your_data['latitude'].values[lat_idx] - 8
                lat_max = your_data['latitude'].values[lat_idx] + 8
                lon_min = your_data['longitude'].values[lon_idx] - 12
                lon_max = your_data['longitude'].values[lon_idx] + 12

                swh_data1 = your_data.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
                n = (lat_max - swh_data1['latitude'].values[0])
                m = (swh_data1['longitude'].values[0] - lon_min)

                swh_data = swh_data1['swh'].sel(time=time).values

                u10 = ds_u10['u10'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
                u10_data = u10.sel(time=time).fillna(0)

                v10 = ds_u10['v10'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
                v10_data = v10.sel(time=time).fillna(0)

                ud, vd = vort_function(u10.longitude.values, u10.latitude.values, u10_data.values, v10_data.values)
                ur, vr = stream_function(u10.longitude, u10.latitude, u10_data, v10_data)

                ug = u10_data.values - ud - ur
                vg = v10_data.values - vd - vr
                # print(vg.shape)

                # mslp_data = ds_mslp['msl'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
                # mslp_data = mslp_data.sel(time=hurricane_time).values
                swh_data = np.nan_to_num(swh_data, nan=0)

                # mslp_data = np.nan_to_num(mslp_data, nan=0)
                rows = len(swh_data)
                cols = len(swh_data[0])

                # for i in range(rows):
                #     for j in range(cols):
                #         if swh_data[i][j] == 0:
                #             ug[i][j] = 0
                #             vg[i][j] = 0
                #             # mslp_data[i][j] = 0
                data = np.stack((ug, vg, swh_data), axis=2)
                n = int(n)
                m = int(m)
                # print(m)
                if data.shape[0] != 17 or data.shape[1] != 25:
                    # 创建一个新的17x17数组，用0填充
                    new_data = np.zeros((17, 25, data.shape[2]))

                    # # 计算要填充的行和列的范围
                    # row_min = (17 - data.shape[0]) // 2
                    # row_max = row_min + data.shape[0]
                    # col_min = (17 - data.shape[1]) // 2
                    # col_max = col_min + data.shape[1]

                    # 将原始数据复制到新数组中
                    new_data[n:data.shape[0] + n, m:data.shape[1] + m, :] = data

                    # 更新data为新填充的数组
                    data = new_data
                if data.shape[0] == 17 and data.shape[1] == 25:
                    One_p = data
                    origantest = (lat_max, lon_min)  # 这个是
                    One_p = One_p.reshape(-1, 3)
                    One_p = scaler.transform(One_p)
                    One_p = One_p.reshape(1, 17, 25, 3)
                    One_p = One_p.transpose(0, 3, 1, 2)
                    One_p = torch.Tensor(One_p)

            # print(data_track.date[j + 1])
            #
            #
            #
            #
            #
            # time = datetime642(data_track.date[j + 1])
            # lat_idx = round((your_data['latitude'].values[0] - lat) )
            # lon_idx = round((lon - your_data['longitude'].values[0]) )
            #
            # lat_min = your_data['latitude'].values[lat_idx] - 8
            # lat_max = your_data['latitude'].values[lat_idx] + 8
            # lon_min = your_data['longitude'].values[lon_idx] - 12
            # lon_max = your_data['longitude'].values[lon_idx] + 12
            #
            # swh_data1 = your_data.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
            # n = (lat_max - swh_data1['latitude'].values[0])
            # m = (swh_data1['longitude'].values[0] - lon_min)
            #
            # swh_data = swh_data1['swh'].sel(time=time).values
            #
            # u10 = ds_u10['u10'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
            # u10_data = u10.sel(time=time).fillna(0)
            #
            # v10 = ds_u10['v10'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
            # v10_data = v10.sel(time=time).fillna(0)
            #
            # ud, vd = vort_function(u10.longitude.values, u10.latitude.values, u10_data.values, v10_data.values)
            # ur, vr = stream_function(u10.longitude, u10.latitude, u10_data, v10_data)
            #
            # ug = u10_data.values - ud - ur
            # vg = v10_data.values - vd - vr
            # # print(vg.shape)
            #
            # # mslp_data = ds_mslp['msl'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
            # # mslp_data = mslp_data.sel(time=hurricane_time).values
            # swh_data = np.nan_to_num(swh_data, nan=0)
            #
            # # mslp_data = np.nan_to_num(mslp_data, nan=0)
            # rows = len(swh_data)
            # cols = len(swh_data[0])
            #
            # # for i in range(rows):
            # #     for j in range(cols):
            # #         if swh_data[i][j] == 0:
            # #             ug[i][j] = 0
            # #             vg[i][j] = 0
            # #             # mslp_data[i][j] = 0
            # data = np.stack((ug, vg, swh_data), axis=2)
            # n = int(n)
            # m = int(m)
            # # print(m)
            # if data.shape[0] != 17 or data.shape[1] != 25:
            #     # 创建一个新的17x17数组，用0填充
            #     new_data = np.zeros((17, 25, data.shape[2]))
            #
            #     # # 计算要填充的行和列的范围
            #     # row_min = (17 - data.shape[0]) // 2
            #     # row_max = row_min + data.shape[0]
            #     # col_min = (17 - data.shape[1]) // 2
            #     # col_max = col_min + data.shape[1]
            #
            #     # 将原始数据复制到新数组中
            #     new_data[n:data.shape[0] + n, m:data.shape[1] + m, :] = data
            #
            #     # 更新data为新填充的数组
            #     data = new_data
            # if data.shape[0] == 17 and data.shape[1] == 25:
            #     One_p = data
            #     origantest = (lat_max, lon_min)  # 这个是
            #     One_p = One_p.reshape(-1, 3)
            #     One_p = scaler.transform(One_p)
            #     One_p = One_p.reshape(1, 17, 25, 3)
            #     One_p = One_p.transpose(0, 3, 1, 2)
            #     One_p = torch.Tensor(One_p)


predictions = np.array(result)

    # targets = np.array(y)
origan = np.array(X2)
    # print(targets.shape)
print(predictions.shape)
predictions_denorm = denormalize_coordinates(predictions, width, height)
targets_denorm = denormalize_coordinates(predictions, width, height)
oooku = np.array(oooku)
for i in range(predictions_denorm.shape[0]):
    predictions_denorm[i, 0] = oooku[i, 0] - predictions_denorm[i, 0]
    predictions_denorm[i, 1] = oooku[i, 1] + predictions_denorm[i, 1]



for i in range( predictions_denorm.shape[0]):
    targets_denorm[i, 0] = origan[i, 0]
    targets_denorm[i, 1] = origan[i, 1]
#
# 将数据分组并在每组之前加上66666
grouped_targets = []
grouped_predictions = []

for idx in range(len(index_i)):
    group_start = index_i[idx]-(idx)*2
    if idx == len(index_i) - 1:
        group_end = len(targets_denorm)
    else:
        group_end = index_i[idx + 1]-(idx+1)*2

    group_targets = np.vstack(([66666, 66666], targets_denorm[group_start:group_end]))
    group_predictions = np.vstack(([66666, 66666], predictions_denorm[group_start:group_end]))

    grouped_targets.append(group_targets)
    grouped_predictions.append(group_predictions)
#
# 将所有分组数据拼接起来
all_targets = np.vstack(targets_denorm)
all_predictions = np.vstack(predictions_denorm)
#
# 将数据转换为DataFrame
df_targets = pd.DataFrame(all_targets, columns=['lat', 'lon'])
df_predictions = pd.DataFrame(all_predictions, columns=['lat', 'lon'])

# # 将DataFrame保存为TXT文件
# df_targets.to_csv('coodi_cnntruth24.txt', sep=' ', index=False, header=False)
# df_predictions.to_csv('coodi_cnnpred24.txt', sep=' ', index=False, header=False)
# #
# # print("Data saved successfully.")
# #
# np.save('coodi_cnntruth24.npy',targets_denorm)
# np.save('coodi_cnnpred24.npy',predictions_denorm)


from haversine import haversine
# 计算每一对经纬度之间的距离，并将结果保存到一个列表中
distances = []
for coord1, coord2 in zip(targets_denorm, predictions_denorm):
    distance = haversine((coord1[0], coord1[1]), (coord2[0], coord2[1]))
    distances.append(distance)

# 计算平均距离
average_distance = np.mean(distances)

print(f"平均误差距离为: {average_distance} 公里")







