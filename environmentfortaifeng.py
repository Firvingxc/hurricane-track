import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
# 选择特定时间


selected_datetime = '2020-08-25T12:00'
# selected_date = '2020091200'
selected_date = '2020082512'
# 打开NetCDF文件并选择特定时间
f1 = xr.open_dataset(r'D:\latlon\downloadswhbb1980-2021_file.nc').sel(time=selected_datetime)
f2 = xr.open_dataset(r'D:\latlon\downloaduvbb1980-2021_file.nc').sel(time=selected_datetime)
# f1 = xr.open_dataset('du1_swh2017-2020.nc').sel(time=selected_datetime)
# f2 = xr.open_dataset('du1_uv2017-2020_24z.nc').sel(time=selected_datetime)
# 提取数据
swh = f1['swh'].values
u10 = f2['u10'].values
v10 = f2['v10'].values
lat = f1['latitude'].values
lon = f1['longitude'].values

# 创建经纬度网格
lon2d, lat2d = np.meshgrid(lon, lat)

# 设置投影
proj = ccrs.PlateCarree()

# 创建画布和子图
fig = plt.figure(figsize=(4.5, 3), dpi=600)
ax = fig.add_subplot(1, 1, 1, projection=proj)

# 设置地图范围（extent）
extent = [-120, -20, 7, 40]  # 确保显示的范围是指定的经纬度范围
ax.set_extent(extent, crs=ccrs.PlateCarree())  # 使用 PlateCarree 投影设置范围

# 添加地图特征
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=2)
ax.add_feature(cfeature.OCEAN.with_scale('50m'))

# # 添加经纬度网格
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='k', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.xformatter = LongitudeFormatter()
# gl.yformatter = LatitudeFormatter()
# gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1] + 1, 10))
# gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3] + 1, 10))
# gl.xlabel_style = {'size': 10}
# gl.ylabel_style = {'size': 10}

# 绘制填色图（显著波高 SWH）
levels = np.arange(0, 10, 0.05)
cb = ax.contourf(lon2d, lat2d, swh, levels=levels, cmap='bwr', transform=ccrs.PlateCarree())

# 绘制风场图
cq = ax.quiver(lon2d[::5, ::5], lat2d[::5, ::5], u10[::5, ::5], v10[::5, ::5], color='k', scale=200, zorder=10,
               width=0.003, headwidth=3, headlength=4.5, transform=ccrs.PlateCarree())

# 添加风场箭头说明
# ax.quiverkey(cq, 0.85, 1.02, 10, '10 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 12})

# # 添加 colorbar
# cb = fig.colorbar(cb, ax=ax, orientation='vertical', ticks=np.arange(0, 5, 0.5), aspect=30, shrink=0.9, pad=0.02)
# cb.set_label('Significant Wave Height (m)', fontsize=18)

# 添加文本和标题
# ax.text(0.85, 0.87, 'L', color='red', fontsize=15, weight='bold', va='bottom', ha='center', transform=ax.transAxes)
# ax.text(0.92, 0.12, 'H', color='red', fontsize=15, weight='bold', va='bottom', ha='center', transform=ax.transAxes)
# plt.suptitle('2000-01-15 12:00 Wind Field and SWH', fontsize=14, y=0.93)

# 保存和显示图像
# plt.savefig(selected_datetime+"truth.png", bbox_inches='tight')
# plt.show()
plt.savefig(selected_date+"truth.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)

# 打开NetCDF文件并选择特定时间
# f1 = xr.open_dataset(r'D:\latlon\downloadswhbb1980-2021_file.nc').sel(time=selected_datetime)
# f2 = xr.open_dataset(r'D:\latlon\downloaduvbb1980-2021_file.nc').sel(time=selected_datetime)
f1 = xr.open_dataset('D:\latlon\coverland/du1_swh2017-2020.nc').sel(time=selected_datetime)
f2 = xr.open_dataset('D:\latlon\coverland/du1_uv2017-2020_24znew_6.nc').sel(time=selected_datetime)
# 提取数据
swh = f1['swh'].values
u10 = f2['u10'].values
v10 = f2['v10'].values
lat = f1['latitude'].values
lon = f1['longitude'].values

# 创建经纬度网格
lon2d, lat2d = np.meshgrid(lon, lat)

# 设置投影
proj = ccrs.PlateCarree()

# 创建画布和子图
fig = plt.figure(figsize=(4.5, 3), dpi=600)
ax = fig.add_subplot(1, 1, 1, projection=proj)

# 设置地图范围（extent）
extent = [-120, -20, 7, 40]  # 确保显示的范围是指定的经纬度范围
ax.set_extent(extent, crs=ccrs.PlateCarree())  # 使用 PlateCarree 投影设置范围

# 添加地图特征
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=2)
ax.add_feature(cfeature.OCEAN.with_scale('50m'))

# # 添加经纬度网格
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='k', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.xformatter = LongitudeFormatter()
# gl.yformatter = LatitudeFormatter()
# gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1] + 1, 10))
# gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3] + 1, 10))
# gl.xlabel_style = {'size': 10}
# gl.ylabel_style = {'size': 10}

# 绘制填色图（显著波高 SWH）
levels = np.arange(0, 10, 0.05)
cb = ax.contourf(lon2d, lat2d, swh, levels=levels, cmap='bwr', transform=ccrs.PlateCarree())

# 绘制风场图
cq = ax.quiver(lon2d[::5, ::5], lat2d[::5, ::5], u10[::5, ::5], v10[::5, ::5], color='k', scale=200, zorder=10,
               width=0.003, headwidth=3, headlength=4.5, transform=ccrs.PlateCarree())

# 添加风场箭头说明
# ax.quiverkey(cq, 0.85, 1.02, 10, '10 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 12})

# # 添加 colorbar
# cb = fig.colorbar(cb, ax=ax, orientation='vertical', ticks=np.arange(0, 5, 0.5), aspect=30, shrink=0.9, pad=0.02)
# cb.set_label('Significant Wave Height (m)', fontsize=18)

# 添加文本和标题
# ax.text(0.85, 0.87, 'L', color='red', fontsize=15, weight='bold', va='bottom', ha='center', transform=ax.transAxes)
# ax.text(0.92, 0.12, 'H', color='red', fontsize=15, weight='bold', va='bottom', ha='center', transform=ax.transAxes)
# plt.suptitle('2000-01-15 12:00 Wind Field and SWH', fontsize=14, y=0.93)

# 保存和显示图像
# plt.savefig(selected_datetime+"pred.png", bbox_inches='tight')
plt.savefig(selected_date+"pred.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)

