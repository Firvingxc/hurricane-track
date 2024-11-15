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
# 生成训练集
X = np.load('X_data_big.npy')


#
scaler = MinMaxScaler(feature_range=(-1,1))
X = X.reshape(-1,3)
scaler = scaler.fit(X)
joblib.dump(scaler, 'scaler.pkl')
X = scaler.transform(X)

X = X.reshape(-1,17,25,3)
X = X.transpose(0,3,1,2)
print(X.shape)
y = np.load('y_data_big.npy')


origan = np.load('origan_col.npy')



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

y = normalize_coordinates(y,width,height)
X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y)
origan = torch.Tensor(origan)
# 划分数据集为训练、验证和测试
# X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

num_samples = X.shape[0]
num_train = round(num_samples*0.7)
num_test = round(num_samples*0.2)
print(num_test)
num_val = num_samples-num_train-num_test

origantest = origan[-num_test:]
origantrain = origan[:num_train]
origanval = origan[num_train:num_train+num_val]

# X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3)
X_train, y_train = X_tensor[:num_train],y_tensor[:num_train]
X_val,y_val = X_tensor[num_train:num_train+num_val], y_tensor[num_train:num_train+num_val]
X_test,y_test = X_tensor[-num_test:], y_tensor[-num_test:]


# 创建数据加载器
train_loader = DataLoader(TensorDataset(X_train, y_train,origantrain), batch_size=32,shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val,origanval), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)



import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


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
# Creating the model instance







# 实现ResNet-18模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1)

        self.pool1 = nn.AvgPool2d(3, 2, 1)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(17, 17, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128, 2)
        # self.fc2 = nn.Linear(50, 2)


    def forward(self, x):

        x = F.tanh(self.conv1(x))
        x = self.pool1(x)

        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        x = self.pool2(x)
        x = F.tanh(self.conv4(x))
        x = self.pool2(x)
        x = F.tanh(self.conv5(x))
        # x = self.pool(x)

        # x = self.pool(x)
        # x = F.sigmoid(self.conv1(x))
        # x = self.pool2(x)
        #
        # x = F.sigmoid(self.conv2(x))
        # x = self.pool2(x)
        # x = F.sigmoid(self.conv3(x))
        # x = self.pool2(x)
        # x = F.sigmoid(self.conv4(x))
        # x = self.pool2(x)
        # x = F.sigmoid(self.conv5(x))
        # x = F.sigmoid(self.conv6(x))

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.fc2(x)

        return x

#model = ConvNet()
model = UNet(3,2)
# model = ResNet(ResBlock)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#criterion = nn.SmoothL1Loss()
criterion = nn.L1Loss()
# 训练和验证模型
# best_valid_loss = float('inf')
# for epoch in range(600):
#     model.train()
#     train_loss = 0  # 初始化训练损失
#     for inputs, labels,origan in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         outputs = origan+outputs
#         labels = origan+labels
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()  # 累积训练损失
#
#     train_loss /= len(train_loader)  # 计算平均训练损失
#
#     model.eval()
#     valid_loss = 0
#     with torch.no_grad():
#         for inputs, labels,origan in val_loader:
#
#
#             outputs = model(inputs)
#             outputs = origan + outputs
#             labels = origan + labels
#             vloss = criterion(outputs, labels)
#             valid_loss += vloss.item()
#     valid_loss /= len(val_loader)  # 计算平均验证损失
#
#     print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
#
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'best_model3.pth')
# # # 训练和验证模型略...
model.load_state_dict(torch.load('best_model3.pth'))
# 测试模型
model.eval()
test_losses = []
predictions = []
targets = []
with torch.no_grad():
    for inputs, labels in test_loader:
        # inputs = inputs.unsqueeze(1)

        outputs = model(inputs)
        test_loss = criterion(outputs, labels)
        test_losses.append(test_loss.item())
        predictions.extend(outputs.numpy())
        targets.extend(labels.numpy())

# 计算平均测试损失
average_test_loss = np.mean(test_losses)
print(f'Average Test Loss: {average_test_loss}')

# 反归一化预测和真实坐标
predictions = np.array(predictions)
targets = np.array(targets)
predictions_denorm = denormalize_coordinates(predictions, width, height)
targets_denorm = denormalize_coordinates(targets, width, height)

for i in range(targets_denorm.shape[0]):
    predictions_denorm[i,0] = origantest[i,0]-predictions_denorm[i,0]
    predictions_denorm[i,1] = origantest[i,1] + predictions_denorm[i,1]

for i in range(targets_denorm.shape[0]):
    targets_denorm[i,0] = origantest[i,0]-targets_denorm[i,0]
    targets_denorm[i,1] = origantest[i,1] + targets_denorm[i,1]

from haversine import haversine
# 计算每一对经纬度之间的距离，并将结果保存到一个列表中
distances = []
for coord1, coord2 in zip(targets_denorm, predictions_denorm):
    distance = haversine((coord1[0], coord1[1]), (coord2[0], coord2[1]))
    distances.append(distance)

# 计算平均距离
average_distance = np.mean(distances)

print(f"平均误差距离为: {average_distance} 公里")
# predictions_denorm = predictions
# targets_denorm = targets
# print(targets_denorm[0:10])
#
print(targets_denorm.shape)
predictions_denorm = predictions_denorm[-1500:]
targets_denorm = targets_denorm[-1500:]
print(targets_denorm)


# 可视化结果
plt.figure(figsize=(10, 5))
for i in range(min(100, len(predictions_denorm))):  # 只显示前10个结果
    plt.scatter(predictions_denorm[i, 0], predictions_denorm[i, 1], c='r', label='Predicted' if i == 0 else "")
    plt.scatter(targets_denorm[i, 0], targets_denorm[i, 1], c='b', label='Actual' if i == 0 else "")
plt.title('Predicted vs Actual Coordinates')
plt.xlabel('Width')
plt.ylabel('Height')
plt.legend()
plt.show()