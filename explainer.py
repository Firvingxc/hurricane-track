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
import matplotlib as mpl

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
        self.up3 = Up(96,3, bilinear)
        self.outc = nn.Conv2d(3, n_classes, kernel_size=1)
        self.fc = nn.Linear(n_classes*17*25, 2)  # Output 2 coordinates
        # self.fc1 = nn.Linear(10,2)

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
        # output = torch.tanh(output)
        # output = self.fc1(output)
        return output
#model = ConvNet()
model = UNet(3,1)
model.load_state_dict(torch.load('D:\latlon/12h/best_model3t_ORIGMSE4.pth'))
model = model.eval()





height = 17
width = 25
num_classes = 2
# 生成训练集
X = np.load('D:\latlon/12h/X_data_big_ex.npy')

ff = X
#
scaler = MinMaxScaler(feature_range=(-1,1))
X = X.reshape(-1,3)
scaler = scaler.fit(X)
#scaler = joblib.load('scaler.pkl')
joblib.dump(scaler, 'D:\latlon/12h/scaler.pkl')
X = scaler.transform(X)

X = X.reshape(-1,17,25,3)
X = X.transpose(0,3,1,2)
ff = ff.transpose(0,3,1,2)
print(X.shape)
y = np.load('D:\latlon/12h/y_data_big_ex.npy')


origan = np.load('D:\latlon/12h/origan_col_ex.npy')



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
print(y[0:10])
y = normalize_coordinates(y,width,height)
print(y.shape)
X_tensor = X
y_tensor = y
origan = torch.Tensor(origan)
# 划分数据集为训练、验证和测试
# X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

num_samples = X.shape[0]
num_train = round(num_samples*0.7)
num_test = round(num_samples*0.2)
print(num_test)
num_val = num_samples-num_train-num_test

origantest = origan[:]
origantrain = origan[:num_train]
origanval = origan[num_train:num_train+num_val]

# X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3)
X_train, y_train = X_tensor[:num_train],y_tensor[:num_train]
X_val,y_val = X_tensor[num_train:num_train+num_val], y_tensor[num_train:num_train+num_val]
X_test,y_test = X_tensor[:], y_tensor[:]
ff = ff[:]



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
def compute_gradcam(model, input_tensor, target_layer, target_coords):
    model.eval()
    activations = []
    gradients = []

    # Register a forward hook to capture activations
    def forward_hook(module, input, output):
        activations.append(output)

    # Register a backward hook to capture gradients
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Attach hooks
    hook_fwd = target_layer.register_forward_hook(forward_hook)
    hook_bwd = target_layer.register_backward_hook(backward_hook)

    # Get the input tensor and create a copy to modify the desired dimension
    modified_input = input_tensor.clone()

    # Zero out the other dimensions except the one we're interested in
    # for dim in range(3):
    #     if dim != 2:
    #         modified_input[:, dim, :, :] = 0
    # print(input_tensor[0,2,:,:])

    # Perform forward pass
    output = model(input_tensor)
    predicted_coords = output[0].detach().numpy()
    coord = (predicted_coords[0], predicted_coords[1])
    print(f"Predicted Coordinates: {coord}")

    # Perform backward pass on target coordinate (e.g., x or y)
    model.zero_grad()
    output[0][1].backward(retain_graph=True)

    # Remove hooks
    hook_fwd.remove()
    hook_bwd.remove()

    # Ensure gradients are available
    if len(gradients) == 0:
        raise ValueError("Gradients not captured. Ensure target layer and hooks are correct.")

    # Compute weights from gradients
    grad_mean = torch.mean(gradients[0], dim=[0, 2, 3])
    activation_map = activations[0][0]
    print(activation_map.shape,'map')

    # Generate Grad-CAM
    grad_cam = torch.zeros(activation_map.shape[1:], dtype=torch.float32)
    for i in range(len(grad_mean)):
        print(i,'i')
        grad_cam += grad_mean[i] * activation_map[i, :, :]

    grad_cam = F.relu(grad_cam)
    grad_cam = grad_cam / grad_cam.max() if grad_cam.max() > 0 else grad_cam

    return grad_cam.detach().numpy(), predicted_coords






def visualize_gradcam(img, grad_cam, target_coords):
    print(grad_cam)

    grad_cam_resized = cv2.resize(grad_cam, (img.shape[2], img.shape[1]))

    # Normalize grad_cam to [0, 255] and convert to uint8
    heatmap = np.uint8(255 * grad_cam_resized)
    print(heatmap)


    # Convert heatmap to color (3 channels)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    print(heatmap.shape)

    # Convert img from tensor to numpy and check shape
    img_np = ff[num]  # Convert tensor to numpy
    if img_np.ndim == 4:
        img_np = img_np[0]  # Remove batch dimension if present
    if img_np.shape[0] == 1:  # If grayscale, expand to 3 channels
        img_np = np.repeat(img_np, 3, axis=0)

    # Convert CHW to HWC for compatibility with OpenCV
    img_np = img_np.transpose(1, 2, 0)

    # Normalize image to range [0, 255] for visualization
    # img_np = np.uint8(255 * (img_np - img_np.min()) / (img_np.max() - img_np.min()))

    # Display the result
    # 设置全局绘图风格
    mpl.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 12,
        'figure.figsize': (8, 6),
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.5,
        'lines.markersize': 1.5,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'grid.alpha': 0.5,
        'legend.frameon': False,
        'font.family': 'serif',  # 设置为衬线字体
        'font.serif': ['Times New Roman'],  # 使用 Times New Roman
    })
    plt.figure(figsize=(4, 3))
    #
    # # Show the input image and mark the true coordinates
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_np[:,:,:])
    #
    #
    # plt.plot(target_coords[1], target_coords[0], 'ro', markersize=8)  # 标记真实坐标点
    # # plt.annotate('True Coordinates', xy=target_coords, xytext=(target_coords[1] + 5, target_coords[0] + 5),
    # #              arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='white')
    # plt.title('Input Image')
    # plt.axis('off')

    # Show the heatmap
    # plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    print(heatmap.shape)

    plt.plot(target_coords[1], target_coords[0], 'ro', markersize=8)  # 在 Grad-CAM 热力图上标记真实坐标点
    # plt.annotate('True Coordinates', xy=target_coords, xytext=(target_coords[1] + 10, target_coords[0] + 10),
    #              arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='black')
    # plt.title('Grad-CAM Heatmap')
    # plt.axis('off')

    # # 在热力图上显示每个像素的数值
    # for i in range(heatmap.shape[0]):
    #     for j in range(heatmap.shape[1]):
    #         value = grad_cam_resized[i, j]  # 获取原始的 Grad-CAM 值
    #         plt.text(j, i, f'{value:.2f}', fontsize=6, ha='center', va='center', color='black')
    plt.savefig(f"grad-cam_laur.jpg", dpi=600, format='jpg', bbox_inches='tight', pad_inches=0)
    print(1)

    plt.show()


# Grad-CAM 示例
def gradcam_example(model, input_tensor, target_layer):
    # 执行前向传播，获得预测坐标
    # with torch.no_grad():
    output = model(input_tensor)

    predicted_coords = output.detach().numpy()  # 获取第一个样本的输出
    # predicted_coords = denormalize_coordinates(predicted_coords, width, height)

    target_coords = (int(predicted_coords[0, 0]), int(predicted_coords[0, 1]))  # 转换为整数坐标
    print(f"Predicted Coordinates: {target_coords}")  # 打印预测坐标

    # 计算 Grad-CAM
    grad_cam, predicted_coords = compute_gradcam(model, input_tensor, target_layer, target_coords)
    visualize_gradcam(input_tensor[0], grad_cam, target_coords)  # 传递真实坐标进行可视化
#
# # Grad-CAM 示例
# def gradcam_example(model, input_tensor, target_layer):
#     # 执行前向传播，获得预测坐标
#     with torch.no_grad():
#         output = model(input_tensor)
#
#     predicted_coords = output.numpy()  # 获取第一个样本的输出
#     predicted_coords = denormalize_coordinates(predicted_coords, width, height)
#     target_coords = (predicted_coords[0,0], predicted_coords[0,1])  # 转换为整数坐标
#     print(f"Predicted Coordinates: {target_coords}")  # 打印预测坐标
#
#     # 计算 Grad-CAM
#     grad_cam, predicted_coords = compute_gradcam(model, input_tensor, target_layer, target_coords)
#     visualize_gradcam(input_tensor[0], grad_cam)  # 可视化热力图

num = 103
# num = 32
# 选择测试集中的一个样本
sample_index =num # 选择第一个样本
X_test_sample = X_test[sample_index:sample_index + 1]  # 增加batch维度
y_test_sample = y_test[sample_index:sample_index + 1]


# Grad-CAM分析
target_layer = model.outc  # 选择目标层
gradcam_example(model, torch.Tensor(X_test_sample), target_layer)
