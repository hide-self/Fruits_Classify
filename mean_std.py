from PIL import Image
import os
import numpy as np

# 文件夹路径，包含所有图片文件
folder_path = 'fruits'

# 初始化累积变量
channel_pixel_count = 0  # 每个通道的像素总数（H×W的和）
sum_pixel_values = np.zeros(3)  # 三个通道的像素值总和
sum_squared_pixel_values = np.zeros(3)  # 三个通道的平方和

# 遍历文件夹中的图片文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            try:
                image = Image.open(image_path)

                # 转换为RGB格式，确保3通道
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image_array = np.array(image)  # 形状: (H, W, 3)

                # 归一化到0-1
                normalized_image_array = image_array / 255.0

                # 获取图像尺寸
                h, w, c = normalized_image_array.shape

                # 累加每个通道的像素数（H×W）
                channel_pixel_count += h * w

                # 累加每个通道的像素值总和
                # 按通道求和：axis=(0,1) 对H和W维度求和，保留通道维度
                sum_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

                # 累加每个通道的像素值平方和（用于计算方差）
                sum_squared_pixel_values += np.sum(normalized_image_array ** 2, axis=(0, 1))

            except Exception as e:
                print(f"处理图片 {filename} 时出错: {e}")
                continue

# 计算每个通道的均值
mean = sum_pixel_values / channel_pixel_count

# 计算每个通道的方差（使用公式：E[X²] - E[X]²）
variance = sum_squared_pixel_values / channel_pixel_count - mean ** 2

print("Mean (R, G, B):", mean)
print("Variance (R, G, B):", variance)
print("Std (R, G, B):", np.sqrt(variance))