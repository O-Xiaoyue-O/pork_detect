import cv2
import numpy as np
import matplotlib.pyplot as plt

# 去除背景并设置背景颜色
def remove_black_background(image):
    mask = np.all(image != [0, 0, 0], axis=-1)
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    
    # 设置背景区域为白色
    background_color = [255, 255, 255]  # 可以选择其他中性色
    masked_image[~mask] = background_color
    
    return masked_image

# 计算图像的颜色直方图
def compute_color_histogram(image):
    hist = []
    for i in range(3):  # 对每个颜色通道
        channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist.append(channel_hist)
    return hist

# 进行颜色直方图匹配
def match_color_histogram(source_image, reference_hist):
    src_hist = compute_color_histogram(source_image)
    matched_image = np.zeros_like(source_image)
    
    for i in range(3):
        src_cdf = np.cumsum(src_hist[i])
        src_cdf /= src_cdf[-1]  # 归一化
        ref_cdf = np.cumsum(reference_hist[i])
        ref_cdf /= ref_cdf[-1]  # 归一化

        lookup_table = np.interp(src_cdf, ref_cdf, np.arange(256))
        matched_image[:, :, i] = lookup_table[source_image[:, :, i].astype(np.uint8)]

    return matched_image

# 将 RGB 值转换为 Lab 颜色空间并提取 L 分量
def rgb_to_lab(rgb):
    rgb = rgb.astype('float32') / 255.0
    lab = cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)
    return lab[0, 0, 0]

# 计算图像中非背景像素的 L 分量的平均值
def calculate_mean_l(image, background_color=[255, 255, 255]):
    l_sum = 0
    valid_pixel_count = 0
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rgb = image[i, j]
            
            if np.array_equal(rgb, background_color):
                continue  # 跳过背景像素
            
            l = rgb_to_lab(rgb)
            l_sum += l 
            valid_pixel_count += 1

    if valid_pixel_count > 0:
        l_mean = l_sum / valid_pixel_count
    else:
        l_mean = 0

    return l_mean

# 显示图像和直方图
def display_images_and_histograms(result_image, reference_image, matched_image):
    plt.figure(figsize=(20, 12))
    
    # 显示原图像 (result.png)
    plt.subplot(4, 4, 1)
    plt.title("原图像 (result.png)")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 显示匹配后图像
    plt.subplot(4, 4, 2)
    plt.title("匹配后图像")
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 显示参考图像 (result copy.png 或 result copy 2.png)
    plt.subplot(4, 4, 3)
    plt.title("参考图像")
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # 计算并显示处理后图像的平均 L 值
    mean_l = calculate_mean_l(matched_image)
    print("匹配后图像的 L 分量的平均值：", mean_l)

# 主函数，处理图像并选择参考图像
def process_images(result_image_path, ref_image_1_path, ref_image_2_path):
    # 读取图像
    result_image = cv2.imread(result_image_path)
    
    # 计算原始图像的平均 L 值
    masked_result_image = remove_black_background(result_image)
    mean_l_value = calculate_mean_l(masked_result_image)
    print(f"未均衡化的 L 值平均值：{mean_l_value}")

    # 根据 L 值选择参考图像
    if 43< mean_l_value <= 50:
        reference_image_path = ref_image_1_path  # 使用 result copy.png
    elif 20 <= mean_l_value <= 43:
        reference_image_path = ref_image_2_path  # 使用 result copy 4.png
    else:
        print("L 值不在指定范围内，不进行匹配。")
        return
    
    # 读取参考图像并计算其直方图
    reference_image = cv2.imread(reference_image_path)
    masked_reference_image = remove_black_background(reference_image)
    reference_hist = compute_color_histogram(masked_reference_image)
    
    # 进行颜色直方图匹配
    matched_image = match_color_histogram(masked_result_image, reference_hist)
    
    # 显示图像和直方图
    display_images_and_histograms(masked_result_image, masked_reference_image, matched_image)

# 使用示例
process_images('result.png', 'result copy.png', 'result copy 5.png')
