import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# 1. 加载已经训练好的模型
model = tf.keras.models.load_model('digit_recognition_model.h5')

# 2. 定义图片预处理函数
def preprocess_image(image_path):
    """
    加载并预处理图片，转换为 28x28 的灰度图，归一化到 [0, 1]
    """
    image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))  # 加载并调整为28x28的灰度图
    image = img_to_array(image) / 255.0  # 转换为数组并归一化
    image = np.expand_dims(image, axis=0)  # 增加一个维度 (1, 28, 28, 1)，适应模型输入
    return image

# 3. 对文件夹中的图片进行预测
def predict_images_in_folder(folder_path):
    """
    加载文件夹中的所有图片，进行预处理并使用模型进行预测
    """
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        try:
            # 预处理图片
            image = preprocess_image(image_path)
            # 使用模型预测
            prediction = model.predict(image)
            # 打印图片名称和预测结果
            print(f"图片: {image_name}, 预测值: {prediction[0][0]:.2f}")
            
            # 可视化图片和预测值
            plt.imshow(image.reshape(28, 28), cmap='gray')
            plt.title(f"预测值: {prediction[0][0]:.2f}")
            plt.show()

        except Exception as e:
            print(f"无法处理图片 {image_name}: {e}")

# 4. 指定你想要识别的图片文件夹路径
folder_path = r'G:\python code\XiaoYuanKouSuan_Auto-main\new_images' # 替换为你的实际图片文件夹路径

# 5. 对文件夹中的图片进行识别
predict_images_in_folder(folder_path)
