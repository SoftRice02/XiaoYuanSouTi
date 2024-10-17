import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# 设置图片的路径和标签文件路径
image_folder = 'G:\python code\CreatedData\image'  # 替换为你实际的图片文件夹路径
label_file = 'G:\python code\CreatedData\label.txt'       # 替换为你实际的标签文件路径

# 读取标签文件
def load_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            # 文件格式：image/0.png	-9,784.74
            image_name, label = line.split('\t')
            # 去掉逗号，并转换为浮点数
            labels[image_name.split('/')[-1]] = float(label.strip().replace(',', ''))  
    return labels

# 加载图片和标签
def load_data(image_folder, label_file):
    labels = load_labels(label_file)
    images = []
    image_labels = []
    
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        # 加载图片并调整大小为 28x28，转换为灰度图像
        image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
        image = img_to_array(image) / 255.0  # 归一化到 [0, 1] 范围
        images.append(image)
        image_labels.append(labels[image_name])
    
    return np.array(images), np.array(image_labels)

# 加载数据
images, labels = load_data(image_folder, label_file)

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 构建卷积神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)  # 回归问题，输出一个数值
])

# 编译模型
model.compile(optimizer='adam',
              loss='mean_squared_error',  # 使用均方误差作为损失函数
              metrics=['mae'])  # 平均绝对误差

# 查看模型架构
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# 保存训练好的模型
model.save('digit_recognition_model.h5')

# 如果想要加载模型可以使用以下代码：
# model = tf.keras.models.load_model('digit_recognition_model.h5')
