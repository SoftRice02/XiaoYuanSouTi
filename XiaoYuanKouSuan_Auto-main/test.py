import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 读取标签文件
def load_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            image_name, label = line.split('\t')
            labels[image_name.split('/')[-1]] = float(label.strip().replace(',', ''))  # 去掉逗号并转换为浮点数
    return labels

# 2. 加载图片数据
def load_data(image_folder, label_file):
    labels = load_labels(label_file)
    images = []
    image_labels = []
    
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        try:
            # 加载图片，转为灰度图并调整大小为28x28
            image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
            image = img_to_array(image) / 255.0  # 将图片数据归一化到 [0, 1]
            images.append(image)
            image_labels.append(labels[image_name])
        except Exception as e:
            print(f"无法加载图片 {image_name}: {e}")
    
    return np.array(images), np.array(image_labels)

# 3. 确认数据路径
image_folder = 'G:\\python code\\CreatedData\\image'  # 替换为你的实际路径
label_file = 'G:\\python code\\CreatedData\\label.txt' # 替换为你的实际路径

# 4. 加载图片和标签数据
images, labels = load_data(image_folder, label_file)

# 5. 确认数据加载是否正确
print(f"图片数据形状: {images.shape}, 标签数据形状: {labels.shape}")

# 6. 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 确认训练集和测试集的数据形状
print(f"训练集: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
print(f"测试集: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

# 7. 构建卷积神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)  # 回归问题，输出一个数值
])

# 8. 编译模型
model.compile(optimizer='adam', 
              loss='mean_squared_error',  # 均方误差 (MSE) 作为损失函数
              metrics=['mae'])  # 平均绝对误差 (MAE)

# 9. 查看模型结构
model.summary()

# 10. 训练模型
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# 11. 保存模型
model.save('digit_recognition_model.h5')

# 12. 评估模型
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"测试集上的损失: {test_loss}")
print(f"测试集上的 MAE: {test_mae}")

# 13. 使用测试集进行预测
predictions = model.predict(X_test)

# 14. 对比前5个样本的真实值和预测值
print("真实值 vs 预测值（前5个样本）:")
for i in range(5):
    print(f"真实值: {y_test[i]:.2f}, 预测值: {predictions[i][0]:.2f}")

# 15. 可视化前5个样本，显示它们的真实值和预测值
def visualize_predictions(X_test, y_test, predictions):
    for i in range(5):
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"真实值: {y_test[i]:.2f}, 预测值: {predictions[i][0]:.2f}")
        plt.show()

visualize_predictions(X_test, y_test, predictions)
