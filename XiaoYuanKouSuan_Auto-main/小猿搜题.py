import cv2
import numpy as np
import pyautogui
import keyboard
import sys
import time
import logging
from tensorflow.keras.models import load_model  # 加载TensorFlow/Keras模型

# 设置日志
logging.basicConfig(level=logging.INFO)

# 加载你训练的模型，替换为你的模型路径
model = load_model('./digit_recognition_model.h5')  # 假设你的模型文件为 'your_model.h5'

# 跟踪状态的变量
not_found_count = 0
last_not_found_time = 0
last_numbers = None  # 存储上次识别的数字
skip_count = 0  # 跳过次数计数器

def capture_area(region):
    time.sleep(0.1)  # 确保屏幕内容稳定
    screenshot = pyautogui.screenshot(region=region)
    return np.array(screenshot)

def recognize_numbers(image):
    try:
        # 对图像进行预处理，假设模型需要28x28的灰度图输入
        resized_image = cv2.resize(image, (28, 28))  # 调整图像大小
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        normalized_image = gray / 255.0  # 归一化

        # 为模型准备输入格式
        input_image = np.expand_dims(normalized_image, axis=(0, -1))  # 添加批次和通道维度

        # 使用加载的模型进行预测
        predictions = model.predict(input_image)
        predicted_digit = np.argmax(predictions)  # 获取预测的数字

        logging.info(f"识别的数字: {predicted_digit}")
        return [predicted_digit]  # 返回识别的数字
    except Exception as e:
        logging.error(f"识别数字时发生错误: {e}")
        return []

def handle_insufficient_numbers():
    global not_found_count, last_not_found_time

    current_time = time.time()
    not_found_count = not_found_count + 1 if current_time - last_not_found_time <= 1 else 1
    last_not_found_time = current_time

    logging.warning("未找到足够的数字进行比较")
    if not_found_count >= 5:
        logging.info("...准备重新开始程序...")
        click_buttons()
        time.sleep(1)
        main()

def click_buttons():
    pyautogui.click(1555, 490)  # 点击“开心收下”按钮
    time.sleep(0.3)
    pyautogui.click(1670, 900)  # 点击“继续”按钮

def draw_comparison(first, second):
    global not_found_count, last_numbers, skip_count

    if first is None or second is None:
        handle_insufficient_numbers()
        return

    if last_numbers is not None and last_numbers == (first, second):
        skip_count += 1
        logging.info(f"当前结果与上次相同，跳过此次执行 (次数: {skip_count})")
        if skip_count > 5:  # 强制执行一次
            skip_count = 0
            execute_drawing_logic(first, second)
        return

    execute_drawing_logic(first, second)

    not_found_count = 0  # 重置未找到计数
    last_numbers = (first, second)  # 更新上次识别的数字
    skip_count = 0  # 重置跳过次数

def execute_drawing_logic(first, second):
    logging.info(f"识别的数字: {first}, {second}")

    if first > second:
        logging.info(f"{first} > {second}")
        draw_greater_than()
    elif first < second:
        logging.info(f"{first} < {second}")
        draw_less_than()
    else:
        logging.info(f"{first} = {second}")
        draw_same_than()

def draw_greater_than():
    pyautogui.press(".")  # BlueStacks中的大于号快捷键

def draw_less_than():
    pyautogui.press(",")  # BlueStacks中的小于号快捷键

def draw_same_than():
    pyautogui.press("/")  # BlueStacks中的等于号快捷键

def main():
    keyboard.add_hotkey('=', lambda: sys.exit("进程已结束"))  # 默认的退出快捷键

    try:
        while True:
            # 定义两个区域
            region1 = (1560, 180, 45, 45)  # 第一区域
            region2 = (1670, 180, 50, 50)  # 第二区域

            image1 = capture_area(region1)  # 截取第一区域
            image2 = capture_area(region2)  # 截取第二区域

            numbers1 = recognize_numbers(image1)  # 识别第一区域的数字
            numbers2 = recognize_numbers(image2)  # 识别第二区域的数字

            # 记录识别的数字
            logging.info(f"第一区域识别的数字: {numbers1}")
            logging.info(f"第二区域识别的数字: {numbers2}")

            first = numbers1[0] if len(numbers1) > 0 else None  # 获取第一区域的第一个数字
            second = numbers2[0] if len(numbers2) > 0 else None  # 获取第二区域的第一个数字

            draw_comparison(first, second)  # 比较并绘制结果
            time.sleep(0.01)
    except SystemExit as e:
        logging.info(e)
    except Exception as e:
        logging.error(f"主循环发生错误: {e}")

if __name__ == "__main__":
    main()  # 启动主程序
