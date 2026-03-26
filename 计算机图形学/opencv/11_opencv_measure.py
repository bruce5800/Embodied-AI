import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

class BlockMeasurer:
    def __init__(self):
        self.pixel_per_cm = None  # 每厘米的像素数
        self.reference_size_cm = 3.0  # 参考正方形积木块的实际尺寸（厘米）
        self.calibrated = False
        
    def calibrate_with_reference_block(self, width_pixels, height_pixels):
        """
        使用已知3cm x 3cm的正方形积木块进行校准
        """
        # 取宽度和高度的平均值作为参考
        avg_size_pixels = (width_pixels + height_pixels) / 2
        self.pixel_per_cm = avg_size_pixels / self.reference_size_cm
        self.calibrated = True
        print(f"校准完成：每厘米 {self.pixel_per_cm:.2f} 像素")
        
    def pixels_to_cm(self, pixels):
        """
        将像素转换为厘米
        """
        if not self.calibrated:
            return None
        return pixels / self.pixel_per_cm
        
    def is_square_block(self, width, height, tolerance=0.2):
        """
        判断是否为正方形积木块（用于校准）
        tolerance: 宽高比的容忍度
        """
        ratio = min(width, height) / max(width, height)
        return ratio > (1 - tolerance)

def put_chinese_text(img, text, position, font_size=16, color=(255, 255, 255)):
    """
    在OpenCV图像上绘制中文文本
    """
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试使用系统字体，如果失败则使用默认字体
    try:
        # Windows系统中文字体
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttf", font_size)
            except:
                # 如果都找不到，使用默认字体
                font = ImageFont.load_default()
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

def main():
    """
    使用OpenCV打开摄像头并实时展示画面，同时进行蓝色积木块的检测和尺寸测量
    """
    # 创建VideoCapture对象
    cap = cv2.VideoCapture(2)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，蓝色积木块测量程序启动")
    print("程序会自动使用3cm x 3cm的正方形积木块进行校准")
    print("按 'q' 键退出")
    
    # 创建测量器对象
    measurer = BlockMeasurer()
    
    # 定义蓝色在HSV颜色空间中的范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        # 获取原始画面的尺寸并缩小
        height, width = frame.shape[:2]
        new_width = width // 2
        new_height = height // 2
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        # 创建蓝色掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 形态学操作去除噪声
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建结果图像
        result_frame = resized_frame.copy()
        
        # 处理检测到的轮廓
        blue_objects_count = 0
        square_blocks = []  # 存储正方形积木块信息
        rectangular_blocks = []  # 存储长方形积木块信息
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面积阈值
                blue_objects_count += 1
                
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 判断是正方形还是长方形
                if measurer.is_square_block(w, h):
                    # 正方形积木块，用于校准
                    square_blocks.append((x, y, w, h))
                    
                    # 如果还未校准，使用第一个正方形积木块进行校准
                    if not measurer.calibrated:
                        measurer.calibrate_with_reference_block(w, h)
                    
                    # 绘制绿色边界框（正方形）
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = "正方形 3x3cm"
                    result_frame = put_chinese_text(result_frame, label, (x, y - 20), 14, (0, 255, 0))
                else:
                    # 长方形积木块，需要测量
                    rectangular_blocks.append((x, y, w, h))
                    
                    # 绘制红色边界框（长方形）
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # 如果已校准，显示实际尺寸
                    if measurer.calibrated:
                        width_cm = measurer.pixels_to_cm(w)
                        height_cm = measurer.pixels_to_cm(h)
                        label = f"长方形 {width_cm:.1f}x{height_cm:.1f}cm"
                        result_frame = put_chinese_text(result_frame, label, (x, y - 20), 14, (0, 0, 255))
                        
                        # 在积木块中心显示尺寸信息
                        center_x = x + w // 2
                        center_y = y + h // 2
                        size_text = f"{width_cm:.1f}x{height_cm:.1f}cm"
                        result_frame = put_chinese_text(result_frame, size_text, (center_x - 25, center_y - 8), 12, (255, 255, 255))
                    else:
                        label = "长方形 (需要校准)"
                        result_frame = put_chinese_text(result_frame, label, (x, y - 20), 14, (0, 0, 255))
                
                # 绘制中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result_frame, (cx, cy), 3, (255, 0, 0), -1)
        
        # 显示状态信息
        status_y = 25
        if measurer.calibrated:
            status_text = f"已校准 - 每厘米: {measurer.pixel_per_cm:.1f}像素"
            result_frame = put_chinese_text(result_frame, status_text, (10, status_y), 16, (0, 255, 0))
        else:
            status_text = "等待校准 - 请放置3x3cm正方形积木块"
            result_frame = put_chinese_text(result_frame, status_text, (10, status_y), 16, (0, 0, 255))
        
        # 显示检测统计
        stats_text = f"正方形: {len(square_blocks)}, 长方形: {len(rectangular_blocks)}"
        result_frame = put_chinese_text(result_frame, stats_text, (10, status_y + 25), 14, (255, 255, 255))
        
        # 显示画面
        cv2.imshow('Blue Block Measurement', result_frame)
        cv2.imshow('Blue Mask', mask)
        
        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('r'):
            # 重置校准
            measurer.calibrated = False
            measurer.pixel_per_cm = None
            print("校准已重置")
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("蓝色积木块测量程序结束")

if __name__ == "__main__":
    main()