import cv2
import numpy as np
import math

def main():
    """
    使用OpenCV打开摄像头并实时展示画面，同时进行蓝色积木块的颜色分拣
    增强功能：显示真实边框、中心点坐标和旋转角度
    """
    # 创建VideoCapture对象，参数0表示默认摄像头
    cap = cv2.VideoCapture(2) #如果电脑上有多个摄像头，需要调整摄像头的id，默认值0
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，蓝色积木块检测程序启动（增强版）")
    print("功能：真实边框包裹 + 中心点坐标 + 旋转角度")
    print("按 'q' 键退出")
    
    # 定义蓝色在HSV颜色空间中的范围
    lower_blue = np.array([100, 50, 50])   # 蓝色下限
    upper_blue = np.array([130, 255, 255]) # 蓝色上限
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        # 检查是否成功读取到画面
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        # 获取原始画面的尺寸
        height, width = frame.shape[:2]
        
        # 将画面缩小为原来的1/2
        new_width = width // 2
        new_height = height // 2
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 将BGR颜色空间转换为HSV颜色空间
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        # 创建蓝色掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 进行形态学操作，去除噪声
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建结果图像的副本
        result_frame = resized_frame.copy()
        
        # 处理检测到的轮廓
        blue_objects_count = 0
        for contour in contours:
            # 计算轮廓面积，过滤掉太小的区域
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面积阈值
                blue_objects_count += 1
                
                # 方法1：绘制轮廓的真实边框（轮廓本身）
                cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)
                
                # 方法2：计算最小外接矩形（带旋转角度）
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # 绘制最小外接矩形
                cv2.drawContours(result_frame, [box], 0, (255, 0, 0), 2)
                
                # 获取矩形的中心点、尺寸和角度
                center, (width_rect, height_rect), angle = rect
                center_x, center_y = int(center[0]), int(center[1])
                
                # 计算轮廓的质心（更精确的中心点）
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = center_x, center_y
                
                # 绘制中心点
                cv2.circle(result_frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(result_frame, (cx, cy), 12, (255, 255, 255), 2)
                
                # 绘制角度指示线
                # 将角度转换为弧度
                angle_rad = math.radians(angle)
                line_length = 50
                
                # 计算指示线的终点
                end_x = int(cx + line_length * math.cos(angle_rad))
                end_y = int(cy + line_length * math.sin(angle_rad))
                
                # 绘制角度指示线
                cv2.arrowedLine(result_frame, (cx, cy), (end_x, end_y), (255, 255, 0), 3)
                
                # 添加文本标签
                label_y_offset = 0
                
                # 显示积木块编号
                label = f"Block {blue_objects_count}"
                cv2.putText(result_frame, label, (cx - 50, cy - 60 + label_y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                label_y_offset += 20
                
                # 显示中心点坐标
                center_text = f"Center: ({cx}, {cy})"
                cv2.putText(result_frame, center_text, (cx - 50, cy - 60 + label_y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                label_y_offset += 15
                
                # 显示旋转角度
                angle_text = f"Angle: {angle:.1f}°"
                cv2.putText(result_frame, angle_text, (cx - 50, cy - 60 + label_y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                label_y_offset += 15
                
                # 显示面积
                area_text = f"Area: {int(area)}"
                cv2.putText(result_frame, area_text, (cx - 50, cy - 60 + label_y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 在图像左上角显示详细信息
                info_start_y = 30 + (blue_objects_count - 1) * 80
                cv2.putText(result_frame, f"=== Block {blue_objects_count} ===", 
                           (10, info_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(result_frame, f"Center: ({cx}, {cy})", 
                           (10, info_start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(result_frame, f"Rotation: {angle:.1f}°", 
                           (10, info_start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(result_frame, f"Area: {int(area)} px²", 
                           (10, info_start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(result_frame, f"Size: {width_rect:.0f}x{height_rect:.0f}", 
                           (10, info_start_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 在图像顶部显示检测到的蓝色积木块总数
        total_info = f"Total Blue Blocks Detected: {blue_objects_count}"
        cv2.putText(result_frame, total_info, (10, result_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加图例说明
        legend_x = result_frame.shape[1] - 250
        cv2.putText(result_frame, "Legend:", (legend_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_frame, "Green: Contour", (legend_x, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result_frame, "Blue: Min Area Rect", (legend_x, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(result_frame, "Red: Center Point", (legend_x, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(result_frame, "Yellow: Rotation", (legend_x, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 显示原始画面和处理结果
        cv2.imshow('Blue Block Detection - Enhanced', result_frame)
        cv2.imshow('Blue Mask', mask)
        
        # 检查按键输入，如果按下 'q' 键则退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已关闭，蓝色积木块检测程序结束")

if __name__ == "__main__":
    main()
