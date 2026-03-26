import cv2
import numpy as np
from collections import deque

def main():
    """
    使用OpenCV实时跟踪紫色积木块并绘制移动轨迹
    """
    # 创建VideoCapture对象，参数2表示第三个摄像头（根据参考代码）
    cap = cv2.VideoCapture(1)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，开始跟踪紫色积木块")
    print("按 'q' 键退出，按 'c' 键清除轨迹")
    
    # 定义紫色的HSV颜色范围
    lower_purple = np.array([0, 0, 0])   # 紫色下限
    upper_purple = np.array([179, 188, 31]) # 紫色上限
    
    # 创建轨迹点队列，最多保存100个点
    trajectory_points = deque(maxlen=100)
    
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
        
        # 将BGR图像转换为HSV颜色空间
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        # 创建紫色区域的掩码
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # 使用形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果找到轮廓
        if contours:
            # 找到最大的轮廓（假设是我们要跟踪的积木块）
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓面积，过滤掉太小的区域
            area = cv2.contourArea(largest_contour)
            if area > 500:  # 最小面积阈值
                # 计算轮廓的质心
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 将质心添加到轨迹点队列
                    trajectory_points.append((cx, cy))
                    
                    # 在积木块周围绘制边界框
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 在质心位置绘制圆点
                    cv2.circle(resized_frame, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # 显示坐标信息
                    cv2.putText(resized_frame, f"Center: ({cx}, {cy})", 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制轨迹
        if len(trajectory_points) > 1:
            # 将轨迹点转换为numpy数组
            points = np.array(trajectory_points, dtype=np.int32)
            
            # 绘制轨迹线
            for i in range(1, len(points)):
                # 计算线条透明度（越新的点越不透明）
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                
                # 绘制连接线
                cv2.line(resized_frame, tuple(points[i-1]), tuple(points[i]), 
                        (255, 0, 255), thickness)
            
            # 在轨迹点上绘制小圆点
            for i, point in enumerate(points):
                alpha = (i + 1) / len(points)
                radius = max(1, int(3 * alpha))
                cv2.circle(resized_frame, tuple(point), radius, (255, 255, 0), -1)
        
        # 显示轨迹点数量
        cv2.putText(resized_frame, f"Trajectory Points: {len(trajectory_points)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示操作提示
        cv2.putText(resized_frame, "Press 'q' to quit, 'c' to clear trajectory", 
                   (10, resized_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 在窗口中显示处理后的画面
        cv2.imshow('Purple Block Tracker', resized_frame)
        
        # 可选：显示掩码窗口（用于调试）
        cv2.imshow('Purple Mask', mask)
        
        # 检查按键输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('c'):
            print("清除轨迹")
            trajectory_points.clear()
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已关闭，程序结束")

if __name__ == "__main__":
    main()