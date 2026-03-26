import cv2
import numpy as np

def main():
    """
    使用OpenCV打开摄像头并实时展示画面，同时进行蓝色积木块的颜色分拣
    使用HSV颜色空间提高分拣准确率
    """
    # 创建VideoCapture对象，参数0表示默认摄像头
    cap = cv2.VideoCapture(2) #如果电脑上有多个摄像头，需要调整摄像头的id，默认值0
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，蓝色积木块分拣程序启动")
    print("按 'q' 键退出")
    
    # 定义蓝色在HSV颜色空间中的范围
    # HSV中蓝色的色调(Hue)范围大约在100-130之间
    # 饱和度(Saturation)和明度(Value)设置合适的范围以适应不同光照条件
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
            if area > 500:  # 最小面积阈值，可根据实际情况调整
                blue_objects_count += 1
                
                # 获取轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 在原图上绘制边界框
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加标签
                label = f"Blue Block {blue_objects_count}"
                cv2.putText(result_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 计算轮廓中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 在中心点绘制圆点
                    cv2.circle(result_frame, (cx, cy), 5, (255, 0, 0), -1)
        
        # 在图像上显示检测到的蓝色积木块数量
        info_text = f"Detected Blue Blocks: {blue_objects_count}"
        cv2.putText(result_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示原始画面和处理结果
        cv2.imshow('Blue Block Detection', result_frame)
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
    print("摄像头已关闭，蓝色积木块分拣程序结束")

if __name__ == "__main__":
    main()