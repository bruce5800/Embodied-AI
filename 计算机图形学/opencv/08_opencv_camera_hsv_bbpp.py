import cv2
import numpy as np

def main():
    """
    使用OpenCV打开摄像头并实时展示画面，同时进行蓝色、黑色、紫色和粉色积木块的颜色分拣
    使用HSV颜色空间提高分拣准确率
    BBPP = Blue, Black, Purple, Pink
    """
    # 创建VideoCapture对象，参数0表示默认摄像头
    cap = cv2.VideoCapture(2) #如果电脑上有多个摄像头，需要调整摄像头的id，默认值0
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，蓝色、黑色、紫色和粉色积木块分拣程序启动")
    print("按 'q' 键退出")
    
    # 定义蓝色在HSV颜色空间中的范围
    # HSV中蓝色的色调(Hue)范围大约在100-130之间
    lower_blue = np.array([104, 106, 24])   # 蓝色下限
    upper_blue = np.array([124, 255, 255]) # 蓝色上限
    
    # 定义黑色在HSV颜色空间中的范围
    # 黑色的特点是明度(Value)很低，色调和饱和度范围可以较宽
    lower_black = np.array([0, 0, 0])      # 黑色下限
    upper_black = np.array([180, 255, 50]) # 黑色上限（主要通过低明度来识别）
    
    # 定义紫色在HSV颜色空间中的范围
    # HSV中紫色的色调(Hue)范围大约在130-160之间
    lower_purple = np.array([133, 44, 39])   # 紫色下限
    upper_purple = np.array([161, 255, 252]) # 紫色上限
    
    # 定义粉色在HSV颜色空间中的范围
    # HSV中粉色的色调(Hue)范围大约在160-180之间（接近红色但饱和度较低）
    lower_pink = np.array([156, 47, 146])   # 粉色下限
    upper_pink = np.array([173, 88, 230]) # 粉色上限
    
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
        
        # 创建四种颜色的掩码
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # 合并四个掩码用于显示
        mask_combined = cv2.bitwise_or(mask_blue, mask_black)
        mask_combined = cv2.bitwise_or(mask_combined, mask_purple)
        mask_combined = cv2.bitwise_or(mask_combined, mask_pink)
        
        # 进行形态学操作，去除噪声
        kernel = np.ones((5,5), np.uint8)
        
        # 对蓝色掩码进行形态学操作
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        
        # 对黑色掩码进行形态学操作
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
        
        # 对紫色掩码进行形态学操作
        mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
        mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
        
        # 对粉色掩码进行形态学操作
        mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_OPEN, kernel)
        mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_CLOSE, kernel)
        
        # 更新合并掩码
        mask_combined = cv2.bitwise_or(mask_blue, mask_black)
        mask_combined = cv2.bitwise_or(mask_combined, mask_purple)
        mask_combined = cv2.bitwise_or(mask_combined, mask_pink)
        
        # 创建结果图像的副本
        result_frame = resized_frame.copy()
        
        # 处理蓝色积木块
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_objects_count = 0
        
        for contour in contours_blue:
            # 计算轮廓面积，过滤掉太小的区域
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面积阈值，可根据实际情况调整
                blue_objects_count += 1
                
                # 获取轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 在原图上绘制蓝色边界框（绿色框）
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加蓝色标签
                label = f"Blue Block {blue_objects_count}"
                cv2.putText(result_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 计算轮廓中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 在中心点绘制蓝色圆点
                    cv2.circle(result_frame, (cx, cy), 5, (255, 0, 0), -1)
        
        # 处理黑色积木块
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        black_objects_count = 0
        
        for contour in contours_black:
            # 计算轮廓面积，过滤掉太小的区域
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面积阈值，可根据实际情况调整
                black_objects_count += 1
                
                # 获取轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 在原图上绘制黑色边界框（红色框）
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # 添加黑色标签
                label = f"Black Block {black_objects_count}"
                cv2.putText(result_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 计算轮廓中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 在中心点绘制红色圆点
                    cv2.circle(result_frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # 处理紫色积木块
        contours_purple, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        purple_objects_count = 0
        
        for contour in contours_purple:
            # 计算轮廓面积，过滤掉太小的区域
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面积阈值，可根据实际情况调整
                purple_objects_count += 1
                
                # 获取轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 在原图上绘制紫色边界框（洋红色框）
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                
                # 添加紫色标签
                label = f"Purple Block {purple_objects_count}"
                cv2.putText(result_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # 计算轮廓中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 在中心点绘制洋红色圆点
                    cv2.circle(result_frame, (cx, cy), 5, (255, 0, 255), -1)
        
        # 处理粉色积木块
        contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pink_objects_count = 0
        
        for contour in contours_pink:
            # 计算轮廓面积，过滤掉太小的区域
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面积阈值，可根据实际情况调整
                pink_objects_count += 1
                
                # 获取轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 在原图上绘制粉色边界框（青色框）
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
                # 添加粉色标签
                label = f"Pink Block {pink_objects_count}"
                cv2.putText(result_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 计算轮廓中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 在中心点绘制青色圆点
                    cv2.circle(result_frame, (cx, cy), 5, (255, 255, 0), -1)
        
        # 在图像上显示检测到的积木块数量统计
        total_objects = blue_objects_count + black_objects_count + purple_objects_count + pink_objects_count
        info_text1 = f"Blue Blocks: {blue_objects_count}"
        info_text2 = f"Black Blocks: {black_objects_count}"
        info_text3 = f"Purple Blocks: {purple_objects_count}"
        info_text4 = f"Pink Blocks: {pink_objects_count}"
        info_text5 = f"Total Blocks: {total_objects}"
        
        cv2.putText(result_frame, info_text1, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, info_text2, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result_frame, info_text3, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(result_frame, info_text4, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(result_frame, info_text5, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示原始画面和处理结果
        cv2.imshow('Blue & Black & Purple & Pink Block Detection', result_frame)
        cv2.imshow('Combined Mask', mask_combined)
        cv2.imshow('Blue Mask', mask_blue)
        cv2.imshow('Black Mask', mask_black)
        cv2.imshow('Purple Mask', mask_purple)
        cv2.imshow('Pink Mask', mask_pink)
        
        # 检查按键输入，如果按下 'q' 键则退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已关闭，蓝色、黑色、紫色和粉色积木块分拣程序结束")

if __name__ == "__main__":
    main()