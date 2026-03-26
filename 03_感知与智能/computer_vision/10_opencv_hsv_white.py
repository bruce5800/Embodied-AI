import cv2
import numpy as np

def main():
    """
    使用OpenCV打开摄像头，缩小图像尺寸后切割ROI区域，并检测白色圆形瓶盖
    """
    # 创建VideoCapture对象，参数2表示第三个摄像头（参考03_opencv_camera.py）
    cap = cv2.VideoCapture(2)  # 如果电脑上有多个摄像头，需要调整摄像头的id，默认值0
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # ROI区域参数
    roi_x = 68
    roi_y = 63
    roi_w = 186
    roi_h = 137
    
    # 白色圆形瓶盖的HSV颜色范围
    lower_white = np.array([91, 18, 118])   # HSV下限
    upper_white = np.array([126, 123, 218]) # HSV上限
    
    print("摄像头已打开，按 'q' 键退出")
    print(f"ROI区域设置: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
    print(f"白色瓶盖HSV范围: Lower={lower_white}, Upper={upper_white}")
    print("图像处理流程:")
    print("  1. 高斯滤波 -> 双边滤波 -> CLAHE直方图均衡化")
    print("  2. 形态学操作: 开运算 -> 闭运算 -> 膨胀 -> 腐蚀 -> 闭运算")
    print("  3. 圆形检测: 面积>300, 圆形度>0.4, 长宽比0.6-1.4")
    print("快捷键: 'q'-退出, 's'-保存, 'i'-信息, 'r'-重置参数")
    
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
        
        # 检查ROI区域是否在缩小后的图像范围内
        if (roi_x + roi_w <= new_width) and (roi_y + roi_h <= new_height):
            # 切割ROI区域
            roi_region = resized_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # === 图像预处理：滤波去噪，减少光线影响 ===
            # 1. 高斯滤波去除噪声
            roi_filtered = cv2.GaussianBlur(roi_region, (5, 5), 0)
            
            # 2. 双边滤波保持边缘的同时去噪
            roi_filtered = cv2.bilateralFilter(roi_filtered, 9, 75, 75)
            
            # 3. 直方图均衡化改善光照不均
            roi_lab = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2LAB)
            roi_lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(roi_lab[:,:,0])
            roi_filtered = cv2.cvtColor(roi_lab, cv2.COLOR_LAB2BGR)
            
            # 将预处理后的ROI区域转换为HSV颜色空间
            roi_hsv = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2HSV)
            
            # 创建白色瓶盖的掩码
            white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)
            
            # === 形态学操作优化：多步骤处理 ===
            # 1. 定义不同大小的核
            kernel_small = np.ones((3,3), np.uint8)  # 小核用于细节处理
            kernel_medium = np.ones((5,5), np.uint8) # 中核用于一般处理
            kernel_large = np.ones((7,7), np.uint8)  # 大核用于连接断开的区域
            
            # 2. 先用小核进行开运算去除小噪点
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
            
            # 3. 用中核进行闭运算填充内部空洞
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
            
            # 4. 膨胀操作增强目标区域
            white_mask = cv2.dilate(white_mask, kernel_medium, iterations=1)
            
            # 5. 腐蚀操作恢复原始大小并去除细小连接
            white_mask = cv2.erode(white_mask, kernel_small, iterations=1)
            
            # 6. 最后用大核进行闭运算连接可能断开的圆形区域
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建ROI区域的副本用于绘制检测结果
            roi_with_detection = roi_filtered.copy()  # 使用预处理后的图像
            
            # 检测到的圆形瓶盖数量
            bottle_cap_count = 0
            
            # 遍历所有轮廓
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                
                # 过滤太小的轮廓（降低面积阈值以适应更好的检测）
                if area > 300:  # 降低最小面积阈值
                    # 计算轮廓的外接圆
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    # 计算轮廓的圆形度（用于判断是否为圆形）
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # 降低圆形度阈值以适应形态学处理后的形状变化
                        if circularity > 0.4:  # 降低圆形度阈值
                            # 额外的形状验证：检查长宽比
                            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
                            aspect_ratio = float(w_rect) / h_rect
                            
                            # 圆形的长宽比应该接近1
                            if 0.6 <= aspect_ratio <= 1.4:  # 长宽比范围
                                bottle_cap_count += 1
                                
                                # 绘制检测到的圆形瓶盖
                                cv2.circle(roi_with_detection, center, radius, (0, 255, 0), 2)
                                cv2.circle(roi_with_detection, center, 2, (0, 0, 255), 3)
                                
                                # 添加标签
                                cv2.putText(roi_with_detection, f'Cap {bottle_cap_count}', 
                                           (center[0]-20, center[1]-radius-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # 显示详细信息
                                info_text = f'A:{int(area)} C:{circularity:.2f} R:{aspect_ratio:.2f}'
                                cv2.putText(roi_with_detection, info_text, 
                                           (center[0]-40, center[1]+radius+20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            # 在缩小后的图像上绘制ROI区域的边框
            cv2.rectangle(resized_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
            
            # 添加ROI区域标签和检测结果
            cv2.putText(resized_frame, f'ROI Region - Caps: {bottle_cap_count}', (roi_x, roi_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示原始缩小后的画面（带ROI框）
            cv2.imshow('Camera Live View (Resized)', resized_frame)
            
            # 显示切割出的ROI区域（带检测结果）
            cv2.imshow('ROI Region with Detection', roi_with_detection)
            
            # 显示预处理后的ROI区域
            cv2.imshow('ROI Filtered', roi_filtered)
            
            # 显示白色瓶盖的掩码
            cv2.imshow('White Cap Mask', white_mask)
            
            # 将ROI区域放大显示，便于观察
            roi_enlarged = cv2.resize(roi_with_detection, (roi_w*2, roi_h*2))
            cv2.imshow('ROI Region (Enlarged)', roi_enlarged)
            
        else:
            # ROI区域超出图像范围，显示警告
            cv2.putText(resized_frame, 'ROI out of bounds!', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Camera Live View (Resized)', resized_frame)
            print(f"警告：ROI区域超出图像范围！图像尺寸: {new_width}x{new_height}")
        
        # 在主窗口显示图像信息
        info_text = f"Original: {width}x{height}, Resized: {new_width}x{new_height}"
        cv2.putText(resized_frame, info_text, (10, new_height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 检查按键输入，如果按下 'q' 键则退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('s'):
            # 按's'键保存当前ROI区域图像和检测结果
            if (roi_x + roi_w <= new_width) and (roi_y + roi_h <= new_height):
                roi_region = resized_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                cv2.imwrite('roi_capture.png', roi_region)
                
                # 如果有检测结果，也保存带检测标记的图像
                roi_hsv = cv2.cvtColor(roi_region, cv2.COLOR_BGR2HSV)
                white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)
                if cv2.countNonZero(white_mask) > 0:
                    cv2.imwrite('roi_with_detection.png', roi_with_detection)  # 保存检测结果
                    cv2.imwrite('roi_filtered.png', roi_filtered)  # 保存预处理结果
                    cv2.imwrite('white_cap_mask.png', white_mask)
                    print("ROI区域图像、预处理结果、检测结果和掩码已保存")
                else:
                    print("ROI区域图像已保存为 roi_capture.png")
        elif key == ord('i'):
            # 按'i'键显示当前图像和ROI信息
            print(f"\n=== 图像信息 ===")
            print(f"原始图像尺寸: {width}x{height}")
            print(f"缩小后图像尺寸: {new_width}x{new_height}")
            print(f"ROI区域: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
            print(f"白色瓶盖HSV范围: Lower={lower_white}, Upper={upper_white}")
            
            # 如果ROI区域有效，显示检测统计信息
            if (roi_x + roi_w <= new_width) and (roi_y + roi_h <= new_height):
                roi_region = resized_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                roi_hsv = cv2.cvtColor(roi_region, cv2.COLOR_BGR2HSV)
                white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)
                white_pixels = cv2.countNonZero(white_mask)
                total_pixels = roi_w * roi_h
                percentage = (white_pixels / total_pixels) * 100
                print(f"ROI区域中白色像素: {white_pixels}/{total_pixels} ({percentage:.1f}%)")
                
                # 显示当前检测参数
                print(f"\n=== 检测参数 ===")
                print(f"最小轮廓面积: 300")
                print(f"最小圆形度: 0.4")
                print(f"长宽比范围: 0.6 - 1.4")
                print(f"形态学核大小: 小核3x3, 中核5x5, 大核7x7")
                
        elif key == ord('r'):
            # 重置检测参数（可以在这里添加参数重置功能）
            print("\n=== 参数重置 ===")
            print("当前使用默认参数，如需调整请修改代码中的相关数值")
            print("可调整参数包括:")
            print("- HSV颜色范围: lower_white, upper_white")
            print("- 轮廓筛选: 最小面积, 圆形度阈值, 长宽比范围")
            print("- 形态学操作: 核大小, 迭代次数")
            print("- 滤波参数: 高斯核大小, 双边滤波参数, CLAHE参数")
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已关闭，程序结束")

if __name__ == "__main__":
    main()