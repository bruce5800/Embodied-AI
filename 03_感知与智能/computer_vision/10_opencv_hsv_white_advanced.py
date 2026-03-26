#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级白色圆形瓶盖检测程序 - 带参数调节功能
基于摄像头实时检测，支持实时调节各种检测参数
"""

import cv2
import numpy as np

def create_trackbars():
    """创建参数调节滑动条"""
    cv2.namedWindow('Parameters Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Parameters Control', 400, 600)
    
    # HSV颜色范围调节
    cv2.createTrackbar('H_min', 'Parameters Control', 91, 179, lambda x: None)
    cv2.createTrackbar('S_min', 'Parameters Control', 18, 255, lambda x: None)
    cv2.createTrackbar('V_min', 'Parameters Control', 118, 255, lambda x: None)
    cv2.createTrackbar('H_max', 'Parameters Control', 126, 179, lambda x: None)
    cv2.createTrackbar('S_max', 'Parameters Control', 123, 255, lambda x: None)
    cv2.createTrackbar('V_max', 'Parameters Control', 218, 255, lambda x: None)
    
    # 形态学操作参数
    cv2.createTrackbar('Kernel_Size', 'Parameters Control', 5, 15, lambda x: None)
    cv2.createTrackbar('Morph_Iter', 'Parameters Control', 2, 10, lambda x: None)
    
    # 检测参数
    cv2.createTrackbar('Min_Area', 'Parameters Control', 300, 2000, lambda x: None)
    cv2.createTrackbar('Circularity', 'Parameters Control', 40, 100, lambda x: None)  # 0.4 * 100
    
    # 滤波参数
    cv2.createTrackbar('Gaussian_Size', 'Parameters Control', 5, 15, lambda x: None)
    cv2.createTrackbar('Bilateral_d', 'Parameters Control', 9, 20, lambda x: None)

def get_trackbar_values():
    """获取滑动条当前值"""
    h_min = cv2.getTrackbarPos('H_min', 'Parameters Control')
    s_min = cv2.getTrackbarPos('S_min', 'Parameters Control')
    v_min = cv2.getTrackbarPos('V_min', 'Parameters Control')
    h_max = cv2.getTrackbarPos('H_max', 'Parameters Control')
    s_max = cv2.getTrackbarPos('S_max', 'Parameters Control')
    v_max = cv2.getTrackbarPos('V_max', 'Parameters Control')
    
    kernel_size = cv2.getTrackbarPos('Kernel_Size', 'Parameters Control')
    if kernel_size % 2 == 0:  # 确保核大小为奇数
        kernel_size += 1
    
    morph_iter = cv2.getTrackbarPos('Morph_Iter', 'Parameters Control')
    min_area = cv2.getTrackbarPos('Min_Area', 'Parameters Control')
    circularity = cv2.getTrackbarPos('Circularity', 'Parameters Control') / 100.0
    
    gaussian_size = cv2.getTrackbarPos('Gaussian_Size', 'Parameters Control')
    if gaussian_size % 2 == 0:  # 确保高斯核大小为奇数
        gaussian_size += 1
    
    bilateral_d = cv2.getTrackbarPos('Bilateral_d', 'Parameters Control')
    
    return {
        'lower_hsv': np.array([h_min, s_min, v_min]),
        'upper_hsv': np.array([h_max, s_max, v_max]),
        'kernel_size': kernel_size,
        'morph_iter': morph_iter,
        'min_area': min_area,
        'circularity': circularity,
        'gaussian_size': gaussian_size,
        'bilateral_d': bilateral_d
    }

def advanced_morphology(mask, kernel_size, iterations):
    """高级形态学处理"""
    # 创建不同大小的核
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size+2, kernel_size+2))
    
    # 多步形态学处理
    # 1. 开运算去除小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
    
    # 2. 闭运算填充空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, medium_kernel, iterations=iterations)
    
    # 3. 膨胀增强目标
    mask = cv2.dilate(mask, medium_kernel, iterations=iterations)
    
    # 4. 腐蚀平滑边界
    mask = cv2.erode(mask, small_kernel, iterations=iterations)
    
    # 5. 最终闭运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large_kernel, iterations=1)
    
    return mask

def advanced_filtering(image, gaussian_size, bilateral_d):
    """高级滤波处理"""
    # 1. 高斯滤波去噪
    filtered = cv2.GaussianBlur(image, (gaussian_size, gaussian_size), 0)
    
    # 2. 双边滤波保边去噪
    filtered = cv2.bilateralFilter(filtered, bilateral_d, 80, 80)
    
    # 3. CLAHE直方图均衡化增强对比度
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    filtered = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return filtered

def detect_white_caps(roi_region, params):
    """检测白色瓶盖"""
    # 预处理
    roi_filtered = advanced_filtering(roi_region, params['gaussian_size'], params['bilateral_d'])
    
    # HSV转换
    roi_hsv = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2HSV)
    
    # 创建掩码
    white_mask = cv2.inRange(roi_hsv, params['lower_hsv'], params['upper_hsv'])
    
    # 形态学处理
    white_mask = advanced_morphology(white_mask, params['kernel_size'], params['morph_iter'])
    
    # 查找轮廓
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_caps = []
    roi_with_detection = roi_region.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > params['min_area']:
            # 计算圆形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > params['circularity']:
                    # 计算边界矩形和长宽比
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 0.6 <= aspect_ratio <= 1.4:
                        # 计算最小外接圆
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        center = (int(cx), int(cy))
                        radius = int(radius)
                        
                        # 绘制检测结果
                        cv2.circle(roi_with_detection, center, radius, (0, 255, 0), 2)
                        cv2.circle(roi_with_detection, center, 2, (0, 0, 255), -1)
                        
                        # 添加标签
                        label = f"Cap: A={area:.0f}, C={circularity:.2f}, R={aspect_ratio:.2f}"
                        cv2.putText(roi_with_detection, label, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        detected_caps.append({
                            'center': center,
                            'radius': radius,
                            'area': area,
                            'circularity': circularity,
                            'aspect_ratio': aspect_ratio
                        })
    
    return roi_with_detection, white_mask, roi_filtered, detected_caps

def main():
    # 摄像头设置
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # ROI区域设置
    roi_x, roi_y, roi_w, roi_h = 68, 63, 186, 137
    
    # 创建参数控制面板
    create_trackbars()
    
    print("高级白色瓶盖检测程序已启动")
    print("使用参数控制面板实时调节检测参数")
    print("快捷键: 'q'-退出, 's'-保存, 'i'-信息, 'p'-打印参数")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 缩放图像
        height, width = frame.shape[:2]
        new_width, new_height = width // 2, height // 2
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 检查ROI边界
        if (roi_x + roi_w <= new_width and roi_y + roi_h <= new_height):
            # 提取ROI区域
            roi_region = resized_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # 获取当前参数
            params = get_trackbar_values()
            
            # 检测白色瓶盖
            roi_with_detection, white_mask, roi_filtered, detected_caps = detect_white_caps(roi_region, params)
            
            # 在缩放后的图像上绘制ROI框
            cv2.rectangle(resized_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 2)
            cv2.putText(resized_frame, f"ROI: {len(detected_caps)} caps detected", 
                       (roi_x, roi_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 显示各种图像
            cv2.imshow('Camera Feed (Resized)', resized_frame)
            cv2.imshow('ROI Region with Detection', roi_with_detection)
            cv2.imshow('ROI Filtered', roi_filtered)
            cv2.imshow('White Cap Mask', white_mask)
            
            # 放大显示ROI
            roi_enlarged = cv2.resize(roi_with_detection, (roi_w*2, roi_h*2))
            cv2.imshow('ROI Region (Enlarged)', roi_enlarged)
        
        # 键盘事件处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前图像
            cv2.imwrite('roi_capture_advanced.png', roi_region)
            cv2.imwrite('roi_with_detection_advanced.png', roi_with_detection)
            cv2.imwrite('roi_filtered_advanced.png', roi_filtered)
            cv2.imwrite('white_cap_mask_advanced.png', white_mask)
            print("所有图像已保存（高级版本）")
        elif key == ord('i'):
            # 显示信息
            params = get_trackbar_values()
            print(f"\n=== 当前检测参数 ===")
            print(f"HSV范围: {params['lower_hsv']} - {params['upper_hsv']}")
            print(f"形态学核大小: {params['kernel_size']}, 迭代次数: {params['morph_iter']}")
            print(f"最小面积: {params['min_area']}, 圆形度阈值: {params['circularity']:.2f}")
            print(f"高斯核大小: {params['gaussian_size']}, 双边滤波d: {params['bilateral_d']}")
            if 'detected_caps' in locals():
                print(f"当前检测到 {len(detected_caps)} 个白色瓶盖")
        elif key == ord('p'):
            # 打印当前参数（用于代码复制）
            params = get_trackbar_values()
            print(f"\n=== 参数代码 ===")
            print(f"lower_white = np.array({params['lower_hsv'].tolist()})")
            print(f"upper_white = np.array({params['upper_hsv'].tolist()})")
            print(f"kernel_size = {params['kernel_size']}")
            print(f"morph_iter = {params['morph_iter']}")
            print(f"min_area = {params['min_area']}")
            print(f"circularity_threshold = {params['circularity']:.2f}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()