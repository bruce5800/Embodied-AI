import cv2
import numpy as np

# 全局变量存储HSV范围
hsv_lower = [0, 0, 0]
hsv_upper = [179, 255, 255]

def nothing(val):
    """滑动条回调函数"""
    pass

def update_hsv_range():
    """更新HSV范围值"""
    global hsv_lower, hsv_upper
    
    # 获取滑动条的值
    hsv_lower[0] = cv2.getTrackbarPos('H Min', 'HSV Controls')
    hsv_lower[1] = cv2.getTrackbarPos('S Min', 'HSV Controls')
    hsv_lower[2] = cv2.getTrackbarPos('V Min', 'HSV Controls')
    
    hsv_upper[0] = cv2.getTrackbarPos('H Max', 'HSV Controls')
    hsv_upper[1] = cv2.getTrackbarPos('S Max', 'HSV Controls')
    hsv_upper[2] = cv2.getTrackbarPos('V Max', 'HSV Controls')

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 创建窗口
    cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('HSV', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Result', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('HSV Controls', cv2.WINDOW_AUTOSIZE)
    
    # 创建一个空白图像用于控制面板
    control_panel = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.imshow('HSV Controls', control_panel)
    
    # 创建滑动条
    # H (色调) 范围: 0-179
    cv2.createTrackbar('H Min', 'HSV Controls', 0, 179, nothing)
    cv2.createTrackbar('H Max', 'HSV Controls', 179, 179, nothing)
    
    # S (饱和度) 范围: 0-255
    cv2.createTrackbar('S Min', 'HSV Controls', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'HSV Controls', 255, 255, nothing)
    
    # V (亮度) 范围: 0-255
    cv2.createTrackbar('V Min', 'HSV Controls', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'HSV Controls', 255, 255, nothing)
    
    print("HSV颜色提取器已启动!")
    print("使用滑动条调整HSV范围来提取目标颜色")
    print("按 'q' 键退出程序")
    print("按 'p' 键打印当前HSV范围值")
    print("按 'r' 键重置HSV范围")
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头数据")
            break
        
        # 调整帧大小
        frame = cv2.resize(frame, (640, 480))
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 更新HSV范围
        update_hsv_range()
        
        # 创建掩码
        lower_bound = np.array(hsv_lower)
        upper_bound = np.array(hsv_upper)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 应用掩码到原图像
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 创建控制面板显示当前HSV值
        control_panel = np.zeros((300, 600, 3), dtype=np.uint8)
        
        # 显示当前HSV范围值
        cv2.putText(control_panel, f'HSV Lower: [{hsv_lower[0]}, {hsv_lower[1]}, {hsv_lower[2]}]', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(control_panel, f'HSV Upper: [{hsv_upper[0]}, {hsv_upper[1]}, {hsv_upper[2]}]', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示操作说明
        cv2.putText(control_panel, 'Controls:', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(control_panel, 'q - Quit', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_panel, 'p - Print HSV values', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_panel, 'r - Reset HSV range', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示颜色样本
        color_sample = np.full((50, 100, 3), 
                              (int(hsv_upper[2]), int(hsv_upper[1]), int((hsv_lower[0] + hsv_upper[0])/2)), 
                              dtype=np.uint8)
        color_sample_bgr = cv2.cvtColor(color_sample, cv2.COLOR_HSV2BGR)
        control_panel[200:250, 10:110] = color_sample_bgr
        cv2.putText(control_panel, 'Color Sample', (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示检测到的像素数量
        detected_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = (detected_pixels / total_pixels) * 100
        cv2.putText(control_panel, f'Detected: {detected_pixels} pixels ({percentage:.1f}%)', 
                   (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 显示所有窗口
        cv2.imshow('Original', frame)
        cv2.imshow('HSV', hsv)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)
        cv2.imshow('HSV Controls', control_panel)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"当前HSV范围:")
            print(f"Lower: [{hsv_lower[0]}, {hsv_lower[1]}, {hsv_lower[2]}]")
            print(f"Upper: [{hsv_upper[0]}, {hsv_upper[1]}, {hsv_upper[2]}]")
            print(f"检测到的像素: {detected_pixels} ({percentage:.1f}%)")
        elif key == ord('r'):
            # 重置HSV范围
            cv2.setTrackbarPos('H Min', 'HSV Controls', 0)
            cv2.setTrackbarPos('H Max', 'HSV Controls', 179)
            cv2.setTrackbarPos('S Min', 'HSV Controls', 0)
            cv2.setTrackbarPos('S Max', 'HSV Controls', 255)
            cv2.setTrackbarPos('V Min', 'HSV Controls', 0)
            cv2.setTrackbarPos('V Max', 'HSV Controls', 255)
            print("HSV范围已重置")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()