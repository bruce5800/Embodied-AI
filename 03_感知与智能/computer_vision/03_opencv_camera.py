import cv2

def main():
    """
    使用OpenCV打开摄像头并实时展示画面
    """
    # 创建VideoCapture对象，参数0表示默认摄像头
    cap = cv2.VideoCapture(2) #如果电脑上有多个摄像头，需要调整摄像头的id，默认值0
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，按 'q' 键退出")
    
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
        
        # 在窗口中显示缩小后的画面
        cv2.imshow('Camera Live View', resized_frame)
        
        # 检查按键输入，如果按下 'q' 键则退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已关闭，程序结束")

if __name__ == "__main__":
    main()