import cv2

def main():
    """
    使用OpenCV和Haar级联分类器进行实时人脸检测
    """
    # 创建VideoCapture对象，参数0表示默认摄像头
    cap = cv2.VideoCapture(1)  # 如果电脑上有多个摄像头，需要调整摄像头的id，默认值0
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 加载Haar级联分类器用于人脸检测
    # 使用OpenCV内置的人脸检测分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 检查分类器是否成功加载
    if face_cascade.empty():
        print("错误：无法加载人脸检测分类器")
        return
    
    print("摄像头已打开，人脸检测已启动，按 'q' 键退出")
    
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
        
        # 将图像转换为灰度图，因为Haar分类器在灰度图上工作更好
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # 使用Haar级联分类器检测人脸
        # detectMultiScale参数说明：
        # - scaleFactor: 图像缩放因子，用于创建图像金字塔
        # - minNeighbors: 每个候选矩形应该保留的邻居数量
        # - minSize: 检测窗口的最小尺寸
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 在检测到的人脸周围绘制矩形框
        for (x, y, w, h) in faces:
            # 绘制绿色矩形框
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 在矩形框上方添加文字标签
            cv2.putText(resized_frame, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 在窗口左上角显示检测到的人脸数量
        face_count_text = f'Faces detected: {len(faces)}'
        cv2.putText(resized_frame, face_count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 在窗口中显示处理后的画面
        cv2.imshow('Face Detection - Camera Live View', resized_frame)
        
        # 检查按键输入，如果按下 'q' 键则退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已关闭，人脸检测程序结束")

if __name__ == "__main__":
    main()