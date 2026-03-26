import cv2
import numpy as np

class ROISelector:
    def __init__(self):
        self.drawing = False  # 是否正在绘制
        self.start_point = None  # 起始点
        self.end_point = None    # 结束点
        self.roi_selected = False  # 是否已选择ROI
        self.roi_rect = None     # ROI矩形区域 (x, y, w, h)
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 鼠标左键按下，开始绘制
            self.drawing = True
            self.start_point = (x, y)
            self.roi_selected = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 鼠标移动，更新结束点
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 鼠标左键释放，完成绘制
            self.drawing = False
            self.end_point = (x, y)
            
            # 计算ROI矩形区域
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                # 确保坐标正确（左上角和右下角）
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                
                # 确保ROI区域有效（宽度和高度大于0）
                if x_max > x_min and y_max > y_min:
                    self.roi_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                    self.roi_selected = True
                    print(f"ROI区域已选择: x={x_min}, y={y_min}, w={x_max-x_min}, h={y_max-y_min}")
                    
    def draw_roi_rectangle(self, frame):
        """在画面上绘制ROI选择框"""
        if self.drawing and self.start_point and self.end_point:
            # 正在绘制时显示临时矩形
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 0), 2)
        elif self.roi_selected and self.roi_rect:
            # ROI已选择时显示确定的矩形
            x, y, w, h = self.roi_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # 添加文本标签
            cv2.putText(frame, "ROI Selected", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame
        
    def get_roi_image(self, frame):
        """从原始画面中提取ROI区域"""
        if self.roi_selected and self.roi_rect:
            x, y, w, h = self.roi_rect
            # 确保ROI区域在画面范围内
            frame_height, frame_width = frame.shape[:2]
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = min(w, frame_width - x)
            h = min(h, frame_height - y)
            
            if w > 0 and h > 0:
                return frame[y:y+h, x:x+w]
        return None

def main():
    """
    使用OpenCV打开摄像头并实现ROI选择功能
    """
    # 创建VideoCapture对象
    cap = cv2.VideoCapture(2)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 创建ROI选择器
    roi_selector = ROISelector()
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow('Camera Live View')
    cv2.setMouseCallback('Camera Live View', roi_selector.mouse_callback)
    
    print("摄像头已打开")
    print("使用说明：")
    print("- 用鼠标拖拽选择感兴趣的区域(ROI)")
    print("- 按 'r' 键重置ROI选择")
    print("- 按 'q' 键退出程序")
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        # 获取原始画面的尺寸并缩小为1/2
        height, width = frame.shape[:2]
        new_width = width // 2
        new_height = height // 2
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 在缩小的画面上绘制ROI选择框
        display_frame = roi_selector.draw_roi_rectangle(resized_frame.copy())
        
        # 显示主画面
        cv2.imshow('Camera Live View', display_frame)
        
        # 如果选择了ROI，显示ROI区域
        roi_image = roi_selector.get_roi_image(resized_frame)
        if roi_image is not None:
            # 将ROI图像放大以便更好地查看
            roi_height, roi_width = roi_image.shape[:2]
            if roi_width > 0 and roi_height > 0:
                # 计算放大倍数，确保ROI窗口不会太小
                scale_factor = max(1, min(300 // roi_width, 300 // roi_height))
                enlarged_roi = cv2.resize(roi_image, 
                                        (roi_width * scale_factor, roi_height * scale_factor))
                cv2.imshow('ROI Region', enlarged_roi)
        
        # 检查按键输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('r'):
            # 重置ROI选择
            roi_selector.roi_selected = False
            roi_selector.roi_rect = None
            roi_selector.drawing = False
            roi_selector.start_point = None
            roi_selector.end_point = None
            cv2.destroyWindow('ROI Region')
            print("ROI选择已重置")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭，程序结束")

if __name__ == "__main__":
    main()