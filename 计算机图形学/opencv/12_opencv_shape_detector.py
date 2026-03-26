import cv2
import numpy as np
import os
from datetime import datetime

class ShapeDetector:
    def __init__(self, camera_id=2):
        """
        初始化形状检测器
        :param camera_id: 摄像头ID，默认为2（根据参考代码）
        """
        self.camera_id = camera_id
        self.cap = None
        self.background = None
        self.background_subtractor = None
        self.min_contour_area = 500  # 最小轮廓面积，过滤小噪声
        self.detection_method = "frame_diff"  # 默认使用帧差法
        self.background_gray = None  # 灰度背景图片
        self.prev_frame = None  # 前一帧，用于帧差法
        
        # 参数调整界面的参数
        self.show_parameter_window = False
        self.binary_threshold = 30      # 二值化阈值
        self.gaussian_blur_size = 5     # 高斯模糊核大小
        self.morph_close_size = 7       # 形态学闭运算核大小
        self.morph_open_size = 3        # 形态学开运算核大小
        self.var_threshold = 100        # MOG2方差阈值
        self.learning_rate = 0.001      # MOG2学习率
        
        # HSV颜色过滤参数（针对淡粉色优化）
        self.use_hsv_filter = False
        self.hsv_lower = [150, 30, 100]   # HSV下限 (粉色范围)
        self.hsv_upper = [180, 255, 255]  # HSV上限
        
    def initialize_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"错误：无法打开摄像头 {self.camera_id}")
            return False
            
        print(f"摄像头 {self.camera_id} 已成功打开")
        return True
    
    def create_parameter_window(self):
        """创建参数调整窗口"""
        cv2.namedWindow('Parameter Adjustment', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Parameter Adjustment', 400, 600)
        
        # 创建滑动条
        cv2.createTrackbar('Binary Threshold', 'Parameter Adjustment', self.binary_threshold, 100, self.on_binary_threshold_change)
        cv2.createTrackbar('Gaussian Blur', 'Parameter Adjustment', self.gaussian_blur_size, 15, self.on_gaussian_blur_change)
        cv2.createTrackbar('Morph Close Size', 'Parameter Adjustment', self.morph_close_size, 15, self.on_morph_close_change)
        cv2.createTrackbar('Morph Open Size', 'Parameter Adjustment', self.morph_open_size, 15, self.on_morph_open_change)
        cv2.createTrackbar('Var Threshold', 'Parameter Adjustment', self.var_threshold, 200, self.on_var_threshold_change)
        cv2.createTrackbar('Learning Rate x1000', 'Parameter Adjustment', int(self.learning_rate * 1000), 50, self.on_learning_rate_change)
        cv2.createTrackbar('Min Area', 'Parameter Adjustment', self.min_contour_area, 2000, self.on_min_area_change)
        
        # HSV颜色过滤滑动条
        cv2.createTrackbar('Use HSV Filter', 'Parameter Adjustment', int(self.use_hsv_filter), 1, self.on_hsv_filter_toggle)
        cv2.createTrackbar('HSV H Min', 'Parameter Adjustment', self.hsv_lower[0], 179, self.on_hsv_h_min_change)
        cv2.createTrackbar('HSV H Max', 'Parameter Adjustment', self.hsv_upper[0], 179, self.on_hsv_h_max_change)
        cv2.createTrackbar('HSV S Min', 'Parameter Adjustment', self.hsv_lower[1], 255, self.on_hsv_s_min_change)
        cv2.createTrackbar('HSV S Max', 'Parameter Adjustment', self.hsv_upper[1], 255, self.on_hsv_s_max_change)
        cv2.createTrackbar('HSV V Min', 'Parameter Adjustment', self.hsv_lower[2], 255, self.on_hsv_v_min_change)
        cv2.createTrackbar('HSV V Max', 'Parameter Adjustment', self.hsv_upper[2], 255, self.on_hsv_v_max_change)
        
        self.show_parameter_window = True
    
    # 滑动条回调函数
    def on_binary_threshold_change(self, val):
        self.binary_threshold = val
    
    def on_gaussian_blur_change(self, val):
        self.gaussian_blur_size = max(1, val if val % 2 == 1 else val + 1)  # 确保是奇数
    
    def on_morph_close_change(self, val):
        self.morph_close_size = max(1, val if val % 2 == 1 else val + 1)  # 确保是奇数
    
    def on_morph_open_change(self, val):
        self.morph_open_size = max(1, val if val % 2 == 1 else val + 1)  # 确保是奇数
    
    def on_var_threshold_change(self, val):
        self.var_threshold = val
        # 重新初始化背景减法器
        if self.background_subtractor is not None:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True,
                varThreshold=self.var_threshold,
                history=200)
            self.background_subtractor.setBackgroundRatio(0.9)
    
    def on_learning_rate_change(self, val):
        self.learning_rate = val / 1000.0
    
    def on_min_area_change(self, val):
        self.min_contour_area = val
    
    def on_hsv_filter_toggle(self, val):
        self.use_hsv_filter = bool(val)
    
    def on_hsv_h_min_change(self, val):
        self.hsv_lower[0] = val
    
    def on_hsv_h_max_change(self, val):
        self.hsv_upper[0] = val
    
    def on_hsv_s_min_change(self, val):
        self.hsv_lower[1] = val
    
    def on_hsv_s_max_change(self, val):
        self.hsv_upper[1] = val
    
    def on_hsv_v_min_change(self, val):
        self.hsv_lower[2] = val
    
    def on_hsv_v_max_change(self, val):
        self.hsv_upper[2] = val
    
    def capture_background(self):
        """拍摄背景图片"""
        if self.cap is None:
            print("错误：摄像头未初始化")
            return False
        print("准备拍摄背景图片...")
        print("请确保画面中没有任何物体，然后按 'c' 键拍摄背景，按 'q' 键退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                return False
            
            # 缩放画面
            height, width = frame.shape[:2]
            new_width = width // 2
            new_height = height // 2
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # 在画面上添加提示文字
            cv2.putText(resized_frame, "Press 'c' to capture background, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Background Capture', resized_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # 拍摄背景
                self.background = frame.copy()
                self.background_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
                
                # 保存背景图片
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                bg_filename = f"background_{timestamp}.jpg"
                cv2.imwrite(bg_filename, self.background)
                print(f"背景图片已保存为: {bg_filename}")
                
                # 初始化背景减法器（降低学习率，避免静止物体被吸收，优化影子检测）
                self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=True,    # 启用影子检测
                    varThreshold=self.var_threshold,      # 使用可调整的方差阈值
                    history=200)           # 增加历史帧数
                
                # 设置非常低的学习率，防止静止物体被学习为背景
                self.background_subtractor.setBackgroundRatio(0.9)  # 背景比例
                
                # 用背景图片训练背景减法器
                for _ in range(50):  # 增加训练次数以稳定背景模型
                    self.background_subtractor.apply(self.background, learningRate=0.01)  # 使用很低的学习率
                
                cv2.destroyWindow('Background Capture')
                print("背景拍摄完成！")
                return True
                
            elif key == ord('q'):
                cv2.destroyWindow('Background Capture')
                return False
    
    def detect_objects(self):
        """实时检测物体"""
        if self.background is None:
            print("错误：请先拍摄背景图片")
            return
            
        print("开始物体检测...")
        print("按 '1' 键切换到帧差法，按 '2' 键切换到背景减法")
        print("按 'r' 键重新拍摄背景，按 's' 键保存当前检测结果，按 'q' 键退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                break
            
            # 根据选择的方法进行物体检测
            if self.detection_method == "frame_diff":
                fg_mask = self._frame_difference_detection(frame)
            else:  # background_subtraction
                fg_mask = self._background_subtraction_detection(frame)
            
            # 形态学操作去除噪声（使用可调整参数）
            kernel_close = np.ones((self.morph_close_size, self.morph_close_size), np.uint8)
            kernel_open = np.ones((self.morph_open_size, self.morph_open_size), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            
            # 高斯模糊进一步去噪
            fg_mask = cv2.GaussianBlur(fg_mask, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
            
            # 二值化处理
            _, fg_mask = cv2.threshold(fg_mask, self.binary_threshold, 255, cv2.THRESH_BINARY)
            
            # 查找轮廓
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建结果图像
            result_frame = frame.copy()
            object_count = 0
            
            # 处理每个轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 过滤小面积轮廓
                if area > self.min_contour_area:
                    object_count += 1
                    
                    # 识别形状
                    shape_name, shape_confidence = self._identify_shape(contour)
                    
                    # 绘制轮廓
                    cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)
                    
                    # 获取边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # 标记物体编号、形状和面积
                    cv2.putText(result_frame, f"Object {object_count}: {shape_name}", 
                               (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(result_frame, f"Confidence: {shape_confidence:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(result_frame, f"Area: {int(area)}", 
                               (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 添加统计信息和方法显示
            method_text = "Frame Diff" if self.detection_method == "frame_diff" else "Background Sub"
            param_text = f"Params - Blur:{self.gaussian_blur_size} Thresh:{self.binary_threshold} Area:{self.min_contour_area}"
            hsv_text = f"HSV Filter: {'ON' if self.use_hsv_filter else 'OFF'}"
            
            cv2.putText(result_frame, f"Method: {method_text} | Objects: {object_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(result_frame, param_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_frame, hsv_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示操作提示
            cv2.putText(result_frame, "Press 'p' for parameter window, 'h' for HSV toggle", 
                       (10, result_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(result_frame, "Press '1':Frame Diff, '2':BG Sub, 'r':Recapture, 's':Save, 'q':Quit", 
                       (10, result_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 缩放显示
            height, width = result_frame.shape[:2]
            new_width = width // 2
            new_height = height // 2
            resized_result = cv2.resize(result_frame, (new_width, new_height))
            resized_mask = cv2.resize(fg_mask, (new_width, new_height))
            
            # 显示结果
            cv2.imshow('Object Detection', resized_result)
            cv2.imshow('Foreground Mask', resized_mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出物体检测")
                break
            elif key == ord('p'):
                # 打开/关闭参数调整窗口
                if not self.show_parameter_window:
                    self.create_parameter_window()
                else:
                    cv2.destroyWindow('Parameter Adjustment')
                    self.show_parameter_window = False
            elif key == ord('h'):
                # 切换HSV过滤
                self.use_hsv_filter = not self.use_hsv_filter
                print(f"HSV过滤: {'开启' if self.use_hsv_filter else '关闭'}")
            elif key == ord('1'):
                self.detection_method = "frame_diff"
                print("切换到帧差法检测")
            elif key == ord('2'):
                self.detection_method = "background_subtraction"
                print("切换到背景减法检测")
            elif key == ord('r'):
                print("重新拍摄背景...")
                cv2.destroyAllWindows()
                if self.capture_background():
                    print("背景重新拍摄完成，继续检测...")
                else:
                    break
            elif key == ord('s'):
                # 保存检测结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_filename = f"detection_result_{timestamp}.jpg"
                cv2.imwrite(result_filename, result_frame)
                print(f"检测结果已保存为: {result_filename}")
    
    def _frame_difference_detection(self, frame):
        """帧差法检测物体"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 与背景图片做差分
        diff = cv2.absdiff(self.background_gray, gray)
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(diff, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        
        # 二值化
        _, binary = cv2.threshold(blurred, self.binary_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪声
        kernel_close = np.ones((self.morph_close_size, self.morph_close_size), np.uint8)
        kernel_open = np.ones((self.morph_open_size, self.morph_open_size), np.uint8)
        
        # 闭运算填充物体内部的小洞
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        # 开运算去除小的噪声点
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        return opened
    
    def _background_subtraction_detection(self, frame):
        """背景减法检测物体（使用极低学习率）"""
        if self.background_subtractor is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # 使用极低的学习率进行背景减法，防止静止物体被吸收
        fg_mask = self.background_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # 处理影子：MOG2中影子像素值通常为127，前景为255，背景为0
        # 创建一个只保留前景物体的掩码
        fg_only_mask = np.zeros_like(fg_mask)
        fg_only_mask[fg_mask == 255] = 255  # 只保留前景物体，去除影子
        
        # 额外的影子过滤：基于颜色空间的影子检测
        shadow_filtered_mask = self._filter_shadows_by_color(frame, fg_only_mask)
        
        # 如果启用HSV颜色过滤，进行颜色过滤
        if self.use_hsv_filter:
            hsv_filtered_mask = self._apply_hsv_filter(frame, shadow_filtered_mask)
            return hsv_filtered_mask
        
        return shadow_filtered_mask
    
    def _filter_shadows_by_color(self, frame, fg_mask):
        """
        基于颜色空间的影子过滤
        影子通常具有较低的亮度但保持相似的色调
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建结果掩码
        result_mask = fg_mask.copy()
        
        # 获取前景区域
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 创建单个轮廓的掩码
            single_mask = np.zeros_like(fg_mask)
            cv2.drawContours(single_mask, [contour], -1, 255, -1)
            
            # 计算该区域的平均HSV值
            masked_hsv = cv2.bitwise_and(hsv, hsv, mask=single_mask)
            mean_hsv = cv2.mean(masked_hsv, mask=single_mask)
            
            # 影子检测：亮度(V)较低但饱和度(S)不会太高
            if mean_hsv[2] < 80 and mean_hsv[1] < 100:  # V < 80 且 S < 100 可能是影子
                # 从结果中移除这个轮廓
                cv2.drawContours(result_mask, [contour], -1, 0, -1)
        
        return result_mask
    
    def _apply_hsv_filter(self, frame, mask):
        """
        应用HSV颜色过滤，专门针对淡粉色物体优化
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建HSV颜色范围掩码
        lower_hsv = np.array(self.hsv_lower)
        upper_hsv = np.array(self.hsv_upper)
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # 将HSV掩码与原始掩码结合
        # 只保留既在原始掩码中又在HSV颜色范围内的像素
        combined_mask = cv2.bitwise_and(mask, hsv_mask)
        
        # 对结果进行形态学操作以去除噪声
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def _identify_shape(self, contour):
        """
        识别轮廓的形状
        :param contour: 输入轮廓
        :return: (形状名称, 置信度)
        """
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # 轮廓近似，减少顶点数量
        epsilon = 0.02 * perimeter  # 近似精度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 获取顶点数量
        vertices = len(approx)
        
        # 计算轮廓面积和边界框
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算长宽比
        aspect_ratio = float(w) / h
        
        # 计算轮廓面积与边界框面积的比值
        extent = float(area) / (w * h)
        
        # 计算圆形度（4π*面积/周长²）
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 形状识别逻辑
        shape_name = "Unknown"
        confidence = 0.0
        
        # 圆形检测
        if circularity > 0.7 and 0.7 < aspect_ratio < 1.3:
            shape_name = "Circle"
            confidence = min(circularity, 1.0)
        
        # 矩形检测（包括正方形和长方形）
        elif vertices == 4:
            # 计算四个角的角度
            angles = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % 4][0]
                p3 = approx[(i + 2) % 4][0]
                
                # 计算向量
                v1 = p1 - p2
                v2 = p3 - p2
                
                # 计算角度
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
            
            # 检查是否接近90度
            right_angles = sum(1 for angle in angles if 80 < angle < 100)
            
            if right_angles >= 3:  # 至少3个直角
                if 0.8 < aspect_ratio < 1.2:  # 接近正方形
                    shape_name = "Square"
                    confidence = 1.0 - abs(1.0 - aspect_ratio)  # 越接近1:1比例，置信度越高
                else:  # 长方形
                    shape_name = "Rectangle"
                    confidence = 0.8 + 0.2 * extent  # 基于填充度调整置信度
        
        # 三角形检测
        elif vertices == 3:
            shape_name = "Triangle"
            confidence = 0.7 + 0.3 * extent
        
        # 多边形检测
        elif vertices > 4:
            if circularity > 0.6:
                shape_name = "Circle"  # 可能是圆形但近似为多边形
                confidence = circularity * 0.8
            else:
                shape_name = f"Polygon({vertices})"
                confidence = 0.6
        
        # 如果置信度太低，标记为未知
        if confidence < 0.3:
            shape_name = "Unknown"
            confidence = 0.0
        
        return shape_name, confidence
    
    def run(self):
        """运行主程序"""
        print("=== OpenCV 差分物体检测器 ===")
        print("程序功能：先拍摄空白背景，然后实时检测画面中新增的物体")
        
        # 初始化摄像头
        if not self.initialize_camera():
            return
        
        try:
            # 拍摄背景
            if self.capture_background():
                # 开始物体检测
                self.detect_objects()
            else:
                print("背景拍摄取消，程序退出")
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"程序运行出错: {e}")
        finally:
            # 清理资源
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("摄像头已关闭，程序结束")

def main():
    """主函数"""
    # 创建检测器实例（使用摄像头ID=2，与参考代码保持一致）
    detector = ShapeDetector(camera_id=2)
    
    # 运行检测器
    detector.run()

if __name__ == "__main__":
    main()