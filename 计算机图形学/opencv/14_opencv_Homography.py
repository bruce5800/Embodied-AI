import cv2
import numpy as np
import os

class PerspectiveTransformer:
    def __init__(self):
        self.points = []  # 存储用户选择的4个点
        self.image = None
        self.original_image = None
        self.window_name = "Perspective Transformer"
        self.result_window = "Transformed Result"
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，处理用户点击选择顶点"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"选择了第{len(self.points)}个点: ({x}, {y})")
                
                # 在图像上绘制选择的点
                cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.image, str(len(self.points)), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 如果选择了多个点，绘制连接线
                if len(self.points) > 1:
                    cv2.line(self.image, self.points[-2], self.points[-1], (0, 255, 0), 2)
                
                # 如果选择了4个点，闭合多边形
                if len(self.points) == 4:
                    cv2.line(self.image, self.points[-1], self.points[0], (0, 255, 0), 2)
                    print("已选择4个点，按 't' 键进行透视变换，按 'r' 键重置")
                
                cv2.imshow(self.window_name, self.image)
    
    def reset_points(self):
        """重置选择的点"""
        self.points = []
        self.image = self.original_image.copy()
        cv2.imshow(self.window_name, self.image)
        print("已重置选择的点")
    
    def order_points(self, pts):
        """
        对4个点进行排序，确保顺序为：左上、右上、右下、左下
        """
        # 将点转换为numpy数组
        pts = np.array(pts, dtype=np.float32)
        
        # 计算每个点的坐标和
        sum_pts = pts.sum(axis=1)
        # 左上角的点坐标和最小，右下角的点坐标和最大
        top_left = pts[np.argmin(sum_pts)]
        bottom_right = pts[np.argmax(sum_pts)]
        
        # 计算每个点的坐标差
        diff_pts = np.diff(pts, axis=1)
        # 右上角的点x-y最大，左下角的点x-y最小
        top_right = pts[np.argmax(diff_pts)]
        bottom_left = pts[np.argmin(diff_pts)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    def transform_perspective(self, width=400, height=600):
        """
        执行透视变换，并进行后处理（90度逆时针旋转 + 左右镜像）
        """
        if len(self.points) != 4:
            print("错误：需要选择4个点才能进行透视变换")
            return
        
        # 对选择的点进行排序
        ordered_points = self.order_points(self.points)
        
        # 定义目标矩形的4个角点（俯视图）
        dst_points = np.array([
            [0, 0],                    # 左上
            [width - 1, 0],            # 右上
            [width - 1, height - 1],   # 右下
            [0, height - 1]            # 左下
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(ordered_points, dst_points)
        
        # 应用透视变换
        transformed = cv2.warpPerspective(self.original_image, matrix, (width, height))
        
        # 后处理：90度逆时针旋转
        # cv2.ROTATE_90_COUNTERCLOCKWISE 表示逆时针旋转90度
        rotated = cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE)
        
        # 后处理：左右镜像翻转
        # cv2.flip(image, 1) 表示水平翻转（左右镜像）
        final_result = cv2.flip(rotated, 1)
        
        # 显示最终结果
        cv2.imshow(self.result_window, final_result)
        
        # 同时显示处理步骤（可选）
        cv2.imshow("Step 1: Perspective Transform", transformed)
        cv2.imshow("Step 2: Rotated 90° CCW", rotated)
        cv2.imshow("Step 3: Final (Mirrored)", final_result)
        
        # 保存结果
        output_filename = "transformed_result.jpg"
        cv2.imwrite(output_filename, final_result)
        print(f"透视变换完成！")
        print(f"已应用后处理：90度逆时针旋转 + 左右镜像")
        print(f"最终结果已保存为: {output_filename}")
        
        return final_result
    
    def load_image(self, image_path):
        """加载图像"""
        if not os.path.exists(image_path):
            print(f"错误：图像文件不存在: {image_path}")
            return False
        
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"错误：无法加载图像: {image_path}")
            return False
        
        # 调整图像大小以适应屏幕
        height, width = self.original_image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.original_image = cv2.resize(self.original_image, (new_width, new_height))
        
        self.image = self.original_image.copy()
        return True
    
    def run(self, image_path):
        """运行透视变换工具"""
        if not self.load_image(image_path):
            return
        
        print("=== 透视变换工具 ===")
        print("操作说明：")
        print("1. 用鼠标左键依次点击4个角点（建议按顺时针或逆时针顺序）")
        print("2. 选择完4个点后，按 't' 键进行透视变换")
        print("3. 透视变换后会自动应用后处理：90度逆时针旋转 + 左右镜像")
        print("4. 按 'r' 键重置选择的点")
        print("5. 按 's' 键保存当前选择状态的图像")
        print("6. 按 'q' 键退出程序")
        print("=" * 50)
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 显示原始图像
        cv2.imshow(self.window_name, self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("退出程序")
                break
            elif key == ord('r'):
                self.reset_points()
            elif key == ord('t'):
                if len(self.points) == 4:
                    self.transform_perspective()
                else:
                    print(f"需要选择4个点，当前已选择{len(self.points)}个点")
            elif key == ord('s'):
                save_filename = "selected_points.jpg"
                cv2.imwrite(save_filename, self.image)
                print(f"当前状态已保存为: {save_filename}")
        
        cv2.destroyAllWindows()

def main():
    """主函数"""
    # 创建透视变换器实例
    transformer = PerspectiveTransformer()
    
    # 检查是否有abc.jpg文件，如果有就使用它，否则提示用户
    default_image = "abc.jpg"
    
    if os.path.exists(default_image):
        print(f"找到默认图像: {default_image}")
        transformer.run(default_image)
    else:
        print("未找到默认图像文件 abc.jpg")
        print("请将要处理的图像文件放在当前目录下，并命名为 abc.jpg")
        print("或者修改代码中的图像路径")
        
        # 尝试使用其他可能存在的图像文件
        possible_images = ["background_20250927_144028.jpg", "background_20250927_144942.jpg", "green_image_with_red_line.png"]
        
        for img_file in possible_images:
            if os.path.exists(img_file):
                print(f"找到图像文件: {img_file}，使用此文件进行演示")
                transformer.run(img_file)
                break
        else:
            print("未找到任何可用的图像文件")

if __name__ == "__main__":
    main()