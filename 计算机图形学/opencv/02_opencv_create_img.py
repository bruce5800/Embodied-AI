import cv2
import numpy as np

# 创建一个480x320分辨率的绿色图片
# 注意：OpenCV中图片的形状是(height, width, channels)
height = 320
width = 480
channels = 3

# 创建绿色图片 (BGR格式，绿色是[0, 255, 0])
img = np.zeros((height, width, channels), dtype=np.uint8)
img[:, :] = [255, 0, 0]  # 设置为绿色

# 在图片中间画一条红色的线
# 计算中间位置
center_y = height // 2
start_point = (0, center_y)  # 线的起点 (x, y)
end_point = (width, center_y)  # 线的终点 (x, y)
color = (0, 0, 255)  # 红色 (BGR格式)
thickness = 2  # 线的粗细

# 画线
cv2.line(img, start_point, end_point, color, thickness)

# 显示图片
cv2.imshow('Green Image with Red Line', img)

# 保存图片
cv2.imwrite('green_image_with_red_line.png', img)

print("图片已创建并保存为 'green_image_with_red_line.png'")
print("按任意键关闭窗口...")

# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()