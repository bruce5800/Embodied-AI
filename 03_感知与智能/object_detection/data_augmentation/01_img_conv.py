import cv2
import numpy as np


def main():
    img_path = "img.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}. 请确认文件存在且路径正确。")
        return

    # 卷积核（5x5），与用户提供一致
    kernel = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, -4, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    # 使用 OpenCV 的 filter2D 进行卷积（该核是中心对称的，卷积与相关等价）
    convolved = cv2.filter2D(img, ddepth=-1, kernel=kernel)

    # 展示结果
    cv2.imshow("Original", img)
    cv2.imshow("Convolved", convolved)

    # 保存输出，便于后续查看
    out_path = "img_conv_out.jpg"
    cv2.imwrite(out_path, convolved)
    print(f"卷积后的图片已保存到: {out_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()