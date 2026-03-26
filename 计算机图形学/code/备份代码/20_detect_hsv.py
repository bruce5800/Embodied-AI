import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control", 800, 600)

def nothing(x):
    pass

# 创建滑动条
cv2.createTrackbar("H_min", "Control", 0, 179, nothing)
cv2.createTrackbar("H_max", "Control", 179, 179, nothing)
cv2.createTrackbar("S_min", "Control", 0, 255, nothing)
cv2.createTrackbar("S_max", "Control", 255, 255, nothing)
cv2.createTrackbar("V_min", "Control", 0, 255, nothing)
cv2.createTrackbar("V_max", "Control", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 获取滑动条值
    h_min = cv2.getTrackbarPos("H_min", "Control")
    h_max = cv2.getTrackbarPos("H_max", "Control")
    s_min = cv2.getTrackbarPos("S_min", "Control")
    s_max = cv2.getTrackbarPos("S_max", "Control")
    v_min = cv2.getTrackbarPos("V_min", "Control")
    v_max = cv2.getTrackbarPos("V_max", "Control")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 缩小画面（提高流畅度）
    frame_small = cv2.resize(frame, (320, 240))
    mask_small = cv2.resize(mask, (320, 240))
    result_small = cv2.resize(result, (320, 240))

    # mask 是单通道，转成3通道方便拼接
    mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

    # 生成滑动条可视化（显示当前HSV范围）
    hsv_bar = np.zeros((240, 320, 3), dtype=np.uint8)
    hsv_bar[:] = ((h_min + h_max) // 2, (s_min + s_max) // 2, (v_min + v_max) // 2)
    hsv_bar = cv2.cvtColor(hsv_bar, cv2.COLOR_HSV2BGR)

    # 在滑动条画面上显示数值
    cv2.putText(hsv_bar, f"H:{h_min}-{h_max}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(hsv_bar, f"S:{s_min}-{s_max}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(hsv_bar, f"V:{v_min}-{v_max}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # 拼接成一个画面（上：原图+mask，下：result+HSV显示）
    top = np.hstack((frame_small, mask_small))
    bottom = np.hstack((result_small, hsv_bar))
    combined = np.vstack((top, bottom))

    cv2.imshow("Control", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()