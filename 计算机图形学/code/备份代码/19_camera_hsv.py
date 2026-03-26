import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# 颜色范围
color_ranges = {
    "Red": [
        (np.array([0, 120, 80]), np.array([10, 255, 255])),
        (np.array([170, 120, 80]), np.array([180, 255, 255]))
    ],
    "Green": [
        (np.array([35, 80, 80]), np.array([85, 255, 255]))
    ],
    "Blue": [
        (np.array([90, 80, 80]), np.array([130, 255, 255]))
    ]
}

color_map = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0)
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, ranges in color_ranges.items():
        mask_total = None

        # 处理红色双区间
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else mask_total + mask

        # 去噪（很关键）
        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_DILATE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 1000:  # ⚠️ 可以调大一点更稳定
                x, y, w, h = cv2.boundingRect(cnt)

                # 画框
                cv2.rectangle(display, (x, y), (x + w, y + h), color_map[color_name], 2)

                # 标注颜色
                cv2.putText(display, color_name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_map[color_name],
                            2)

    cv2.imshow("Auto Color Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()