#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main()
{
    /*
     * 使用OpenCV打开摄像头并实时展示画面，同时进行蓝色积木块的颜色分拣
     * 增强功能：显示真实边框、中心点坐标和旋转角度
     */
    
    // 创建VideoCapture对象，参数2表示摄像头ID
    VideoCapture cap(2); // 如果电脑上有多个摄像头，需要调整摄像头的id，默认值0
    
    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {
        cout << "错误：无法打开摄像头" << endl;
        return -1;
    }
    
    cout << "摄像头已打开，蓝色积木块检测程序启动（增强版）" << endl;
    cout << "功能：真实边框包裹 + 中心点坐标 + 旋转角度" << endl;
    cout << "按 'q' 键退出" << endl;
    
    // 定义蓝色在HSV颜色空间中的范围
    Scalar lower_blue(100, 50, 50);   // 蓝色下限
    Scalar upper_blue(130, 255, 255); // 蓝色上限
    
    while (true) {
        Mat frame;
        // 读取一帧画面
        bool ret = cap.read(frame);
        
        // 检查是否成功读取到画面
        if (!ret) {
            cout << "错误：无法读取摄像头画面" << endl;
            break;
        }
        
        // 获取原始画面的尺寸
        int height = frame.rows;
        int width = frame.cols;
        
        // 将画面缩小为原来的1/2
        int new_width = width / 2;
        int new_height = height / 2;
        Mat resized_frame;
        resize(frame, resized_frame, Size(new_width, new_height));
        
        // 将BGR颜色空间转换为HSV颜色空间
        Mat hsv;
        cvtColor(resized_frame, hsv, COLOR_BGR2HSV);
        
        // 创建蓝色掩码
        Mat mask;
        inRange(hsv, lower_blue, upper_blue, mask);
        
        // 进行形态学操作，去除噪声
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        
        // 寻找轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // 创建结果图像的副本
        Mat result_frame = resized_frame.clone();
        
        // 处理检测到的轮廓
        int blue_objects_count = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            // 计算轮廓面积，过滤掉太小的区域
            double area = contourArea(contours[i]);
            if (area > 500) { // 最小面积阈值
                blue_objects_count++;
                
                // 方法1：绘制轮廓的真实边框（轮廓本身）
                drawContours(result_frame, contours, (int)i, Scalar(0, 255, 0), 2);
                
                // 方法2：计算最小外接矩形（带旋转角度）
                RotatedRect rect = minAreaRect(contours[i]);
                Point2f vertices[4];
                rect.points(vertices);
                
                // 将Point2f转换为Point并绘制最小外接矩形
                vector<Point> box_points;
                for (int j = 0; j < 4; j++) {
                    box_points.push_back(Point((int)vertices[j].x, (int)vertices[j].y));
                }
                vector<vector<Point>> box_contour = {box_points};
                drawContours(result_frame, box_contour, 0, Scalar(255, 0, 0), 2);
                
                // 获取矩形的中心点、尺寸和角度
                Point2f center = rect.center;
                Size2f size = rect.size;
                float angle = rect.angle;
                int center_x = (int)center.x;
                int center_y = (int)center.y;
                
                // 计算轮廓的质心（更精确的中心点）
                Moments M = moments(contours[i]);
                int cx, cy;
                if (M.m00 != 0) {
                    cx = (int)(M.m10 / M.m00);
                    cy = (int)(M.m01 / M.m00);
                } else {
                    cx = center_x;
                    cy = center_y;
                }
                
                // 绘制中心点
                circle(result_frame, Point(cx, cy), 8, Scalar(0, 0, 255), -1);
                circle(result_frame, Point(cx, cy), 12, Scalar(255, 255, 255), 2);
                
                // 绘制角度指示线
                // 将角度转换为弧度
                double angle_rad = angle * M_PI / 180.0;
                int line_length = 50;
                
                // 计算指示线的终点
                int end_x = (int)(cx + line_length * cos(angle_rad));
                int end_y = (int)(cy + line_length * sin(angle_rad));
                
                // 绘制角度指示线
                arrowedLine(result_frame, Point(cx, cy), Point(end_x, end_y), Scalar(255, 255, 0), 3);
                
                // 添加文本标签
                int label_y_offset = 0;
                
                // 显示积木块编号
                string label = "Block " + to_string(blue_objects_count);
                putText(result_frame, label, Point(cx - 50, cy - 60 + label_y_offset), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
                label_y_offset += 20;
                
                // 显示中心点坐标
                string center_text = "Center: (" + to_string(cx) + ", " + to_string(cy) + ")";
                putText(result_frame, center_text, Point(cx - 50, cy - 60 + label_y_offset), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                label_y_offset += 15;
                
                // 显示旋转角度
                string angle_text = "Angle: " + to_string((int)angle) + "°";
                putText(result_frame, angle_text, Point(cx - 50, cy - 60 + label_y_offset), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                label_y_offset += 15;
                
                // 显示面积
                string area_text = "Area: " + to_string((int)area);
                putText(result_frame, area_text, Point(cx - 50, cy - 60 + label_y_offset), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                // 在图像左上角显示详细信息
                int info_start_y = 30 + (blue_objects_count - 1) * 80;
                string block_header = "=== Block " + to_string(blue_objects_count) + " ===";
                putText(result_frame, block_header, Point(10, info_start_y), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
                
                string center_info = "Center: (" + to_string(cx) + ", " + to_string(cy) + ")";
                putText(result_frame, center_info, Point(10, info_start_y + 20), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                string rotation_info = "Rotation: " + to_string((int)angle) + "°";
                putText(result_frame, rotation_info, Point(10, info_start_y + 35), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                string area_info = "Area: " + to_string((int)area) + " px²";
                putText(result_frame, area_info, Point(10, info_start_y + 50), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                string size_info = "Size: " + to_string((int)size.width) + "x" + to_string((int)size.height);
                putText(result_frame, size_info, Point(10, info_start_y + 65), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }
        
        // 在图像底部显示检测到的蓝色积木块总数
        string total_info = "Total Blue Blocks Detected: " + to_string(blue_objects_count);
        putText(result_frame, total_info, Point(10, result_frame.rows - 20), 
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        
        // 添加图例说明
        int legend_x = result_frame.cols - 250;
        putText(result_frame, "Legend:", Point(legend_x, 30), 
               FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        putText(result_frame, "Green: Contour", Point(legend_x, 50), 
               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        putText(result_frame, "Blue: Min Area Rect", Point(legend_x, 65), 
               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
        putText(result_frame, "Red: Center Point", Point(legend_x, 80), 
               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        putText(result_frame, "Yellow: Rotation", Point(legend_x, 95), 
               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        
        // 显示原始画面和处理结果
        imshow("Blue Block Detection - Enhanced", result_frame);
        imshow("Blue Mask", mask);
        
        // 检查按键输入，如果按下 'q' 键则退出
        char key = waitKey(1) & 0xFF;
        if (key == 'q') {
            cout << "用户按下 'q' 键，退出程序" << endl;
            break;
        }
    }
    
    // 释放摄像头资源
    cap.release();
    // 关闭所有OpenCV窗口
    destroyAllWindows();
    cout << "摄像头已关闭，蓝色积木块检测程序结束" << endl;
    
    return 0;
}