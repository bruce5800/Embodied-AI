#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // 构建一个空白的矩阵 (30行, 40列, 3通道)
    Mat img = Mat::ones(30, 40, CV_8UC3);
    
    // 将第15行所有像素点全都改成绿色
    for (int i = 0; i < 40; i++) {
        // 设置第15行颜色为绿色 (注意：OpenCV中BGR格式，所以(0,255,0)是绿色)
        img.at<Vec3b>(15, i) = Vec3b(0, 255, 0);
    }
    
    // 显示图片
    imshow("src", img);
    
    // 等待按键
    waitKey(0);
    
    // 关闭所有窗口
    destroyAllWindows();
    
    return 0;
}