//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <All.h>
//
//using namespace std;
//using namespace cv;
//
//int main() {
//	Base bs;
//	//bs.Q3();
//
//	App app;
//	//app.Q2();
//
//	Game gm;
//	gm.Q1();
//
//	return 0;
//}

#include "stdio.h"
#include<iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

const int kThreashold = 180;
const int kMaxVal = 255;
const Size kGaussianBlueSize = Size(5, 5);
const bool color=2;//用于选择判断装甲板的颜色 0红色   2蓝色

int main()
{
    //存在问题：
    //1.装甲板在边缘，只能识别到一个光条
    //2.蓝色装甲板有时候识别不出轮廓？

    VideoCapture video;
    video.open("./images/zjbb.mp4");
    Mat frame;
    Mat channels[3], binary, Gaussian;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    Rect boundRect;//存储最小外接矩形
    while (true)
    {
        Rect point_array[50];//存储合格的外接矩阵
        if (!video.read(frame)) {
            break;
        }
        split(frame, channels);
        if (color == true) {
            threshold(channels[0], binary, kThreashold, kMaxVal, 0);//对b进行二值化
        }
        else {
            threshold(channels[2], binary, kThreashold, kMaxVal, 0);//对r进行二值化
        }

        GaussianBlur(binary, Gaussian, kGaussianBlueSize, 0);
        Mat k = getStructuringElement(1, Size(3, 3));
        findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        int index = 0;
        for (int i = 0; i < contours.size(); i++) {
            //box = minAreaRect(Mat(contours[i]));//计算旋转矩阵
            //box.points(boxPts.data());
            boundRect = boundingRect(Mat(contours[i]));//计算最小外接矩形
            rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//左上 右下
            try
            {
                if (double(boundRect.height / boundRect.width) >= 2 && boundRect.height > 36 && boundRect.height > 20) {//满足条件的矩阵保存在point_array
                    point_array[index] = boundRect;
                    index++;
                }
            }
            catch (const char* msg)
            {
                cout << printf(msg) << endl;
                continue;
            }
        }

        if (index < 2) {//没检测到合格的轮廓 直接 下一帧
            imshow("video", frame);
            cv::waitKey(10);
            continue;
        }

        int point_near[2];
        int min = 10000;
        for (int i = 0; i < index - 1; i++)//找到面积之差最小的的两个轮廓的索引 
        {
            for (int j = i + 1; j < index; j++) {
                int value = abs(point_array[i].area() - point_array[j].area());
                if (value < min)
                {
                    min = value;
                    point_near[0] = i;
                    point_near[1] = j;
                }
            }
        }

        try
        {
            Rect rectangle_1 = point_array[point_near[0]];//找到这两个相似的轮廓
            Rect rectangle_2 = point_array[point_near[1]];

            if (rectangle_2.x == 0 || rectangle_1.x == 0) {
                throw "not enough points";
            }

            Point point1 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y);
            Point point2 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y + rectangle_1.height);
            Point point3 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y);
            Point point4 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y + rectangle_2.height);
            Point p[4] = { point1,point2,point4,point3 };

            cout << p[0] << p[1] << p[2] << p[3] << endl;
            for (int i = 0; i < 4; i++) {
                line(frame, p[i % 4], p[(i + 1) % 4], Scalar(255, 255, 255), 2);
            }
        }
        catch (const char* msg)
        {
            cout << msg << endl;
        }

        imshow("video", frame);
        cv::waitKey(20);
    }
    cv::destroyAllWindows();
    return 0;
}
