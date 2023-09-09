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
const bool color=2;//����ѡ���ж�װ�װ����ɫ 0��ɫ   2��ɫ

int main()
{
    //�������⣺
    //1.װ�װ��ڱ�Ե��ֻ��ʶ��һ������
    //2.��ɫװ�װ���ʱ��ʶ�𲻳�������

    VideoCapture video;
    video.open("./images/zjbb.mp4");
    Mat frame;
    Mat channels[3], binary, Gaussian;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    Rect boundRect;//�洢��С��Ӿ���
    while (true)
    {
        Rect point_array[50];//�洢�ϸ����Ӿ���
        if (!video.read(frame)) {
            break;
        }
        split(frame, channels);
        if (color == true) {
            threshold(channels[0], binary, kThreashold, kMaxVal, 0);//��b���ж�ֵ��
        }
        else {
            threshold(channels[2], binary, kThreashold, kMaxVal, 0);//��r���ж�ֵ��
        }

        GaussianBlur(binary, Gaussian, kGaussianBlueSize, 0);
        Mat k = getStructuringElement(1, Size(3, 3));
        findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        int index = 0;
        for (int i = 0; i < contours.size(); i++) {
            //box = minAreaRect(Mat(contours[i]));//������ת����
            //box.points(boxPts.data());
            boundRect = boundingRect(Mat(contours[i]));//������С��Ӿ���
            rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//���� ����
            try
            {
                if (double(boundRect.height / boundRect.width) >= 2 && boundRect.height > 36 && boundRect.height > 20) {//���������ľ��󱣴���point_array
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

        if (index < 2) {//û��⵽�ϸ������ ֱ�� ��һ֡
            imshow("video", frame);
            cv::waitKey(10);
            continue;
        }

        int point_near[2];
        int min = 10000;
        for (int i = 0; i < index - 1; i++)//�ҵ����֮����С�ĵ��������������� 
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
            Rect rectangle_1 = point_array[point_near[0]];//�ҵ����������Ƶ�����
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
