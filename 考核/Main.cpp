#include <opencv2/opencv.hpp>
#include <iostream>
#include <Base.h>
#include <Armor.h>
#include <Game.h>
#include <App.h>

using namespace std;
using namespace cv;

int main() {
	//Base bs;
	////bs.Q3();

	//App app;
	////app.Q2();

	//Game gm;
	////gm.Q1p();
	//gm.Q2r();

	Armor Arm;
	cin >> Arm.re.Id >> Arm.re.Color;
	if (Arm.re.Color == 0) {
		cout << "ID: " << Arm.re.Id << "��ɫ: ��" << endl;
	}
	else {
		cout << "ID: " << Arm.re.Id << "��ɫ: ��ɫ" << endl;
	}
	

	return 0;
}



//���װ�װ� 
//detector.cpp

//#include"include/detector.h"
//
//
//// ͼ��Ԥ����
//Mat ArmorDetect::preprocess(Mat img)
//{
//
//    // ��������ڲξ���3*3�ͻ����������1*5
//    Mat cameraMatrix = (Mat_<double>(3, 3) << 1576.70020, 0.000000000000, 635.46084, 0.000000000000, 1575.77707, 529.83878, 0.000000000000, 0.000000000000, 1.000000000000);
//    Mat distCoeffs = (Mat_<double>(1, 5) << -0.08325, 0.21277, 0.00022, 0.00033, 0);
//    Mat img_clone;
//    // ��ͼ����н�������
//    // ��һ������Ϊԭͼ�񣬵ڶ�������Ϊ���ͼ�񣬵���������Ϊ����ڲξ��󣬵��ĸ�����Ϊ�������
//    undistort(img, img_clone, cameraMatrix, distCoeffs);
//
//    // ͨ������
//    vector<Mat> channels; // �����洢��������Ķ�ͨ����Mat
//    // ͨ�����뺯��split������ͨ���ľ������ɵ�ͨ������
//    split(img_clone, channels); // ��һ������Ϊ����ͼ�񣬵ڶ�������Ϊ�洢ͨ����Mat
//
//
//    Mat img_gray;              // �洢��������ĵ�ͨ��
//    img_gray = channels.at(2); // ��Ϊֻ��ʶ���ɫ������ֻ��ʹ�ú�ɫͨ����
//    int value = 230;           // ������ֵ
//    // ��ֵ������
//    Mat dst;
//    threshold(img_gray, dst, value, 255, THRESH_BINARY); // ����ͼ�����ͼ����ֵ�����ֵ����ֵ������
//
//    // ��ͼ�������̬ѧ����
//    Mat pre;
//    // �þ����ȡ�ṹԪ�أ���һ������������Բ����ˣ��ڶ�������Ϊ�ṹԪ�صĳߴ磬һ��ߴ�Խ��ʴԽ���ԣ���������������Ϊ���ĵ��λ��
//    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
//    morphologyEx(dst, pre, MORPH_OPEN, kernel); // ����ͼ�����ͼ�񣬿��������ṹԪ��
//    return pre;
//}
////Ѱ������
//vector<RotatedRect> ArmorDetect::Findcounters(Mat pre)
//{
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    // ��һ������ΪԤ�����ĵ�ͨ��ͼ�񣬵ڶ�������Ϊ������������һ��˫��������������������Ϊ�����Ľṹ��ϵ
//    // ���ĸ�����Ϊ���������ļ���ģʽ���������Χ��������������Χ�����ڵ���Χ����������.
//    // ���������Ϊ���������Ľ��Ʒ����������������Ĺյ���Ϣ�������������յ㴦�ĵ㱣����contours������
//    findContours(pre, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));
//    // ����ת����ʵ�ֵ���ɸѡ
//    vector<RotatedRect> light;
//    for (size_t i = 0; i < contours.size(); ++i)
//    {
//
//        RotatedRect light_rect = minAreaRect(contours[i]); // ��㼯��С�ľ���
//        double width = light_rect.size.width;              // ����εĿ��
//        double height = light_rect.size.height;            // ����εĸ߶�
//        double area = width * height;                      // ����ε����
//        if (area < 10)
//            continue; // ������̫С������
//        if (width / height > 0.4)
//            continue; // �������Ȳ����ϣ�����
//        else
//            light.push_back(light_rect); // ������Ҫ��ľ���ɸѡ����������ת������
//    }
//    return light;
//}
//// װ�װ�ʶ��
//vector<RotatedRect> ArmorDetect::Armordetect(vector<RotatedRect> light)
//{
//    vector<RotatedRect> armor_final; // �������ҳ�����ת����
//    RotatedRect armor;               // ��ת���Σ������ҳ��ľ���
//    double angle_differ;             // �������������ǶȵĲ�ֵ
//    if (light.size() < 2)
//        return armor_final; // �����⵽��������С����������ֱ�ӷ���
//    for (size_t i = 0; i < light.size() - 1; i++)
//    {
//        for (size_t j = i + 1; j < light.size(); j++)
//        {
//            angle_differ = abs(light[i].angle - light[j].angle);
//
//            bool if1 = (angle_differ < 5);                                                   // �жϽǶȲ�ֵ�Ƿ����10
//            //bool if2 = (abs(light[i].center.y - light[j].center.y < 20));                     // �ж����ĵ�߶Ȳ��Ƿ����20
//            //bool if3 = ((light[i].center.x + light[j].center.x) * light[i].size.height > 20); // �жϳ�����Ƿ����24
//            if (if1)
//            {
//                armor.center.x = (light[i].center.x + light[j].center.x) / 2; // װ�����ĵ�x����
//                armor.center.y = (light[i].center.y + light[j].center.y) / 2; // װ�����ĵ�y����
//                armor.angle = (light[i].angle + light[j].angle) / 2;
//                armor.size.width = abs(light[i].center.x - light[j].center.x); // װ�װ�Ŀ��
//                armor.size.height = light[i].size.height;                 // װ�װ�ĸ߶�
//                armor_final.push_back(armor);                             // ������Ҫ��ľ��ηŽ�������
//            }
//        }
//    }
//    return armor_final;
//}
//
//// ��ά�����ӻ�����
//vector<Point2f> ArmorDetect::point_2d(Mat img, RotatedRect ve)
//{
//    Point2f pt[4];
//    vector<Point2f> point2d;
//    int i;
//    //��ʼ��
//    for (i = 0; i < 4; i++)
//    {
//        pt[i].x = 0;
//        pt[i].y = 0;
//    }
//    ve.points(pt);
//    //��������
//    //��һ������ΪҪ�����߶ε�ͼ�񣬵ڶ�������Ϊ�߶ε���㣬����������Ϊ�߶ε��յ�
//    //���ĸ�����Ϊ�߶ε���ɫ�����������Ϊ�߶εĿ�ȣ�����������Ϊ�߶ε�����
//    line(img, pt[0], pt[1], Scalar(0, 0, 255), 2, 4, 0);
//    line(img, pt[1], pt[2], Scalar(0, 0, 255), 2, 4, 0);
//    line(img, pt[2], pt[3], Scalar(0, 0, 255), 2, 4, 0);
//    line(img, pt[3], pt[0], Scalar(0, 0, 255), 2, 4, 0);
//    //����ȡ���Ķ�ά�������vector  
//    point2d.push_back(pt[0]); //���½�
//    point2d.push_back(pt[1]); //���Ͻ�
//    point2d.push_back(pt[2]); //���Ͻ�
//    point2d.push_back(pt[3]); //���½�
//    return point2d;
//}
////��ȡ��ά�����
//vector<Point3f> ArmorDetect::point_3d()
//{
//    vector<Point3f> Points3d;
//    Point3f point3f;
//    // ԭ�㣨���½ǣ�
//    point3f.x = 0;
//    point3f.y = 0;
//    point3f.z = 0;
//    Points3d.push_back(point3f);
//    // ���Ͻ�
//    point3f.x = 0;
//    point3f.y = 5.5;
//    point3f.z = 0;
//    Points3d.push_back(point3f);
//    // ���Ͻ�
//    point3f.x = 14.0;
//    point3f.y = 5.5;
//    point3f.z = 0.0;
//    Points3d.push_back(point3f);
//    // ���½�
//    point3f.x = 14;
//    point3f.y = 0;
//    point3f.z = 0;
//    Points3d.push_back(point3f);
//    return Points3d;
//}







//https://github.com/fourthyuan/rm/blob/main/wind.txt 
//#include <iostream>
//#include "opencv2/opencv.hpp"
//#include <math.h>
//#include <ctime> 
//
//using namespace std;
//using namespace cv;
//
//#define multiple  1.5/*���ʣ�����Ŀ�������*/
//#define min -1/*��С��*/
//#define max 10000 /*������*/
//#define pointSetnumeber 5/*�㼯��С*/
//#define speedSetnumber 2/*�ٶȼ��ϴ�С*/
//#define blue 0/*��ɫ*/
//#define red 1/*��ɫ*/
//#define change 1/*ƫ��*/
//#define retain 0/*����*/
//
///*________________������_________________*/
//Point2f pointSet[pointSetnumeber];/*����㼯,���Ŀ���*/
//int pointNumber = 0;/*���pointSet[pointSetnumeber]����ŵ�*/
//int runNumber = 0;/*���Ŀ���Ĵ�����������꿪ʼԤ��*/
//float speed;
//float acceleratedSpeed;/*�ٶȣ����ٶ�*/
//float speedSet[speedSetnumber];/*�ٶȼ���*/
//int speedNumber = 0;/*���speedSet[speedSetnumber]������ٶ�*/
//float predictdistance;
//Point2f predictPoint;/*����Ԥ������Ԥ���*/
//float lastDistance = 0;/*��ʼ������*/
//int frame;/*֡��*/
//int color;/*����ʶ����ɫ��0������ɫ��1�����ɫ*/
//
//int minId;
//double minArea = max;/*������Ĵ����*/
//Point2f center;  /*�������Բ��������*/
//float radius;  /*�������Բ�뾶*/
//Point2f oldcenter;   /*��������ԲԲ������*/
//Point2f newcenter;  /*���������Բ��������*/
//float newradius;  /*���������Բ�뾶*/
//
//int maxId;
//double maxArea = min;/*���ɸѡ������������*/
//float referenceR;/*�뾶�ο�����*/
//Point2f rectMid;/*�뾶�ο���������������������*/
//Point2f target;/*Ŀ���*/
//int state;/*�Ƿ�ƫ��*/
///*-------------------------------------------------------------------------*/
//
///*________________��������___________________*/
//float distance(Point2f lastPoint, Point2f presentPoint);/*���������ľ���*/
///*---------------------------------------------------------------------*/
//
///********************************************************** **********************
//�ļ�����wind.cpp
//����:
//���ߣ�
//�汾��2023.1.15.01
//���ʱ�䣺2023.1.15
//* ********************************************************* ***********************/
//float distance(Point2f lastPoint, Point2f presentPoint) {
//	float distance;
//	distance = sqrt((presentPoint.x - lastPoint.x) * (presentPoint.x - lastPoint.x) + (presentPoint.y - lastPoint.y) * (presentPoint.y - lastPoint.y));
//	return distance;
//}
//
//
//int main()
//{
//	VideoCapture cap("./images/1_blue.mp4");
//	Mat image;
//	color = blue;/*��ɫ*/
//	for (;;) {
//		cap.read(image);
//		/*��ʼ����ͼ��*/
//		clock_t start = clock();
//		/*�ı��С�����֡��*/
//		resize(image, image, Size(image.cols * 0.35, image.rows * 0.35));
//		/*����Ч��չʾ*/
//		Mat test;
//		image.copyTo(test);
//		/*��������ŷ���ͨ�����ͼ��*/
//		vector<Mat> imgChannels;
//		split(image, imgChannels);
//		/*
//		��ɫ
//		Mat midimage = imgChannels.at(2) - imgChannels.at(0);
//		imshow("1", midimage);
//		*/
//		/*��ɫ*/
//		Mat blueImage = imgChannels.at(0) - imgChannels.at(2);
//		Mat binaryImage;
//		Mat binaryImagecricle;
//		/*��ֵ��*/
//		threshold(blueImage, binaryImagecricle, 170, 255, THRESH_BINARY);
//		threshold(blueImage, binaryImage, 90, 255, THRESH_BINARY);
//		/*��ʴ����*/
//		Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
//		Mat dstImage;
//		erode(binaryImagecricle, dstImage, element);
//		/*�ҵ�Բ���˶���Բ�ġ���R*/
//		state = retain;
//		vector<vector<Point>> outlines;
//		vector<Vec4i> hierarchies;
//		findContours(dstImage, outlines, hierarchies, RETR_TREE, CHAIN_APPROX_NONE);
//		for (int i = 0; i < outlines.size(); i++) {
//
//			vector<Point>points;
//			double area = contourArea(outlines[i]);
//			/*����ų�����*/
//			if (area < 10 || area>10000)
//				continue;
//			/*�ҵ�û�и�����������*/
//			if (hierarchies[i][3] >= 0 && hierarchies[i][3] < outlines.size())
//				continue;
//			/*������������*/
//			if (hierarchies[i][2] < 0 || hierarchies[i][2] >= outlines.size())
//				continue;
//			/*������Χ*/
//			if (area <= minArea + 10 && area >= minArea - 20) {
//				minArea = area;
//				minId = i;
//				continue;
//			}
//			/*�����С������*/
//			if (minArea >= area)
//			{
//				minArea = area;
//				minId = i;
//			}
//
//
//
//		}
//		/*��ֹminId���ڷ�Χ�ڱ���*/
//		if (minId >= 0 && minId < outlines.size())
//		{
//			/*�����Բ���ҵ�Բ��*/
//
//			minEnclosingCircle(Mat(outlines[minId]), newcenter, newradius);
//			/*��С���������*/
//			if (distance(newcenter, center) < 2) {
//			}
//			else {
//				oldcenter = center;
//				center = newcenter;
//				state = change;
//			}
//			if (fabs(newradius - radius) < 2) {
//			}
//			else {
//				radius = newradius;
//			}
//			circle(test, center, radius, Scalar(0, 0, 255), 1, 8, 0);
//		}
//		else {
//			continue;
//		}
//
//
//		/*���Ͳ���*/
//		element = getStructuringElement(0, Size(3, 3));
//		Mat dilateImage;
//		/*dilate���һ�����������ʹ���*/
//		dilate(binaryImage, dilateImage, element, Point(-1, -1), 2);
//		/*��������*/
//		vector<vector<Point>> contours;
//		vector<Vec4i> hierarchy;
//		findContours(dilateImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
//		for (int i = 0; i < contours.size(); i++) {
//			vector<Point>points;
//			double area = contourArea(contours[i]);
//			/*����ų�����*/
//			if (area < 20 || area>10000)
//				continue;
//			/*�ҵ�û�и�����������*/
//			if (hierarchy[i][3] >= 0 && hierarchy[i][3] < contours.size())
//				continue;
//			/*��û��������*/
//			if (hierarchy[i][2] >= 0 && hierarchy[i][2] < contours.size())
//				continue;
//			/*�������������*/
//			if (maxArea <= area)
//			{
//				maxArea = area;
//				maxId = i;
//			}
//			/*������Χ*/
//			if (area <= maxArea + 50 && area >= maxArea - 50) {
//				maxArea = area;
//				maxId = i;
//			}
//			cout << maxArea << endl;
//
//		}
//		if (maxId >= 0 && maxId < contours.size()) {
//			/*�����*/
//			Moments rect;
//			rect = moments(contours[maxId], false);
//			/*�������ľ�:*/
//			Point2f rectmid;
//			rectmid = Point2f(rect.m10 / rect.m00, rect.m01 / rect.m00);
//			/*�������λ����*/
//			drawContours(test, contours, maxId, Scalar(0, 255, 255), 1, 8);
//
//			/*��С����*/
//			if (runNumber < 2) {
//				referenceR = distance(rectmid, center);
//				rectMid = rectmid;
//			}
//			else if (distance(rectmid, center) <= referenceR + 2 && distance(rectmid, center) >= referenceR - 2 && distance(rectmid, rectMid) < 0.5) {
//			}
//			else {
//				referenceR = distance(rectmid, center);
//				rectMid = rectmid;
//			}
//
//			/*����������λ���ĵ�*/
//			circle(test, rectMid, 1, Scalar(0, 255, 255), -1, 8, 0);
//			/*2��1����������λ,���*/
//			/*��һ����*/
//			if (rectMid.x >= center.x && rectMid.y <= center.y) {
//				target = Point2f(center.x + (rectMid.x - center.x) * multiple, center.y - (center.y - rectMid.y) * multiple);
//
//			}
//			/*�ڶ�����*/
//			if (rectMid.x <= center.x && rectMid.y <= center.y) {
//				target = Point2f(center.x - (center.x - rectMid.x) * multiple, center.y - (center.y - rectMid.y) * multiple);
//
//			}
//			/*��������*/
//			if (rectMid.x <= center.x && rectMid.y >= center.y) {
//				target = Point2f(center.x - (center.x - rectMid.x) * multiple, center.y + (rectMid.y - center.y) * multiple);
//
//			}
//			/*��������*/
//			if (rectMid.x >= center.x && rectMid.y >= center.y) {
//				target = Point2f(center.x + (rectMid.x - center.x) * multiple, center.y + (rectMid.y - center.y) * multiple);
//
//			}
//			circle(test, target, 1, Scalar(0, 255, 255), -1, 8, 0);
//
//			/*���������ĵ����㼯*/
//			pointSet[pointNumber] = target;
//			pointNumber++;
//			/*ʵ���µ��滻�ɵ�*/
//			if (pointNumber == pointSetnumeber) {
//				pointNumber = 0;
//			}
//
//		}
//		else {
//			continue;
//		}
//		/*��ƫ��*/
//		if (state == change) {
//			float xchange;
//			float ychange;
//			xchange = center.x - oldcenter.x;
//			ychange = center.y - oldcenter.y;
//			/*�ı�㼯*/
//			if (pointNumber == 0) {
//				pointSet[0] = Point2f(pointSet[0].x + xchange, pointSet[0].y + ychange);
//				pointSet[1] = Point2f(pointSet[1].x + xchange, pointSet[1].y + ychange);
//				pointSet[2] = Point2f(pointSet[2].x + xchange, pointSet[2].y + ychange);
//				pointSet[3] = Point2f(pointSet[3].x + xchange, pointSet[3].y + ychange);
//			}
//			if (pointNumber == 1) {
//				pointSet[1] = Point2f(pointSet[1].x + xchange, pointSet[1].y + ychange);
//				pointSet[2] = Point2f(pointSet[2].x + xchange, pointSet[2].y + ychange);
//				pointSet[3] = Point2f(pointSet[3].x + xchange, pointSet[3].y + ychange);
//				pointSet[4] = Point2f(pointSet[4].x + xchange, pointSet[4].y + ychange);
//			}
//			if (pointNumber == 2) {
//				pointSet[2] = Point2f(pointSet[2].x + xchange, pointSet[2].y + ychange);
//				pointSet[3] = Point2f(pointSet[3].x + xchange, pointSet[3].y + ychange);
//				pointSet[4] = Point2f(pointSet[4].x + xchange, pointSet[4].y + ychange);
//				pointSet[0] = Point2f(pointSet[0].x + xchange, pointSet[0].y + ychange);
//			}
//			if (pointNumber == 3) {
//				pointSet[3] = Point2f(pointSet[3].x + xchange, pointSet[3].y + ychange);
//				pointSet[4] = Point2f(pointSet[4].x + xchange, pointSet[4].y + ychange);
//				pointSet[0] = Point2f(pointSet[0].x + xchange, pointSet[0].y + ychange);
//				pointSet[1] = Point2f(pointSet[1].x + xchange, pointSet[1].y + ychange);
//			}
//			if (pointNumber == 4) {
//				pointSet[4] = Point2f(pointSet[4].x + xchange, pointSet[4].y + ychange);
//				pointSet[0] = Point2f(pointSet[0].x + xchange, pointSet[0].y + ychange);
//				pointSet[1] = Point2f(pointSet[1].x + xchange, pointSet[1].y + ychange);
//				pointSet[2] = Point2f(pointSet[2].x + xchange, pointSet[2].y + ychange);
//			}
//		}
//
//
//		/*Ԥ��*/
//		if (runNumber > pointSetnumeber) {
//
//			int i = pointNumber - 1;//ȡ���µĵ����ٶ�
//			int number1 = i;
//			if (number1 < 0) {
//				number1 += pointSetnumeber;
//			}
//			int number2 = i - 1;
//			if (number2 < 0) {
//				number2 += pointSetnumeber;
//			}
//			int number3 = i - 3;
//			if (number3 < 0) {
//				number3 += pointSetnumeber;
//			}
//			int number4 = i - 4;
//			if (number4 < 0) {
//				number4 += pointSetnumeber;
//			}
//			/*ȡ����ĵ㣬���ٶȣ�����ٶ�*/
//			speed = distance(pointSet[number1], pointSet[number2]) * frame;
//			speedSet[0] = speed;
//			speed = distance(pointSet[number3], pointSet[number4]) * frame;
//			speedSet[1] = speed;
//			acceleratedSpeed = fabs((speedSet[0] - speedSet[1]) * frame);
//
//			/* X = V0T + 1 / 2AT'2��ͨ�����빫ʽ����Ԥ��Ĵ������� */
//			predictdistance = 4.5 * speedSet[0] / frame + 1 / 2 * acceleratedSpeed / frame / frame * 18;
//
//
//			/*���Ԥ��ʱx, y������ֵ�ı�ֵ*/
//
//			float xRatio, yRatio;
//			xRatio = fabs(pointSet[number1].x - pointSet[number2].x) / distance(pointSet[number1], pointSet[number2]);
//			yRatio = fabs(pointSet[number1].y - pointSet[number2].y) / distance(pointSet[number1], pointSet[number2]);
//			/*��һ������  ˳  ����*/
//			if (pointSet[number1].x >= pointSet[number2].x && pointSet[number1].y >= pointSet[number2].y) {
//				predictPoint = Point2f(pointSet[number1].x + predictdistance * xRatio, pointSet[number1].y + predictdistance * yRatio);
//			}
//			/*�ڶ�������  ˳  ����*/
//			if (pointSet[number1].x >= pointSet[number2].x && pointSet[number1].y <= pointSet[number2].y) {
//				predictPoint = Point2f(pointSet[number1].x + predictdistance * xRatio, pointSet[number1].y - predictdistance * yRatio);
//			}
//			/*����������  ˳  һ��*/
//			if (pointSet[number1].x <= pointSet[number2].x && pointSet[number1].y <= pointSet[number2].y) {
//				predictPoint = Point2f(pointSet[number1].x - predictdistance * xRatio, pointSet[number1].y - predictdistance * yRatio);
//			}
//			/*����������  ˳  ����*/
//			if (pointSet[number1].x <= pointSet[number2].x && pointSet[number1].y >= pointSet[number2].y) {
//				predictPoint = Point2f(pointSet[number1].x - predictdistance * xRatio, pointSet[number1].y + predictdistance * yRatio);
//			}
//
//
//
//			/*����Ԥ�ⶶ��*/
//			lastDistance = predictdistance;
//
//			/*��켣���·��*/
//			/*��Բ*/
//			circle(test, center, distance(center, target) + 3, Scalar(0, 255, 255), 1, 8, 0);
//			/*Ԥ�����Բ����£*/
//		/*��һ����*/
//			if (predictPoint.x >= center.x && predictPoint.y <= center.y) {
//				predictPoint = Point2f(center.x + (predictPoint.x - center.x) * distance(center, target) / distance(center, predictPoint), center.y - (center.y - predictPoint.y) * distance(center, target) / distance(center, predictPoint));
//
//			}
//			/*�ڶ�����*/
//			if (predictPoint.x <= center.x && predictPoint.y <= center.y) {
//				predictPoint = Point2f(center.x - (center.x - predictPoint.x) * distance(center, target) / distance(center, predictPoint), center.y - (center.y - predictPoint.y) * distance(center, target) / distance(center, predictPoint));
//
//			}
//			/*��������*/
//			if (predictPoint.x <= center.x && predictPoint.y >= center.y) {
//				predictPoint = Point2f(center.x - (center.x - predictPoint.x) * distance(center, target) / distance(center, predictPoint), center.y + (predictPoint.y - center.y) * distance(center, target) / distance(center, predictPoint));
//
//			}
//			/*��������*/
//			if (predictPoint.x >= center.x && predictPoint.y >= center.y) {
//				predictPoint = Point2f(center.x + (predictPoint.x - center.x) * distance(center, target) / distance(center, predictPoint), center.y + (predictPoint.y - center.y) * distance(center, target) / distance(center, predictPoint));
//
//			}
//			/*����Ԥ���*/
//			circle(test, predictPoint, 2, Scalar(0, 0, 255), -1, 8, 0);
//
//		}
//
//		imshow("1", test);
//		imshow("2", dilateImage);
//		runNumber++;
//		clock_t end = clock();
//		frame = 1 / (double)(end - start) * CLOCKS_PER_SEC;
//		cout << frame << "֡" << endl;
//		waitKey(1);
//	}
//}