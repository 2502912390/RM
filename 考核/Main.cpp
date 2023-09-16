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
		cout << "ID: " << Arm.re.Id << "颜色: 红" << endl;
	}
	else {
		cout << "ID: " << Arm.re.Id << "颜色: 蓝色" << endl;
	}
	

	return 0;
}



//检测装甲板 
//detector.cpp

//#include"include/detector.h"
//
//
//// 图像预处理
//Mat ArmorDetect::preprocess(Mat img)
//{
//
//    // 定义相机内参矩阵3*3和畸变参数矩阵1*5
//    Mat cameraMatrix = (Mat_<double>(3, 3) << 1576.70020, 0.000000000000, 635.46084, 0.000000000000, 1575.77707, 529.83878, 0.000000000000, 0.000000000000, 1.000000000000);
//    Mat distCoeffs = (Mat_<double>(1, 5) << -0.08325, 0.21277, 0.00022, 0.00033, 0);
//    Mat img_clone;
//    // 对图像进行矫正畸变
//    // 第一个参数为原图像，第二个参数为输出图像，第三个参数为相机内参矩阵，第四个参数为畸变参数
//    undistort(img, img_clone, cameraMatrix, distCoeffs);
//
//    // 通道分离
//    vector<Mat> channels; // 创建存储分离出来的多通道的Mat
//    // 通道分离函数split，将多通道的矩阵分离成单通道矩阵
//    split(img_clone, channels); // 第一个参数为传入图像，第二个参数为存储通道的Mat
//
//
//    Mat img_gray;              // 存储分离出来的单通道
//    img_gray = channels.at(2); // 因为只用识别红色，所以只用使用红色通道的
//    int value = 230;           // 设置阈值
//    // 二值化处理
//    Mat dst;
//    threshold(img_gray, dst, value, 255, THRESH_BINARY); // 处理图像，输出图像，阈值，最大值，二值化类型
//
//    // 对图像进行形态学操作
//    Mat pre;
//    // 用矩阵获取结构元素，第一个参数返回椭圆卷积核，第二个参数为结构元素的尺寸，一般尺寸越大腐蚀越明显，第三个参数设置为中心点的位置
//    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
//    morphologyEx(dst, pre, MORPH_OPEN, kernel); // 输入图像，输出图像，开操作，结构元素
//    return pre;
//}
////寻找轮廓
//vector<RotatedRect> ArmorDetect::Findcounters(Mat pre)
//{
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    // 第一个参数为预处理后的单通道图像，第二个参数为轮廓数量（是一个双重向量），第三个参数为轮廓的结构关系
//    // 第四个参数为定义轮廓的检索模式，检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略.
//    // 第五个参数为定义轮廓的近似方法，仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours向量内
//    findContours(pre, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));
//    // 用旋转矩阵实现灯条筛选
//    vector<RotatedRect> light;
//    for (size_t i = 0; i < contours.size(); ++i)
//    {
//
//        RotatedRect light_rect = minAreaRect(contours[i]); // 求点集最小的矩形
//        double width = light_rect.size.width;              // 求矩形的宽度
//        double height = light_rect.size.height;            // 求矩形的高度
//        double area = width * height;                      // 求矩形的面积
//        if (area < 10)
//            continue; // 如果面积太小，忽略
//        if (width / height > 0.4)
//            continue; // 如果长宽比不符合，忽略
//        else
//            light.push_back(light_rect); // 将符合要求的矩形筛选出来放入旋转矩形类
//    }
//    return light;
//}
//// 装甲板识别
//vector<RotatedRect> ArmorDetect::Armordetect(vector<RotatedRect> light)
//{
//    vector<RotatedRect> armor_final; // 用来放找出的旋转矩形
//    RotatedRect armor;               // 旋转矩形，定义找出的矩形
//    double angle_differ;             // 定义两个灯条角度的差值
//    if (light.size() < 2)
//        return armor_final; // 如果检测到灯条数量小于两个，则直接返回
//    for (size_t i = 0; i < light.size() - 1; i++)
//    {
//        for (size_t j = i + 1; j < light.size(); j++)
//        {
//            angle_differ = abs(light[i].angle - light[j].angle);
//
//            bool if1 = (angle_differ < 5);                                                   // 判断角度差值是否大于10
//            //bool if2 = (abs(light[i].center.y - light[j].center.y < 20));                     // 判断中心点高度差是否大于20
//            //bool if3 = ((light[i].center.x + light[j].center.x) * light[i].size.height > 20); // 判断长宽比是否大于24
//            if (if1)
//            {
//                armor.center.x = (light[i].center.x + light[j].center.x) / 2; // 装甲中心的x坐标
//                armor.center.y = (light[i].center.y + light[j].center.y) / 2; // 装甲中心的y坐标
//                armor.angle = (light[i].angle + light[j].angle) / 2;
//                armor.size.width = abs(light[i].center.x - light[j].center.x); // 装甲板的宽度
//                armor.size.height = light[i].size.height;                 // 装甲板的高度
//                armor_final.push_back(armor);                             // 将符合要求的矩形放进容器里
//            }
//        }
//    }
//    return armor_final;
//}
//
//// 二维坐标点加画矩形
//vector<Point2f> ArmorDetect::point_2d(Mat img, RotatedRect ve)
//{
//    Point2f pt[4];
//    vector<Point2f> point2d;
//    int i;
//    //初始化
//    for (i = 0; i < 4; i++)
//    {
//        pt[i].x = 0;
//        pt[i].y = 0;
//    }
//    ve.points(pt);
//    //画出矩形
//    //第一个参数为要绘制线段的图像，第二个参数为线段的起点，第三个参数为线段的终点
//    //第四个参数为线段的颜色，第五个参数为线段的宽度，第六个参数为线段的类型
//    line(img, pt[0], pt[1], Scalar(0, 0, 255), 2, 4, 0);
//    line(img, pt[1], pt[2], Scalar(0, 0, 255), 2, 4, 0);
//    line(img, pt[2], pt[3], Scalar(0, 0, 255), 2, 4, 0);
//    line(img, pt[3], pt[0], Scalar(0, 0, 255), 2, 4, 0);
//    //将获取到的二维坐标存入vector  
//    point2d.push_back(pt[0]); //左下角
//    point2d.push_back(pt[1]); //左上角
//    point2d.push_back(pt[2]); //右上角
//    point2d.push_back(pt[3]); //右下角
//    return point2d;
//}
////获取三维坐标点
//vector<Point3f> ArmorDetect::point_3d()
//{
//    vector<Point3f> Points3d;
//    Point3f point3f;
//    // 原点（左下角）
//    point3f.x = 0;
//    point3f.y = 0;
//    point3f.z = 0;
//    Points3d.push_back(point3f);
//    // 左上角
//    point3f.x = 0;
//    point3f.y = 5.5;
//    point3f.z = 0;
//    Points3d.push_back(point3f);
//    // 右上角
//    point3f.x = 14.0;
//    point3f.y = 5.5;
//    point3f.z = 0.0;
//    Points3d.push_back(point3f);
//    // 右下角
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
//#define multiple  1.5/*倍率，换算目标点所用*/
//#define min -1/*极小量*/
//#define max 10000 /*极大量*/
//#define pointSetnumeber 5/*点集大小*/
//#define speedSetnumber 2/*速度集合大小*/
//#define blue 0/*蓝色*/
//#define red 1/*红色*/
//#define change 1/*偏移*/
//#define retain 0/*保持*/
//
///*________________变量区_________________*/
//Point2f pointSet[pointSetnumeber];/*定义点集,存放目标点*/
//int pointNumber = 0;/*配合pointSet[pointSetnumeber]，存放点*/
//int runNumber = 0;/*存放目标点的次数，次数达标开始预测*/
//float speed;
//float acceleratedSpeed;/*速度，加速度*/
//float speedSet[speedSetnumber];/*速度集合*/
//int speedNumber = 0;/*配合speedSet[speedSetnumber]，存放速度*/
//float predictdistance;
//Point2f predictPoint;/*定义预测距离和预测点*/
//float lastDistance = 0;/*初始化距离*/
//int frame;/*帧数*/
//int color;/*控制识别颜色，0代表蓝色，1代表红色*/
//
//int minId;
//double minArea = max;/*存放中心处面积*/
//Point2f center;  /*定义外接圆中心坐标*/
//float radius;  /*定义外接圆半径*/
//Point2f oldcenter;   /*定义旧外接圆圆心坐标*/
//Point2f newcenter;  /*定义新外接圆中心坐标*/
//float newradius;  /*定义新外接圆半径*/
//
//int maxId;
//double maxArea = min;/*存放筛选后最大面积轮廓*/
//float referenceR;/*半径参考长度*/
//Point2f rectMid;/*半径参考长度所在轮廓几何中心*/
//Point2f target;/*目标点*/
//int state;/*是否偏移*/
///*-------------------------------------------------------------------------*/
//
///*________________函数声明___________________*/
//float distance(Point2f lastPoint, Point2f presentPoint);/*计算两点间的距离*/
///*---------------------------------------------------------------------*/
//
///********************************************************** **********************
//文件名：wind.cpp
//介绍:
//作者：
//版本：2023.1.15.01
//完成时间：2023.1.15
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
//	color = blue;/*蓝色*/
//	for (;;) {
//		cap.read(image);
//		/*开始处理图像*/
//		clock_t start = clock();
//		/*改变大小，提高帧率*/
//		resize(image, image, Size(image.cols * 0.35, image.rows * 0.35));
//		/*测试效果展示*/
//		Mat test;
//		image.copyTo(test);
//		/*容器，存放分离通道后的图像*/
//		vector<Mat> imgChannels;
//		split(image, imgChannels);
//		/*
//		红色
//		Mat midimage = imgChannels.at(2) - imgChannels.at(0);
//		imshow("1", midimage);
//		*/
//		/*蓝色*/
//		Mat blueImage = imgChannels.at(0) - imgChannels.at(2);
//		Mat binaryImage;
//		Mat binaryImagecricle;
//		/*二值化*/
//		threshold(blueImage, binaryImagecricle, 170, 255, THRESH_BINARY);
//		threshold(blueImage, binaryImage, 90, 255, THRESH_BINARY);
//		/*腐蚀操作*/
//		Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
//		Mat dstImage;
//		erode(binaryImagecricle, dstImage, element);
//		/*找到圆周运动的圆心——R*/
//		state = retain;
//		vector<vector<Point>> outlines;
//		vector<Vec4i> hierarchies;
//		findContours(dstImage, outlines, hierarchies, RETR_TREE, CHAIN_APPROX_NONE);
//		for (int i = 0; i < outlines.size(); i++) {
//
//			vector<Point>points;
//			double area = contourArea(outlines[i]);
//			/*面积排除噪声*/
//			if (area < 10 || area>10000)
//				continue;
//			/*找到没有父轮廓的轮廓*/
//			if (hierarchies[i][3] >= 0 && hierarchies[i][3] < outlines.size())
//				continue;
//			/*找有子轮廓的*/
//			if (hierarchies[i][2] < 0 || hierarchies[i][2] >= outlines.size())
//				continue;
//			/*控制误差范围*/
//			if (area <= minArea + 10 && area >= minArea - 20) {
//				minArea = area;
//				minId = i;
//				continue;
//			}
//			/*面积最小的轮廓*/
//			if (minArea >= area)
//			{
//				minArea = area;
//				minId = i;
//			}
//
//
//
//		}
//		/*防止minId不在范围内报错*/
//		if (minId >= 0 && minId < outlines.size())
//		{
//			/*画外接圆并找到圆心*/
//
//			minEnclosingCircle(Mat(outlines[minId]), newcenter, newradius);
//			/*减小抖动，误差*/
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
//		/*膨胀操作*/
//		element = getStructuringElement(0, Size(3, 3));
//		Mat dilateImage;
//		/*dilate最后一个数字是膨胀次数*/
//		dilate(binaryImage, dilateImage, element, Point(-1, -1), 2);
//		/*轮廓发现*/
//		vector<vector<Point>> contours;
//		vector<Vec4i> hierarchy;
//		findContours(dilateImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
//		for (int i = 0; i < contours.size(); i++) {
//			vector<Point>points;
//			double area = contourArea(contours[i]);
//			/*面积排除噪声*/
//			if (area < 20 || area>10000)
//				continue;
//			/*找到没有父轮廓的轮廓*/
//			if (hierarchy[i][3] >= 0 && hierarchy[i][3] < contours.size())
//				continue;
//			/*找没子轮廓的*/
//			if (hierarchy[i][2] >= 0 && hierarchy[i][2] < contours.size())
//				continue;
//			/*找面积最大的轮廓*/
//			if (maxArea <= area)
//			{
//				maxArea = area;
//				maxId = i;
//			}
//			/*控制误差范围*/
//			if (area <= maxArea + 50 && area >= maxArea - 50) {
//				maxArea = area;
//				maxId = i;
//			}
//			cout << maxArea << endl;
//
//		}
//		if (maxId >= 0 && maxId < contours.size()) {
//			/*计算矩*/
//			Moments rect;
//			rect = moments(contours[maxId], false);
//			/*计算中心矩:*/
//			Point2f rectmid;
//			rectmid = Point2f(rect.m10 / rect.m00, rect.m01 / rect.m00);
//			/*画出需打部位轮廓*/
//			drawContours(test, contours, maxId, Scalar(0, 255, 255), 1, 8);
//
//			/*减小抖动*/
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
//			/*画出样本部位中心点*/
//			circle(test, rectMid, 1, Scalar(0, 255, 255), -1, 8, 0);
//			/*2：1计算需打击部位,存放*/
//			/*第一象限*/
//			if (rectMid.x >= center.x && rectMid.y <= center.y) {
//				target = Point2f(center.x + (rectMid.x - center.x) * multiple, center.y - (center.y - rectMid.y) * multiple);
//
//			}
//			/*第二象限*/
//			if (rectMid.x <= center.x && rectMid.y <= center.y) {
//				target = Point2f(center.x - (center.x - rectMid.x) * multiple, center.y - (center.y - rectMid.y) * multiple);
//
//			}
//			/*第三象限*/
//			if (rectMid.x <= center.x && rectMid.y >= center.y) {
//				target = Point2f(center.x - (center.x - rectMid.x) * multiple, center.y + (rectMid.y - center.y) * multiple);
//
//			}
//			/*第四象限*/
//			if (rectMid.x >= center.x && rectMid.y >= center.y) {
//				target = Point2f(center.x + (rectMid.x - center.x) * multiple, center.y + (rectMid.y - center.y) * multiple);
//
//			}
//			circle(test, target, 1, Scalar(0, 255, 255), -1, 8, 0);
//
//			/*将几何中心点存入点集*/
//			pointSet[pointNumber] = target;
//			pointNumber++;
//			/*实现新点替换旧点*/
//			if (pointNumber == pointSetnumeber) {
//				pointNumber = 0;
//			}
//
//		}
//		else {
//			continue;
//		}
//		/*算偏移*/
//		if (state == change) {
//			float xchange;
//			float ychange;
//			xchange = center.x - oldcenter.x;
//			ychange = center.y - oldcenter.y;
//			/*改变点集*/
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
//		/*预测*/
//		if (runNumber > pointSetnumeber) {
//
//			int i = pointNumber - 1;//取最新的点算速度
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
//			/*取最近四点，算速度，求加速度*/
//			speed = distance(pointSet[number1], pointSet[number2]) * frame;
//			speedSet[0] = speed;
//			speed = distance(pointSet[number3], pointSet[number4]) * frame;
//			speedSet[1] = speed;
//			acceleratedSpeed = fabs((speedSet[0] - speedSet[1]) * frame);
//
//			/* X = V0T + 1 / 2AT'2，通过距离公式，算预测的打击点距离 */
//			predictdistance = 4.5 * speedSet[0] / frame + 1 / 2 * acceleratedSpeed / frame / frame * 18;
//
//
//			/*算出预测时x, y需增加值的比值*/
//
//			float xRatio, yRatio;
//			xRatio = fabs(pointSet[number1].x - pointSet[number2].x) / distance(pointSet[number1], pointSet[number2]);
//			yRatio = fabs(pointSet[number1].y - pointSet[number2].y) / distance(pointSet[number1], pointSet[number2]);
//			/*第一象限内  顺  三逆*/
//			if (pointSet[number1].x >= pointSet[number2].x && pointSet[number1].y >= pointSet[number2].y) {
//				predictPoint = Point2f(pointSet[number1].x + predictdistance * xRatio, pointSet[number1].y + predictdistance * yRatio);
//			}
//			/*第二象限内  顺  四逆*/
//			if (pointSet[number1].x >= pointSet[number2].x && pointSet[number1].y <= pointSet[number2].y) {
//				predictPoint = Point2f(pointSet[number1].x + predictdistance * xRatio, pointSet[number1].y - predictdistance * yRatio);
//			}
//			/*第三象限内  顺  一逆*/
//			if (pointSet[number1].x <= pointSet[number2].x && pointSet[number1].y <= pointSet[number2].y) {
//				predictPoint = Point2f(pointSet[number1].x - predictdistance * xRatio, pointSet[number1].y - predictdistance * yRatio);
//			}
//			/*第四象限内  顺  二逆*/
//			if (pointSet[number1].x <= pointSet[number2].x && pointSet[number1].y >= pointSet[number2].y) {
//				predictPoint = Point2f(pointSet[number1].x - predictdistance * xRatio, pointSet[number1].y + predictdistance * yRatio);
//			}
//
//
//
//			/*减少预测抖动*/
//			lastDistance = predictdistance;
//
//			/*向轨迹拟合路径*/
//			/*画圆*/
//			circle(test, center, distance(center, target) + 3, Scalar(0, 255, 255), 1, 8, 0);
//			/*预测点像圆弧靠拢*/
//		/*第一象限*/
//			if (predictPoint.x >= center.x && predictPoint.y <= center.y) {
//				predictPoint = Point2f(center.x + (predictPoint.x - center.x) * distance(center, target) / distance(center, predictPoint), center.y - (center.y - predictPoint.y) * distance(center, target) / distance(center, predictPoint));
//
//			}
//			/*第二象限*/
//			if (predictPoint.x <= center.x && predictPoint.y <= center.y) {
//				predictPoint = Point2f(center.x - (center.x - predictPoint.x) * distance(center, target) / distance(center, predictPoint), center.y - (center.y - predictPoint.y) * distance(center, target) / distance(center, predictPoint));
//
//			}
//			/*第三象限*/
//			if (predictPoint.x <= center.x && predictPoint.y >= center.y) {
//				predictPoint = Point2f(center.x - (center.x - predictPoint.x) * distance(center, target) / distance(center, predictPoint), center.y + (predictPoint.y - center.y) * distance(center, target) / distance(center, predictPoint));
//
//			}
//			/*第四象限*/
//			if (predictPoint.x >= center.x && predictPoint.y >= center.y) {
//				predictPoint = Point2f(center.x + (predictPoint.x - center.x) * distance(center, target) / distance(center, predictPoint), center.y + (predictPoint.y - center.y) * distance(center, target) / distance(center, predictPoint));
//
//			}
//			/*画出预测点*/
//			circle(test, predictPoint, 2, Scalar(0, 0, 255), -1, 8, 0);
//
//		}
//
//		imshow("1", test);
//		imshow("2", dilateImage);
//		runNumber++;
//		clock_t end = clock();
//		frame = 1 / (double)(end - start) * CLOCKS_PER_SEC;
//		cout << frame << "帧" << endl;
//		waitKey(1);
//	}
//}