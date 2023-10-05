
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Game.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<algorithm>
#include<set>
#include<map>

using namespace std;
using namespace cv;

//const int kThreashold = 180;//阈值
//const int kMaxVal = 255;
//const Size kGaussianBlueSize = Size(5, 5);

#define PI 3.1415926

void Game::Q1p() {//
	
	Mat frame = imread("./images/zj.png");
	Mat channels[3], binary, Gaussian;

	vector<vector<Point>> contours;//轮廓点
	vector<Vec4i> hierarchy;
	RotatedRect AreaRect;//存储光条最小(旋转)外接矩形

	split(frame, channels);
	//Mat colorimage = channels[0] - channels[2];

	threshold(channels[0], binary, 180, 255, 0);//对b进行二值化
	GaussianBlur(binary, Gaussian, Size(3, 3), 0);
	findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	RotatedRect light[20];//存筛选后的旋转矩阵
	Point2f points[4];//存RotatedRect转换的四点坐标

	int index = 0;//存放筛选后轮廓的下标
	
	for (int i = 0; i < contours.size(); i++) {
		//boundRect = boundingRect(Mat(contours[i]));//计算最小外接矩形
		AreaRect = minAreaRect(Mat(contours[i]));//计算最小的旋转矩阵 因为要使用angle
		double area = AreaRect.size.area();//面积

		//判断有问题  解决
		if (area > 100) { //先把一些太小的干扰轮廓筛选删除
			light[index] = AreaRect;
			//AreaRect.points(points);
			//rectangle(frame, point_array[index].tl(), point_array[index].br(), Scalar(255, 255, 255), 2, 8, 0);//左上 右下
			//for (int i = 0; i < 4; i++) {
			//	line(frame, points[i % 4], points[(i + 1) % 4], Scalar(255, 255, 255), 2);
			//}
			index++;
		}
	}

	//imshow("1", frame);
	//waitKey(0);

	if (index < 1) {//小于一个
		return;
	}
	//vector<RotatedRect> armor_final; //存放最终的矩阵
	RotatedRect armor;// 定义由两个光条构成的矩阵
	double angle_dif;//两个角度的插值
	double hight_dif;//高度差
	double area_diff;//面积差

	for (int i = 0; i < index; i++) {
		for (int j = i + 1; j < index; j++) {
			angle_dif = abs(light[i].angle - light[j].angle);//角度差
			hight_dif = abs(light[i].center.y - light[j].center.y);//高度差
			area_diff = abs(light[i].size.area() - light[i].size.area());//面积差

			if (angle_dif < 4&& hight_dif<20&& area_diff<50) {
				//cout << light[i].angle << " " << light[j].angle<<endl;

				light[i].points(points);
				/************TEST************/
				//for (int i = 0; i < 4; i++) {//画光条框1
				//	line(frame, points[i % 4], points[(i + 1) % 4], Scalar(0, 0, 255), 2);
				//}
				//imshow("frame", frame);
				////cout<<"1：宽" << light[i].size.width << " 高" << light[i].size.height << endl;
				//waitKey(0);
				//
				//light[j].points(points);
				//for (int i = 0; i < 4; i++) {//画光条框2
				//	line(frame, points[i % 4], points[(i + 1) % 4], Scalar(0, 0, 255), 2);
				//}
				//imshow("frame", frame);
				////cout << "2：宽" << light[j].size.width << " 高" << light[j].size.height << endl;
				//waitKey(0);
				/************TEST**********/
				
				armor.center.x = (light[i].center.x + light[j].center.x) / 2; // 装甲中心的x坐标
				armor.center.y = (light[i].center.y + light[j].center.y) / 2; // 装甲中心的y坐标
				
				//bug。。。 解决 原因是预想的和计算定义的长宽是不一样的
				double height, width;//将长的定为高，短的定为宽
				if (light[j].size.width > light[j].size.height) {//注意哪个是高 哪个是宽 
					armor.size.width = light[j].size.width;
					armor.size.height = abs(light[i].center.x - light[j].center.x); // 装甲板的高度
				}
				else {
					armor.size.width = abs(light[i].center.x - light[j].center.x);
					armor.size.height = light[j].size.height;
				}

				armor.angle = (light[i].angle + light[j].angle) / 2;
				//cout <<"面积" << armor.size.area() << endl;
				//cout << "高宽比例" << armor.size.height / armor.size.width << endl;

				//cout << light[i].center.x << " " << light[j].center.x << endl;
				//cout <<"装甲板宽度" << abs(light[i].center.x - light[j].center.x) << endl;

				//Point point1 = Point(light[i].center.x, light[i].center.y - light[i].size.height / 2);
				//Point point2 = Point(light[i].center.x, light[i].center.y + light[i].size.height / 2);
				//Point point3 = Point(light[j].center.x, light[j].center.y - light[j].size.height / 2);
				//Point point4 = Point(light[j].center.x, light[j].center.y + light[j].size.height / 2);
				//Point points[4] = { point1,point2,point4,point3 };

				if (armor.size.area() < 5000) {//用面积筛选吧。。。 待改进。。。
					armor.points(points);
					for (int i = 0; i < 4; i++) {//画两个光条组成的矩形框
						line(frame, points[i % 4], points[(i + 1) % 4], Scalar(0, 0, 255), 2);
					}
					circle(frame, armor.center, 2, Scalar(255, 255, 255), -1);//没问题
					cout <<"击打中心坐标" << armor.center << endl;
					imshow("frame", frame);
					//waitKey(0);
					//armor_final.push_back(armor); // 将符合要求的矩形放进容器里
				}
			}
		}
	}
	
	waitKey(0);
}

//待添加多目标的选定（目前只能检测两个光条），追踪失败判断（连续好几帧没检测到装甲板）
const bool color = true;//用于选择判断装甲板的颜色 flase红色   true蓝色
void Game::Q1v() {
	//思路：分别对红蓝装甲板的r和b通道进行分离并且二值化，然后寻找轮廓
	//对轮廓进行筛选 绘制出两个灯条矩形所围城的面积 以及 中心点

	//存在问题：
   //1.只检测到一个光条的时候，追踪会断开  解决：用前几帧作为参考
   //2.蓝色装甲板有时候识别不出轮廓？ 原因：检测出的部分轮廓高宽比为1.x 在筛选的时候没有使用double类型，将比例1.x的轮廓直接归类为1，导致需要的轮廓没有筛选出来  解决：计算比例时*1.0

	VideoCapture video;
	String s;
	int t;
	if (color == true) {
		s = "./images/zjbb.mp4";
		t = 0;
	}
	else {
		s = "./images/zjbr.mp4";
		t = 2;
	}

	video.open(s);
	Mat frame;
	video >> frame;
	int fps = video.get(CAP_PROP_FPS);
	int width = video.get(CAP_PROP_FRAME_WIDTH);
	int height = video.get(CAP_PROP_FRAME_HEIGHT);
	VideoWriter writer("./images/blue.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);//保存图像 用于查看哪里识别效果不好

	Mat channels[3], binary, Gaussian;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Point prePoint;//上一次打击点
	Point hitPoint;//当前打击点

	namedWindow("video", WINDOW_NORMAL);

	Rect boundRect;//存储最小外接矩形
	while (true)
	{
		double startTime = cv::getTickCount();

		Rect point_array[50];//存储合格的外接矩阵  理论上就两个
		if (!video.read(frame)) {
			break;
		}
		split(frame, channels);
		threshold(channels[t], binary, 180, 255, 0);//进行二值化

		GaussianBlur(binary, Gaussian, Size(5, 5), 0);
		//Mat k = getStructuringElement(1, Size(3, 3));
		findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

		int index = 0;
		for (int i = 0; i < contours.size(); i++) {
			//box = minAreaRect(Mat(contours[i]));//计算旋转矩阵
			//box.points(boxPts.data());
			boundRect = boundingRect(Mat(contours[i]));//计算最小外接矩形

			//问题所在  原因：没有*1.0 导致是一个整数 比例为1.x的矩形被舍去末尾变为1  小于1.5  导致有效轮廓被过滤掉
			if (double(boundRect.height * 1.0 / boundRect.width) > 1.5 && boundRect.width > 10 && boundRect.height > 20) {//满足条件的矩阵保存在point_array
				point_array[index] = boundRect;
				index++;
				rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//绘制矩形 左上 右下   
				//cout << "满足的比例：" << double(boundRect.height*1.0 / boundRect.width) << " 宽：" << boundRect.width <<" 高："<< boundRect.height << endl;
			}
			//else {//输出不满足条件的轮廓看看 用于test
			//	//cout << "不满足的比例：" << double(boundRect.height*1.0 / boundRect.width) << " 宽：" << boundRect.width << " 高：" << boundRect.height << endl;
			//}
		}

		if (index < 2) {//没检测到合格的轮廓 直接 下一帧  
			circle(frame, prePoint, 5, Scalar(255, 255, 255), -1);//若是只检测到1个边框或者没有边框 则打击前面几帧的点 直到再次被检测到
			imshow("video", frame);
			//writer.write(frame);
			cv::waitKey(10);
			//cout << "no" << endl;
			cout << "prePoint:" << prePoint << endl;
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

		hitPoint.x = (point3.x - point1.x) / 2 + point1.x;//矩形中心点
		hitPoint.y = (point4.y - point3.y) / 2 + point1.y;
		prePoint = hitPoint;
		cout << "hitPoint:" << hitPoint << endl;

		circle(frame, hitPoint, 5, Scalar(255, 255, 255), -1);
		//cout << p[0] << p[1] << p[2] << p[3] << endl;
		for (int i = 0; i < 4; i++) {
			line(frame, p[i % 4], p[(i + 1) % 4], Scalar(255, 255, 255), 2);
		}
	
		double endTime = cv::getTickCount();
		double totalTime = (endTime - startTime) / cv::getTickFrequency();
		cout << "运行时长" << totalTime << endl;

		imshow("video", frame);
		writer.write(frame);
		//cout << "yes" << endl;
		cv::waitKey(20);
	}
	cv::destroyAllWindows();

	waitKey(0);
}

//红色能量机关识别
bool colorr = false; //false 红色  true蓝色
void Game::Q2r() {
	namedWindow("image", WINDOW_NORMAL);
	//namedWindow("mask", WINDOW_NORMAL);
	namedWindow("binaryImage", WINDOW_NORMAL);

	//存放通道
	Mat Channels[3];
	Mat image;
	vector<Point2f> centerPoint;

	String s = "./images/2_red.mp4";

	VideoCapture capture(s);
	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);
	VideoWriter writer1("./images/x_red.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
	VideoWriter writer2("./images/x_red_bi.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), false);

	double pretheta=0;//前一帧的角度

	while (true)
	{
		double startTime = cv::getTickCount();
		if (!capture.read(image)) {
			break;
		}

		split(image, Channels);
		Mat colorimage = Channels[2] - Channels[0];//红色

		Mat binaryImage;
		threshold(colorimage, binaryImage, 110, 255, THRESH_BINARY);//红色最佳110

		vector<vector<Point>>contours;//轮廓数组
		vector<Vec4i>hierarchy; //一个参数
		//Point2i center; //用来存放找到的目标的中心坐标
		//提取所有轮廓并建立网状轮廓结构
		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

		//记录每个轮廓的子轮廓的个数 如果子轮廓为3则为目标扇     
		int contour[30] = { 0 };
		int min_area = 10000;
		int r_id = -1;//要画的圆心的轮廓的下标
		for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
		{
			if (hierarchy[i][3] != -1) //有父轮廓 
			{
				contour[hierarchy[i][3]]++; //对该父轮廓进行记录   目的是寻找值为3的轮廓 
			}
			double area = contourArea(contours[i]);

			//没有子轮廓 也没有父轮廓 且面积要合理 为圆心  还是有点问题。。。 会有误检测
			if (hierarchy[i][2] == -1 && hierarchy[i][3] == -1&& (area > 20 && area < 1000)) {
				//再用矩形来拟合 若长宽比为正方形才是圆心
				Rect rect = boundingRect(contours[i]);
				if (double(rect.height * 1.0 / rect.width) < 1.2 && double(rect.height * 1.0 / rect.width) > 0.8) {//圆心位置接近正方形
					if (area <= min_area) {//还要是面积最小的
						min_area = area;
						r_id = i;
					}
				}
			}
		}

		Point2f newcenter;//R圆心
		float newradius;//半径

		Point2f center;//击打点圆心
		float radius;

		if (r_id > 0 && r_id < contours.size()) {//在合理范围内
			minEnclosingCircle(Mat(contours[r_id]), newcenter, newradius);//计算给定轮廓的最小外接圆的半径和圆心
			circle(image, newcenter, newradius, Scalar(0, 255, 255), 5, 8,0);
		}
		
		for (int i = 0; i < 30; i++) {
			if (contour[i] >= 3) {//目标扇叶
				//drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8);
				//imshow("image", image);//符合预先设想
				//waitKey(0);

				//对目标扇叶绘制大矩形 
				RotatedRect rect0 = minAreaRect(contours[i]);
				Point2f vertices0[4];
				rect0.points(vertices0);

				//再找出目标扇叶的两个子轮廓 方便缩小范围 实现精确打击
				int child1 = hierarchy[i][2];//目标扇叶的的第一个子轮廓
				int child2, child3;
				if (hierarchy[child1][0] != -1) {//目标扇叶的的第一个子轮廓的同级轮廓（上一个或者下一个） 也就是目标扇叶的第二个子轮廓  
					child2 = hierarchy[child1][0];
				}
				else {
					child2 = hierarchy[child1][1];
				}

				if (hierarchy[child2][0] != -1) {//同理
					child3 = hierarchy[child2][0];
				}
				else {
					child3 = hierarchy[child2][1];
				}

				//drawContours(binaryImage, contours, child1, Scalar(255, 255, 255), 2, 8);
				//drawContours(binaryImage, contours, child2, Scalar(255, 255, 255), 2, 8);
				//drawContours(binaryImage, contours, child3, Scalar(255, 255, 255), 2, 8);

				vector<Point> points;//三个子轮廓所包含所有的点  用这些点拟合一个圆 圆心就是要打击的点
				points.insert(points.end(), contours[child1].begin(), contours[child1].end());
				points.insert(points.end(), contours[child2].begin(), contours[child2].end());
				points.insert(points.end(), contours[child3].begin(), contours[child3].end());

				//拟合圆
				minEnclosingCircle(points, center, radius);
				circle(image, center, 8, Scalar(255, 255, 255), -1, LINE_AA);
				circle(binaryImage, center, 6, Scalar(255, 255, 255), -1, LINE_AA);
				cout << center << endl;

				//绘制小矩形
				//RotatedRect rect1 = minAreaRect(points);
				// 获取最小面积矩形的四个顶点
				//Point2f vertices1[4];
				//rect1.points(vertices1);

				for (int i = 0; i < 4; i++) {
					//line(image, vertices1[i % 4], vertices1[(i + 1) % 4], Scalar(255, 255, 255),2);
					//line(binaryImage, vertices1[i % 4], vertices1[(i + 1) % 4], Scalar(255, 255, 255), 2);
					line(image, vertices0[i % 4], vertices0[(i + 1) % 4], Scalar(255, 255, 255), 2);
					line(binaryImage, vertices0[i % 4], vertices0[(i + 1) % 4], Scalar(255, 255, 255), 2);
				}
				//circle(image, rect1.center, 5, Scalar(255, 255, 255), -1);
				//centerPoint.push_back(rect1.center);
				//cout<<rect1.center<<endl;
			}
		}

		//待添加计数功能 只有某一方向达到一定数量才认为是该方向，防止出现“跳帧”情况
		////注意 第一个值是y
		//cout << "一二象限点" << (atan2(1, 1) * (180 / 3.1415926)) << " " << (atan2(1, -1) * (180 / 3.1415926)) << endl;//以角度显示
		//cout << "三四象限点" << (atan2(-1, -1) * (180 / 3.1415926)) << " " << (atan2(-1, 1) * (180 / 3.1415926)) << endl;
		//对打击点中心计算极坐标系的角度 利用两帧之间的角度差进行判断是顺时针还是逆时针
		//以R的圆心为坐标原点
		double dx = center.x - newcenter.x;
		double dy = center.y - newcenter.y;
		double p = sqrt(dx * dx + dy * dy);//斜边
		double theta = atan2(dy, dx) * (180 / PI);//换算成角度
		if (dy > 0) {//只在第一/二象限判断
			if (pretheta != 0) {
				double dtheta = theta - pretheta;//当前-前一帧  为什么计算结果当前帧会比前一帧的小？？？ 按理应该大才对啊？？？
				cout << "dx " << dx << "dy " << dy;
				cout << "当前角度" << theta << " 前一帧角度" << pretheta << " 角度差" << dtheta << endl;
				if (dtheta > 0) {//逆时针
					//cout << "逆时针" << endl;
					cout << "顺时针" << endl;
				}
				else {
					//cout << "顺时针" << endl;
					cout << "逆时针" << endl;
				}
			}
			pretheta = theta;
		}

		double endTime = cv::getTickCount();
		double totalTime = (endTime - startTime) / cv::getTickFrequency();

		//writer1.write(image);
		//writer2.write(binaryImage);
		imshow("image", image);
		imshow("binaryImage", binaryImage);
		cout << "运行时间" << totalTime << endl;//识别红色的效率远远高于识别蓝色的效率？？？
		//imshow("mask", mask);
		waitKey(1);
	}
	waitKey(0);
}

//蓝色能量机关识别
void Game::Q2b() {
	namedWindow("image", WINDOW_NORMAL);
	namedWindow("mask", WINDOW_NORMAL);
	namedWindow("binaryImage", WINDOW_NORMAL);

	Mat binaryImage;
	Mat Channels[3];

	Mat image;
	vector<Point2f> centerPoint;

	String s = "./images/1_blue.mp4";
	VideoCapture capture(s);
	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);
	VideoWriter writer1("./images/x_blue.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
	VideoWriter writer2("./images/x_blue_bi.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), false);

	while (true)
	{
		double startTime = cv::getTickCount();

		if (!capture.read(image)) {
			break;
		}

		split(image, Channels);
		threshold(Channels[0], binaryImage, 245, 255, THRESH_BINARY);
		Mat k = getStructuringElement(1, Size(3, 3));
		morphologyEx(binaryImage, binaryImage, 3, k, Point(-1, -1), 3);//复杂度过高，换一种方式提取蓝色特征吧。。。

		vector<vector<Point>>contours;//轮廓数组
		vector<Vec4i>hierarchy; //一个参数
		Point2i center; //用来存放找到的目标的中心坐标
		//提取所有轮廓并建立网状轮廓结构
		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

		//记录每个轮廓的子轮廓的个数 如果子轮廓为5则为目标扇
		int contour[30] = { 0 };
		for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
		{
			if (hierarchy[i][3] != -1) //有父轮廓 
			{
				contour[hierarchy[i][3]]++; //对该父轮廓进行记录   目的是寻找父轮廓为3的轮廓 
			}
		}

		for (int i = 0; i < 30; i++) {
			if (contour[i] >= 5) {//目标扇叶

				//对目标扇叶绘制矩形
				RotatedRect rect0 = minAreaRect(contours[i]);
				Point2f vertices0[4];
				rect0.points(vertices0);

				//再找出目标扇叶的子轮廓 方便缩小范围 实现精确打击
				int child1 = hierarchy[i][2];//目标扇叶的的第一个子轮廓
				int child2, child3, child4, child5;;
				if (hierarchy[child1][0] != -1) {//目标扇叶的的第一个子轮廓的同级轮廓（上一个或者下一个） 也就是目标扇叶的第二个子轮廓  
					child2 = hierarchy[child1][0];
				}
				else {
					child2 = hierarchy[child1][1];
				}

				if (hierarchy[child2][0] != -1) {//同理
					child3 = hierarchy[child2][0];
				}
				else {
					child3 = hierarchy[child2][1];
				}

				if (hierarchy[child3][0] != -1) {//同理
					child4 = hierarchy[child3][0];
				}
				else {
					child4 = hierarchy[child3][1];
				}

				if (hierarchy[child4][0] != -1) {//同理
					child5 = hierarchy[child4][0];
				}
				else {
					child5 = hierarchy[child4][1];
				}

				//drawContours(binaryImage, contours, child1, Scalar(255, 255, 255), 2, 8);
				//drawContours(binaryImage, contours, child2, Scalar(255, 255, 255), 2, 8);
				//drawContours(binaryImage, contours, child3, Scalar(255, 255, 255), 2, 8);
				//drawContours(binaryImage, contours, child4, Scalar(255, 255, 255), 2, 8);
				//drawContours(binaryImage, contours, child5, Scalar(255, 255, 255), 2, 8);

				vector<Point> points;//三个子轮廓所包含所有的点
				points.insert(points.end(), contours[child1].begin(), contours[child1].end());
				points.insert(points.end(), contours[child2].begin(), contours[child2].end());
				points.insert(points.end(), contours[child3].begin(), contours[child3].end());
				points.insert(points.end(), contours[child4].begin(), contours[child4].end());
				points.insert(points.end(), contours[child5].begin(), contours[child5].end());

				Point2f center;
				float radius;
				minEnclosingCircle(points, center, radius);
				circle(image, center, 8, Scalar(255, 255, 255), -1, LINE_AA);
				circle(binaryImage, center, 6, Scalar(255, 255, 255), -1, LINE_AA);
				//cout << center << endl;

				for (int i = 0; i < 4; i++) {
					line(image, vertices0[i % 4], vertices0[(i + 1) % 4], Scalar(255, 255, 255), 2, LINE_AA);
					line(binaryImage, vertices0[i % 4], vertices0[(i + 1) % 4], Scalar(255, 255, 255), 2, LINE_AA);
				}
			}
		}

		writer1.write(image);
		writer2.write(binaryImage);
		imshow("image", image);
		imshow("binaryImage", binaryImage);

		double endTime = cv::getTickCount();
		double totalTime = (endTime - startTime) / cv::getTickFrequency();

		cout << totalTime << endl;//效率太低了....
		waitKey(1);
	}
	cout << "finish" << endl;
	waitKey(0);
}

//picture test  用于测试
void Game::test1() {
	Mat image = imread("./images/dred.png");

	//Mat Channels[3];
	//split(image, Channels);
	//Mat colorimage = Channels[0] - Channels[2];
	//Mat binaryImage; 
	//threshold(colorimage, binaryImage, 140, 255, THRESH_BINARY);
	//Mat k = getStructuringElement(1, Size(3, 3));//操作后效果更加不好
	//morphologyEx(binaryImage, binaryImage,3, k,Point(-1,-1));  //pass

/*	Mat hsv,out, binaryImage;
	cvtColor(image, hsv, COLOR_RGB2HSV);
	inRange(hsv, Scalar(0, 0, 221), Scalar(180, 30, 255), out);*///扣不出来？？？

	Mat Channels[3];
	split(image, Channels);
	Mat colorimage = Channels[2] - Channels[0];//红色

	Mat binaryImage;
	threshold(colorimage, binaryImage, 110, 255, THRESH_BINARY);//红色最佳110

	vector<vector<Point>>contours;//轮廓数组
	vector<Vec4i>hierarchy; //一个参数
	Point2i center; //用来存放找到的目标的中心坐标
	//提取所有轮廓并建立网状轮廓结构
	findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

	//记录每个轮廓的子轮廓的个数 如果子轮廓为3则为目标扇     
	int contour[30] = { 0 };
	int min_area = 10000;
	int r_id = 0;
	for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
	{
		//if (hierarchy[i][3] != -1) //有父轮廓 
		//{
		//	contour[hierarchy[i][3]]++; //对该父轮廓进行记录   目的是寻找父轮廓为3的轮廓 
		//}
		double area = contourArea(contours[i]);
		//没有子轮廓 也没有父轮廓 且面积要合理 为圆心  还是有点问题。。。
		if (hierarchy[i][2] == -1 && hierarchy[i][3] == -1 && (area > 20 && area < 1000)) {
			//再用矩形来拟合 若长宽比为正方形才是圆心
			Rect rect = boundingRect(contours[i]);
			if (double(rect.height * 1.0 / rect.width) < 1.2 && double(rect.height * 1.0 / rect.width) > 0.8) {//圆心位置接近正方形
				cout << double(rect.height * 1.0 / rect.width) << endl;
				drawContours(image, contours, i, Scalar(0, 255, 255), 2, 8);
				imshow("image", image);
				waitKey(0);
				if (area <= min_area) {//还要是面积最小的
					min_area = area;
					r_id = i;
					cout << "yes" << endl;
				}
			}
		}
		imshow("image", image);
	}
	drawContours(image, contours, r_id, Scalar(255, 255, 255), 2, 8);
	imshow("2", image);
	waitKey(0);


	//for (int t = 0; t < contours.size(); t++) {
	//	drawContours(image, contours, t, Scalar(0, 255, 0), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
	//	imshow("image", image);
	//	waitKey(0);
	//}

	////寻找有五个子轮廓的轮廓 就是要击打的扇叶
	//int contour[30] = { 0 };
	//for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
	//{
	//	if (hierarchy[i][3] != -1) //有父轮廓 
	//	{
	//		contour[hierarchy[i][3]]++; //对该父轮廓进行记录  目的是寻找父轮廓为3的轮廓 
	//	}
	//}


	//for (int i = 0; i < 30; i++) {
	//	if (contour[i] >= 5) {//目标扇叶
	//		drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8);
	//		imshow("image", image);//符合预先设想
	//		waitKey(0);

	//		////再找出目标扇叶的两个子轮廓 方便缩小范围 实现精确打击
	//		int child1 = hierarchy[i][2];//目标扇叶的的第一个子轮廓
	//		int child2, child3;
	//		if (hierarchy[child1][0] != -1) {//目标扇叶的的第一个子轮廓的同级轮廓（上一个或者下一个） 也就是目标扇叶的第二个子轮廓  
	//			child2 = hierarchy[child1][0];
	//		}
	//		else {
	//			child2 = hierarchy[child1][1];
	//		}

	//		if (hierarchy[child2][0] != -1) {//同理
	//			child3 = hierarchy[child2][0];
	//		}
	//		else {
	//			child3 = hierarchy[child2][1];
	//		}

	//		drawContours(image, contours, child1, Scalar(255, 255, 255), 2, 8);
	//		imshow("image", image);
	//		waitKey(0);

	//		drawContours(image, contours, child2, Scalar(255, 255, 255), 2, 8);
	//		imshow("image", image);
	//		waitKey(0);

	//		drawContours(image, contours, child3, Scalar(255, 255, 255), 2, 8);
	//		imshow("image", image);
	//		waitKey(0);
	//	}
	//}
	////Mat blueImage = Channels.at(0) - Channels.at(2);//蓝色
	//waitKey(0);
}


//对bar的理解
Mat img;
Mat mask;
void onTrackbar(int value, void* userdata)
{
	//imshow("img2", img);
	int red_value = getTrackbarPos("Red", "Image");//
	int green_value = getTrackbarPos("Green", "Image");
	int blue_value = getTrackbarPos("Blue", "Image");

	Mat dst(img.size(), img.type());
	// 获取滑动条的当前值
	cout << red_value << " " << green_value << " " << blue_value << endl;
	// 更新图像的三个通道的像素值
	dst = Scalar(red_value, green_value, blue_value);
	//imshow("dst", dst);//没问题

	Mat add(img.size(), img.type());//加权结果
	addWeighted(dst, 0.5, img, 0.5, 0, add);
	//imshow("add", add);

	Mat imgc = img.clone();
	add.copyTo(imgc, mask);//原先的方案会改变img（原始图像）
	imshow("Image", imgc);
}

void Game::Dog() {
	Mat img = imread("./images/dog2.png");
	Mat mask = Mat(img.size(), img.type());

	//img = imread("./images/dog2.jpg");
	//mask = Mat(img.size(), img.type());

	Mat channels[3];

	split(img, channels);
	threshold(channels[0], mask, 100, 255, THRESH_BINARY);//提取要变色的图像
	//imshow("mask", mask);

	int r = 0;
	int b = 0;
	int g = 0;
	namedWindow("Image", WINDOW_NORMAL);
	resizeWindow("Image", 500, 500);

	//对dst进行变色 再赋值回原图像
	createTrackbar("Red", "Image", &r, 255, onTrackbar);
	createTrackbar("Green", "Image", &b, 255, onTrackbar);
	createTrackbar("Blue", "Image", &g, 255, onTrackbar);

	waitKey(0);
}

	
//void Game::test2() {
//	VideoCapture cap("./images/2_red.mp4");
//	Mat image;
//	//colorrr = blue;/*蓝色*/
//	for (;;) {
//		cap.read(image);
//		/*开始处理图像*/
//		clock_t start = clock();
//		/*改变大小，提高帧率*/
//		//Mat image= imread("./images/dred.png");
//		resize(image, image, Size(image.cols * 0.35, image.rows * 0.35));
//
//		/*测试效果展示*/
//		Mat test;
//		image.copyTo(test);
//		///////////////////////////////////////////////这部分操作需要换成自己的///////////////////////////////////////////////////////////
//		///*容器，存放分离通道后的图像*/
//		//vector<Mat> imgChannels;
//		//split(image, imgChannels);
//		//
//		////红色
//		//Mat midimage = imgChannels.at(2) - imgChannels.at(0);
//		//imshow("1", midimage);
//		//
//		///*蓝色*/
//		////Mat blueImage = imgChannels.at(0) - imgChannels.at(2);
//		//Mat binaryImage;
//		//Mat binaryImagecricle;
//		///*二值化*/
//		//threshold(midimage, binaryImagecricle, 170, 255, THRESH_BINARY);
//		//threshold(midimage, binaryImage, 90, 255, THRESH_BINARY);
//		///*腐蚀操作*/
//		Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
//		//Mat dstImage;
//		//erode(binaryImagecricle, dstImage, element);
//		
//		Mat Channels[3];
//		split(image, Channels);
//		Mat colorimage = Channels[2] - Channels[0];//红色
//		Mat dstImage, binaryImage;
//		threshold(colorimage, dstImage, 110, 255, THRESH_BINARY);
//		threshold(colorimage, binaryImage, 110, 255, THRESH_BINARY);
//
//		///////////////////////////////////////////////////////////////////////////////////////////////////////////
//		/*找到圆周运动的圆心——R*/
//		state = retain; //state 是否偏移  保持
//		vector<vector<Point>> outlines;//轮廓
//		vector<Vec4i> hierarchies;
//		findContours(dstImage, outlines, hierarchies, RETR_TREE, CHAIN_APPROX_NONE);
//		//for (int t = 0; t < outlines.size(); t++) {//圆心可以找没有子轮廓且没有父轮廓的
//		//	drawContours(image, outlines, t, Scalar(0, 255, 0), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
//		//	imshow("image", image);
//		//	waitKey(0);
//		//}
//
//		for (int i = 0; i < outlines.size(); i++) {
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
//
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
//		}
//
//		/*防止minId不在范围内报错*/
//		if (minId >= 0 && minId < outlines.size())
//		{
//			/*画外接圆并找到圆心*/
//			minEnclosingCircle(Mat(outlines[minId]), newcenter, newradius); //计算得到的最小外接圆的中心坐标和半径
//			/*减小抖动，误差*/
//			if (distance(newcenter, center) < 2) {
//			}
//			else {
//				oldcenter = center;
//				center = newcenter;
//				state = change;
//			}
//
//			if (fabs(newradius - radius) < 2) {
//			}
//			else {
//				radius = newradius;
//			}
//			circle(test, center, radius, Scalar(0, 0, 255), 1, 8, 0);
//		}
//		else {
//			//continue;
//		}///////////////////////////////////////////////圆心是用找轮廓的方式找到的
//
//		/*膨胀操作*/
//		element = getStructuringElement(0, Size(3, 3));
//		//Mat dilateImage;
//		/*dilate最后一个数字是膨胀次数*/
//		//dilate(binaryImage, dilateImage, element, Point(-1, -1), 2);
//		/*轮廓发现*/
//		vector<vector<Point>> contours;
//		vector<Vec4i> hierarchy;
//		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
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
//		}
//
//		if (maxId >= 0 && maxId < contours.size()) {
//			/*计算矩*/
//			Moments rect;
//			rect = moments(contours[maxId], false);
//			/*计算中心矩:*/
//			Point2f rectmid;
//			rectmid = Point2f(rect.m10 / rect.m00, rect.m01 / rect.m00);
//			/*画出需打部位轮廓*/
//			drawContours(test, contours, maxId, Scalar(0, 255, 255), 1, 8);
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
//
//			/*画出样本部位中心点*/
//			circle(test, rectMid, 1, Scalar(0, 255, 255), -1, 8, 0);
//			/*2：1计算需打击部位,存放*/
//			/*第一象限*/
//			if (rectMid.x >= center.x && rectMid.y <= center.y) {
//				target = Point2f(center.x + (rectMid.x - center.x) * multiple, center.y - (center.y - rectMid.y) * multiple);
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
//		}
//		else {
//			//continue;
//		}
//
//
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
//		/*预测*/
//		if (runNumber > pointSetnumeber) {
//
//			int i = pointNumber - 1;//取最新的点算速度
//			int number1 = i;
//			if (number1 < 0) {
//				number1 += pointSetnumeber;
//			}
//
//			int number2 = i - 1;
//			if (number2 < 0) {
//				number2 += pointSetnumeber;
//			}
//
//			int number3 = i - 3;
//			if (number3 < 0) {
//				number3 += pointSetnumeber;
//			}
//
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
//			/* X = V0T + 1 / 2AT'2，通过距离公式，算预测的打击点距离 */  //？？？
//			predictdistance = (4.5 * speedSet[0] / frame) + 1 / 2 * acceleratedSpeed / frame / frame * 18;
//
//			/*算出预测时x, y需增加值的比值*/
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
//		imshow("2", binaryImage);
//		runNumber++;
//		clock_t end = clock();
//		frame = 1 / (double)(end - start) * CLOCKS_PER_SEC;
//		cout << frame << "帧" << endl;
//		waitKey(1);
//	}
//		return;
//}


//卡尔曼滤波
//void Game::test2() {
//	VideoCapture cap("F:/rm/卡尔曼预测/Windmills-master/wind.mp4");
//	Mat image, binary;
//	int stateNum = 4;
//	int measureNum = 2;
//
//	KalmanFilter KF(stateNum, measureNum, 0);
//	//Mat processNoise(stateNum, 1, CV_32F);
//	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
//	KF.transitionMatrix = (Mat_<float>(stateNum, stateNum) << 1, 0, 1, 0,//A 状态转移矩阵
//		0, 1, 0, 1,
//		0, 0, 1, 0,
//		0, 0, 0, 1);
//
//	//这里没有设置控制矩阵B，默认为零
//	setIdentity(KF.measurementMatrix);//H=[1,0,0,0;0,1,0,0] 测量矩阵
//	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));//Q高斯白噪声，单位阵
//	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//R高斯白噪声，单位阵
//	setIdentity(KF.errorCovPost, Scalar::all(1));//P后验误差估计协方差矩阵，初始化为单位阵
//	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));//初始化状态为随机值
//
//	for (;;) {
//		cap.read(image);
//
//		image.copyTo(binary);
//		resize(image, image, Size(image.cols * 0.5, binary.rows * 0.5));
//		resize(binary, binary, Size(binary.cols * 0.5, binary.rows * 0.5));
//
//		cvtColor(image, image, COLOR_BGR2GRAY);
//
//		threshold(image, image, 80, 255, THRESH_BINARY);        //阈值要自己调
//
//		dilate(image, image, Mat());
//		dilate(image, image, Mat());
//
//		floodFill(image, Point(5, 50), Scalar(255), 0, FLOODFILL_FIXED_RANGE);
//
//		threshold(image, image, 80, 255, THRESH_BINARY_INV);
//
//		vector<vector<Point>> contours;
//		findContours(image, contours, RETR_LIST, CHAIN_APPROX_NONE);
//		for (size_t i = 0; i < contours.size(); i++) {
//
//			vector<Point> points;
//			double area = contourArea(contours[i]);
//			if (area < 50 || 1e4 < area) continue;
//			drawContours(image, contours, static_cast<int>(i), Scalar(0), 2);
//
//			points = contours[i];
//			RotatedRect rrect = fitEllipse(points);
//			cv::Point2f* vertices = new cv::Point2f[4];
//			rrect.points(vertices);
//
//			float aim = rrect.size.height / rrect.size.width;
//			if (aim > 1.7 && aim < 2.6) {
//				for (int j = 0; j < 4; j++)
//				{
//					cv::line(binary, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 4);
//				}
//				float middle = 100000;
//
//				for (size_t j = 1; j < contours.size(); j++) {
//
//					vector<Point> pointsA;
//					double area = contourArea(contours[j]);
//					if (area < 50 || 1e4 < area) continue;
//
//					pointsA = contours[j];
//
//					RotatedRect rrectA = fitEllipse(pointsA);
//
//					float aimA = rrectA.size.height / rrectA.size.width;
//
//					if (aimA > 3.0) {
//						float distance = sqrt((rrect.center.x - rrectA.center.x) * (rrect.center.x - rrectA.center.x) +
//							(rrect.center.y - rrectA.center.y) * (rrect.center.y - rrectA.center.y));
//
//						if (middle > distance)
//							middle = distance;
//					}
//				}
//				if (middle > 60) {//这个距离也要根据实际情况调,和图像尺寸和物体远近有关。
//					cv::circle(binary, Point(rrect.center.x, rrect.center.y), 15, cv::Scalar(0, 0, 255), 4);
//					Mat prediction = KF.predict();
//					Point predict_pt = Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));
//
//					measurement.at<float>(0) = (float)rrect.center.x;
//					measurement.at<float>(1) = (float)rrect.center.y;
//					KF.correct(measurement);
//
//					circle(binary, predict_pt, 3, Scalar(34, 255, 255), -1);
//
//					rrect.center.x = (int)prediction.at<float>(0);
//					rrect.center.y = (int)prediction.at<float>(1);
//				}
//			}
//		}
//		imshow("frame", binary);
//		imshow("Original", image);
//		waitKey(1);
//	}
//}

//


