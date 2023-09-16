
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

void Game::Q1p() {//
	//如果只是通过对轮廓的面积大小对背景光条进行筛选 在比赛中会根据近大远小原则 在靠近装甲板光条的时候是否不适应？
	//如果是通过对轮廓的长宽比例来筛选的话，小光条和大光条的轮廓比例似乎差不多
	//如果是通过色彩的阈值来筛选，大小光条的亮度值很接近，无法实现  //pass1


	//先进行初筛选 将长宽比小于2 大于10的先 筛选掉 再将剩余的进行就近组合成矩形 再对矩形的长宽比进行筛选 
	//如何确定两两匹配的光条？  通过矩阵的中心点距离来判断？（距离在比赛中不适用吧？ 近大远小 不可能在某一点固定不动吧？）
	//面积大小相似？ 不行
	//中心点的纵坐标？ 能解决匹配到不在一个层面的光条矩形的问题 但是不能解决在同一层面的矩形匹配问题  面积差不多 且角度差不多
	//查找资料得知可以用旋转矩阵的角度来匹配

	Mat frame = imread("./images/zj.png");
	Mat channels[3], binary, Gaussian;

	vector<vector<Point>> contours;//轮廓点
	vector<Vec4i> hierarchy;
	RotatedRect AreaRect;//存储最小(旋转)外接矩形

	split(frame, channels);
	//Mat colorimage = channels[0] - channels[2];

	threshold(channels[0], binary, 180, 255, 0);//对b进行二值化
	GaussianBlur(binary, Gaussian, Size(3, 3), 0);//size越小越好
	findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	RotatedRect point_array[20];//存筛选后的旋转矩阵
	Point2f points[4];//存RotatedRect转换的四点坐标

	int index = 0;
	for (int i = 0; i < contours.size(); i++) {
		//boundRect = boundingRect(Mat(contours[i]));//计算最小外接矩形
		AreaRect = minAreaRect(Mat(contours[i]));//计算最小的旋转矩阵 因为要使用angle

		double width = AreaRect.size.width;//宽
		double heigth = AreaRect.size.height;//高
		double area = AreaRect.size.area();//面积

		//判断有问题
		if (area > 100) { //先把一些太小的干扰轮廓筛选删除
			point_array[index] = AreaRect;
			AreaRect.points(points);

			//rectangle(frame, point_array[index].tl(), point_array[index].br(), Scalar(255, 255, 255), 2, 8, 0);//左上 右下
			for (int i = 0; i < 4; i++) {
				line(frame, points[i % 4], points[(i + 1) % 4], Scalar(255, 255, 255), 2);
			}
			index++;
		}
	}

	imshow("frame", frame);
	waitKey(0);

	//for (int t = 0; t < contours.size(); t++) {
	//	drawContours(frame, contours, t, Scalar(255, 255, 255), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
	//	imshow("frame", frame);
	//	waitKey(0);
	//}
}

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
	//VideoWriter writer("./images/zjbb.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);//保存图像 用于查看哪里识别效果不好

	Mat channels[3], binary, Gaussian;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Point prePoint;//上一次打击点
	Point hitPoint;//当前打击点

	namedWindow("video", WINDOW_NORMAL);

	Rect boundRect;//存储最小外接矩形
	while (true)
	{
		Rect point_array[50];//存储合格的外接矩阵
		if (!video.read(frame)) {
			break;
		}
		split(frame, channels);
		threshold(channels[t], binary, 180, 255, 0);//对r进行二值化

		GaussianBlur(binary, Gaussian, Size(5, 5), 0);
		//Mat k = getStructuringElement(1, Size(3, 3));
		findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

		int index = 0;
		for (int i = 0; i < contours.size(); i++) {
			//box = minAreaRect(Mat(contours[i]));//计算旋转矩阵
			//box.points(boxPts.data());
			boundRect = boundingRect(Mat(contours[i]));//计算最小外接矩形
			rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//左上 右下

			try
			{
				//问题所在  原因：没有*1.0 导致是一个整数 比例为1.x的矩形被舍去末尾变为1  小于1.5  导致有效轮廓被过滤掉
				if (double(boundRect.height * 1.0 / boundRect.width) > 1.5 && boundRect.width > 10 && boundRect.height > 20) {//满足条件的矩阵保存在point_array
					point_array[index] = boundRect;
					index++;
					//cout << "满足的比例：" << double(boundRect.height*1.0 / boundRect.width) << " 宽：" << boundRect.width <<" 高："<< boundRect.height << endl;
				}
				//else {//输出不满足条件的轮廓看看 用于test
				//	//cout << "不满足的比例：" << double(boundRect.height*1.0 / boundRect.width) << " 宽：" << boundRect.width << " 高：" << boundRect.height << endl;
				//}
			}
			catch (const char* msg)
			{
				cout << printf(msg) << endl;
				continue;
			}
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

			hitPoint.x = (point3.x - point1.x) / 2 + point1.x;//矩形中心点
			hitPoint.y = (point4.y - point3.y) / 2 + point1.y;
			prePoint = hitPoint;
			cout << "hitPoint:" << hitPoint << endl;

			circle(frame, hitPoint, 5, Scalar(255, 255, 255), -1);
			//cout << p[0] << p[1] << p[2] << p[3] << endl;
			for (int i = 0; i < 4; i++) {
				line(frame, p[i % 4], p[(i + 1) % 4], Scalar(255, 255, 255), 2);
			}
		}
		catch (const char* msg)
		{
			cout << msg << endl;
		}

		imshow("video", frame);
		//writer.write(frame);
		//cout << "yes" << endl;
		cv::waitKey(20);
	}
	cv::destroyAllWindows();

	waitKey(0);
}

//红色能量机关识别
bool colorr = false; //false 红色  true蓝色
void Game::Q2r() {
	//问题：
	//1.识别蓝色的效果远远不如红色  通过可视化二值化图发现阈值效果不理想 
	//2.红色效果也变差了？？？ 原因：顺手调了阈值。。。

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
		//

		vector<vector<Point>>contours;//轮廓数组
		vector<Vec4i>hierarchy; //一个参数
		Point2i center; //用来存放找到的目标的中心坐标
		//提取所有轮廓并建立网状轮廓结构
		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

		//记录每个轮廓的子轮廓的个数 如果子轮廓为3则为目标扇
		int contour[30] = { 0 };
		for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
		{
			if (hierarchy[i][3] != -1) //有父轮廓 
			{
				contour[hierarchy[i][3]]++; //对该父轮廓进行记录   目的是寻找父轮廓为3的轮廓 
			}
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
				Point2f center;
				float radius;
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

		double endTime = cv::getTickCount();
		double totalTime = (endTime - startTime) / cv::getTickFrequency();

		//writer1.write(image);
		//writer2.write(binaryImage);
		imshow("image", image);
		imshow("binaryImage", binaryImage);
		cout << totalTime << endl;//识别红色的效率远远高于识别蓝色的效率？？？
		//imshow("mask", mask);
		waitKey(1);
	}

	cout << "finish" << endl;
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

//picture test
void Game::Q2() {
	Mat image = imread("./images/dblue2.png");

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

	Mat binaryImage;
	Mat Channels[3];
	split(image, Channels);
	Mat colorimage = Channels[0] - Channels[1];

	threshold(colorimage, binaryImage, 70, 255, THRESH_BINARY);
	//Mat k = getStructuringElement(1, Size(3, 3));
	//morphologyEx(binaryImage, binaryImage, 3, k,Point(-1,-1),2); //pass
	//morphologyEx(binaryImage, binaryImage, 0, k, Point(-1, -1));

	vector<vector<Point>>contours;//轮廓数组
	vector<Vec4i>hierarchy; //一个参数
	Point2i center; //用来存放找到的目标的中心坐标
	//提取所有轮廓并建立网状轮廓结构
	findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

	for (int t = 0; t < contours.size(); t++) {
		drawContours(image, contours, t, Scalar(0, 255, 0), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
		imshow("image", image);
		waitKey(0);
	}

	//寻找有五个子轮廓的轮廓 就是要击打的扇叶
	int contour[30] = { 0 };
	for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
	{
		if (hierarchy[i][3] != -1) //有父轮廓 
		{
			contour[hierarchy[i][3]]++; //对该父轮廓进行记录  目的是寻找父轮廓为3的轮廓 
		}
	}


	for (int i = 0; i < 30; i++) {
		if (contour[i] >= 5) {//目标扇叶
			drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8);
			imshow("image", image);//符合预先设想
			waitKey(0);

			////再找出目标扇叶的两个子轮廓 方便缩小范围 实现精确打击
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

			drawContours(image, contours, child1, Scalar(255, 255, 255), 2, 8);
			imshow("image", image);
			waitKey(0);

			drawContours(image, contours, child2, Scalar(255, 255, 255), 2, 8);
			imshow("image", image);
			waitKey(0);

			drawContours(image, contours, child3, Scalar(255, 255, 255), 2, 8);
			imshow("image", image);
			waitKey(0);
		}
	}
	//Mat blueImage = Channels.at(0) - Channels.at(2);//蓝色
	waitKey(0);
}




//用于参考筛选轮廓
//vector<vector<Point>> contours;
//vector<Vec4i> hierarchy;
//double maxArea = -1;
//int maxId;
//findContours(dilateImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
//for (int i = 0; i < contours.size(); i++) {
//	vector<Point>points;
//	double area = contourArea(contours[i]);
//	/*面积排除噪声*/
//	if (area < 20 || area>10000)
//		continue;
//	/*找到没有父轮廓的轮廓*/
//	if (hierarchy[i][3] >= 0 && hierarchy[i][3] < contours.size())
//		continue;
//	/*找没子轮廓的*/
//	if (hierarchy[i][2] >= 0 && hierarchy[i][2] < contours.size())
//		continue;
//	/*找面积最大的轮廓*/
//	if (maxArea <= area)
//	{
//		maxArea = area;
//		maxId = i;
//	}
//	/*控制误差范围*/
//	if (area <= maxArea + 50 && area >= maxArea - 50) {
//		maxArea = area;
//		maxId = i;
//	}
//}
//if (maxId >= 0 && maxId < contours.size()) {
//	/*画出需打部位轮廓*/
//	drawContours(test, contours, maxId, Scalar(0, 255, 255), 1, 8);
//}








//对bar的理解
//Mat img;
//Mat mask;
//void onTrackbar(int value, void* userdata)
//{
//	//imshow("img2", img);
//	int red_value = getTrackbarPos("Red", "Image");//
//	int green_value = getTrackbarPos("Green", "Image");
//	int blue_value = getTrackbarPos("Blue", "Image");
//
//	Mat dst(img.size(), img.type());
//	// 获取滑动条的当前值
//	cout << red_value<< " " << green_value<<" " << blue_value << endl;
//	// 更新图像的三个通道的像素值
//	dst = Scalar(red_value, green_value, blue_value);
//	//imshow("dst", dst);//没问题
//
//	Mat add(img.size(), img.type());//加权结果
//	addWeighted(dst, 0.5, img, 0.5, 0, add);
//	//imshow("add", add);
//
//	Mat imgc = img.clone();
//	add.copyTo(imgc, mask);//原先的方案会改变img（原始图像）
//	imshow("Image", imgc);
//}
//
//void Game::Dog() {
//	img = imread("./images/dog.jpg");
//	Mat channels[3];
//	mask = Mat(img.size(),img.type());
//	//画板
//
//	split(img, channels);
//	threshold(channels[0], mask, 150, 255, THRESH_BINARY);//提取要变色的图像
//
//	int r = 0;
//	int b = 0;
//	int g = 0;
//	namedWindow("Image", WINDOW_NORMAL);
//	resizeWindow("Image", 500, 500);
//
//	//namedWindow("dst", WINDOW_NORMAL);
//	//resizeWindow("dst", 500, 500);
//
//	//namedWindow("add", WINDOW_NORMAL);
//	//resizeWindow("add", 500, 500);
//
//
//	//对dst进行变色 再赋值回原图像
//	createTrackbar("Red", "Image", &r, 255, onTrackbar);
//	createTrackbar("Green", "Image", &b, 255, onTrackbar);
//	createTrackbar("Blue", "Image", &g, 255, onTrackbar);
//
//	waitKey(0);
//}

