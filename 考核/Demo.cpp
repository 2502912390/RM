#include <opencv2/opencv.hpp>
#include <iostream>
#include <All.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<algorithm>
#include<set>
#include<map>

using namespace std;
using namespace cv;

/************************************************BASE********************************************************/
void Base::Q1try() {//
	
	Mat img = imread("./images/q1.png");

	Mat hsv, out1;
	cvtColor(img, hsv, COLOR_BGR2HSV);//转成hsv方便扣红色
	//查表得知红色颜色空间有两组
	inRange(hsv, Scalar(156, 43, 46), Scalar(180, 255, 255), out1);//提取到三角形 
	//inRange(hsv, Scalar(0, 43, 46), Scalar(10, 255, 255), out2);//提取到x

	Mat dst;//out结果不理想 尝试使用开运算消除噪点
	Mat k = getStructuringElement(1, Size(3, 3));
	morphologyEx(out1, dst, 2, k, Point(-1, -1), 5);

	vector<Point2f> points;
	findNonZero(dst, points);//提取所有非0的点

	vector<Point2f> triangle;//对离散点拟合三角形
	double area = minEnclosingTriangle(points, triangle);

	Mat ans(img.size(), img.type(),Scalar(0,0,0));

	//绘制三角形
	for (int i = 0; i < 3; i++) {
		if (i == 2) {
			line(ans, triangle[i], triangle[0], Scalar(255, 200, 100), 1, 16);
			break;
		}
		line(ans, triangle[i], triangle[i + 1], Scalar(255, 200, 100), 1, 16);
	}
	waitKey(0);
}

void Base::Q1() {
	Mat img=imread("./images/q1.png");

	Mat hsv, out1;
	cvtColor(img, hsv, COLOR_BGR2HSV);//转成hsv方便扣红色
	//查表得知红色颜色空间有两组
	inRange(hsv, Scalar(156, 43, 46), Scalar(180, 255, 255), out1);//提取到三角形 
	//inRange(hsv, Scalar(0, 43, 46), Scalar(10, 255, 255), out2);//提取到x

	Mat dst;//out结果不理想 尝试使用开运算消除噪点
	Mat k = getStructuringElement(1, Size(3, 3));
	morphologyEx(out1, dst, 2, k,Point(-1,-1),5);

	//闭运算填充空洞 连接临近区域
	morphologyEx(dst, dst, 3, k, Point(-1, -1),20);

	Mat canny;
	Canny(dst, canny, 80, 160, 3, false);

	cvtColor(canny, canny, COLOR_GRAY2BGR);//转成3通道图像方便上色

	Scalar lightBlue(255, 200, 100);//定义浅蓝色

	for (int i = 0; i < canny.rows; i++) {//对每个通道上色 
		for (int j = 0; j < canny.cols; j++) {
			if (canny.at<Vec3b>(i, j)[0]!= 0) {
				canny.at<Vec3b>(i, j)[0] = lightBlue[0];
				canny.at<Vec3b>(i, j)[1] = lightBlue[1];
				canny.at<Vec3b>(i, j)[2] = lightBlue[2];
			}
		}
	}

	resize(img, img, Size(1280, 720));
	resize(canny, canny, Size(1280, 720));
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	imshow("原图", img);
	imshow("黑白图像", gray);
	imshow("canny", canny);
	imwrite("./images/canny.png",canny);
	waitKey(0);
}

Mat frame;//用全局变量存储

// 回调函数，用于处理bar滑动事件
void onExposureChange(int exposure, void* userData) {
	VideoCapture* cap = (VideoCapture*)userData;//void转VideoCapture
	double newExposure = exposure / 100.0;  // 将bar值映射到[0, 1]的范围
	cap->set(CAP_PROP_EXPOSURE, newExposure);//设置曝光度
}

void  Base::Q2() {
	VideoCapture capture(0);

	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);//获取视频总帧率
	cout << "fps" << fps << " " << "宽" << width << " " << "高" << height << " " << "总帧率" << numFrame << endl;

	VideoWriter writer("./images/output1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
	//					保存路径		        四字符编码格式                                    true表示输出的视频文件是彩色的
	
	if (!capture.isOpened()) {  // 检查摄像头是否正确打开
		cout << "Could not open the camera." << endl;
		return;
	}

	if (!writer.isOpened()) {
		cout << "error" << endl;
		return;
	}
	
	namedWindow("显示框", WINDOW_AUTOSIZE);

	int light = 50;//初始值
	//createTrackbar("亮度", "显示框", &light, 255, callBack, 0);//在显示框窗口创建亮度bar
	createTrackbar("亮度", "显示框", &light, 100);//不使用回调函数 最大值设置为100，最多放大两倍亮度
	
	int initialExposure = 50;  // 初始曝光度值
	createTrackbar("曝光", "显示框", &initialExposure, 100, onExposureChange, &capture);//使用回调函数 //有bug？ 摄像头问题？？？
	
	while (true)
	{
		if (!capture.read(frame)) {
			break;
		}
		
		frame.convertTo(frame, -1,light/50.0, 0);//50/50=1初始不对亮度进行调整 
		imshow("显示框", frame);
		writer.write(frame);//对每一帧进行保存

		char c = waitKey(5);
		if (c == 'q') {
			break;
		}
	}

	capture.release();//释放资源
	writer.release();

	//对标定板进行角点检测
	Mat picture = imread("./images/bd.png");
	Size board_size = Size(10, 7);//标定板内角点数目（行 列）
	Mat gray1;
	cvtColor(picture, gray1, COLOR_BGR2GRAY);
	vector<Point2f> img1_points;
	findChessboardCorners(gray1, board_size, img1_points);//检测图像中棋盘格模式的角点
	find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));//对初始的角点坐标进行亚像素级别的优化
	bool pattern = true;
	drawChessboardCorners(picture, board_size, img1_points, pattern);//绘制检测到的棋盘格角点
	imshow("标定板角点检测", picture);
	waitKey(0);
}

Point sp(-1, -1);//开始
Point ep(-1, -1);//结束
Mat temp;

//int event：表示当前的鼠标事件类型，可以是 EVENT_MOUSEMOVE（移动）,EVENT_LBUTTONDOWN（左键按下）,EVENT_LBUTTONUP(抬起)
//int x 和 int y：表示当前鼠标事件发生的坐标位置。
//int flags：表示当前的鼠标事件的附加标志信息，例如鼠标按键状态等。
//void* param：是 setMouseCallback 函数的最后一个参数，可以用于传递额外的参数给回调函数。
static void on_draw(int event, int x, int y, int flags, void* userdata) {
	Mat image = *((Mat*)userdata);//转mat类型指针 解引用
	if (event == EVENT_LBUTTONDOWN) {//按下记录当前位置坐标
		sp.x = x;
		sp.y = y;
	}
	else if (event == EVENT_LBUTTONUP) {
		ep.x = x;//抬起记录最后坐标
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;

		if (dx > 0 && dy > 0) {
			Rect box(sp.x, sp.y, dx, dy);//构建从始到末rect

			temp.copyTo(image);
			Mat ROI = image(box);

			imshow("ROI区域", ROI);//显示 image 图像中的 box 矩形区域
			imwrite("./images/smallmimi.png", ROI);

			rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);//在原图绘制矩阵
			imshow("鼠标绘制", image);
			
			cout << "中心坐标" << ROI.rows / 2 << " " << ROI.cols / 2 << endl;

			// 
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;//宽
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);//不在原图操作
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);

				//读取当前像素信息
				Vec3b rgb = image.at<Vec3b>(y, x);
				int r = static_cast <int> (rgb[2]);
				int g = static_cast <int> (rgb[1]);
				int b = static_cast <int> (rgb[0]);

				String s = "rgb: " + to_string(r) + "," + to_string(g) + "," + to_string(b) + " " + "position: " + to_string(y) + "," + to_string(x);
				putText(image, s, Point(x-dx, y-dy), 0, 0.5, Scalar(255,0,0), 1, 8, false);
				imshow("鼠标绘制", image);
			}
		}
	}
}

void Base::Q3() {
	Mat cat = imread("./images/mimi.png");
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&cat));

	imshow("鼠标绘制", cat);
	temp = cat.clone();
	waitKey(0);
}

/***************************************************APP***************************************************************/
void App::Q1() {
	//思路：
	//一：苹果是红色 对r通道设置阈值做一个筛选  对筛选的结果进行寻找轮廓 在原图中绘制轮廓（效果一般 原因：1.苹果上有阴影 2.苹果的底部色彩不够鲜艳 3.天空区域的r通道数值比较大）
	
	//二：转到hsv对红色进行抠图  相较于上一个方法 可以直接排除天空的干扰 但是苹果底部依旧不能很好识别

	Mat image=imread("./images/apple.png");

	//idea1
	vector<Mat> channels;
	split(image, channels);//分离
	
	Mat r,binary;
	r = channels[2];
	GaussianBlur(r, r, Size(13, 13), 4, 4);
	threshold(r, binary,240 ,255, THRESH_BINARY);//二值化 小于阈值的置0 大于的设置255
	//adaptiveThreshold(r, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 0);自适应二值化

	Mat k = getStructuringElement(1, Size(3, 3));
	morphologyEx(binary, binary, 2, k, Point(-1, -1), 5);//开运算 

	morphologyEx(binary, binary, 1, k, Point(-1, -1), 30);//膨胀

	vector<vector<Point>> contours;//存储检测到的轮廓的向量容器  可能会有多个
	vector<Vec4i> hierarchy;//存储轮廓的层次结构信息
	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());//第二个参数为检测到的轮廓 

	for (int t = 0; t < contours.size(); t++) {
		Rect rect = boundingRect(contours[t]);//获取最大外接矩形
		rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
		//drawContours(image, contours, t, Scalar(255, 0,0 ), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
		imshow("原图", image);
	}

	//idea2
	//Mat hsv,out;
	//cvtColor(image, hsv, COLOR_BGR2HSV);
	//inRange(hsv, Scalar(156, 43, 46), Scalar(180, 255, 255), out);
	
	waitKey(0);
}

/***********************去雾相关函数***********************/
//导向滤波，用来优化t(x)，针对单通道
Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps)
{
	//转换源图像信息
	srcImage.convertTo(srcImage, CV_32FC1, 1 / 255.0);
	srcClone.convertTo(srcClone, CV_32FC1);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	Mat boxResult;
	//步骤一：计算均值 //应用盒滤波器计算相关的值
	boxFilter(Mat::ones(nRows, nCols, srcImage.type()),boxResult, CV_32FC1, Size(r, r));

	//生成导向均值mean_I
	Mat mean_I;
	boxFilter(srcImage, mean_I, CV_32FC1, Size(r, r));

	//生成原始均值mean_p
	Mat mean_p;
	boxFilter(srcClone, mean_p, CV_32FC1, Size(r, r));

	//生成互相关均值mean_Ip
	Mat mean_Ip;
	boxFilter(srcImage.mul(srcClone), mean_Ip,CV_32FC1, Size(r, r));

	//生成自相关均值mean_II
	Mat mean_II;
	boxFilter(srcImage.mul(srcImage), mean_II,CV_32FC1, Size(r, r));

	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//步骤二：计算相关系数
	Mat var_I = mean_II - mean_I.mul(mean_I);
	Mat var_Ip = mean_Ip - mean_I.mul(mean_p);

	//步骤三：计算参数系数a,b
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);

	//步骤四：计算系数a\b的均值
	Mat mean_a;
	boxFilter(a, mean_a, CV_32FC1, Size(r, r));
	mean_a = mean_a / boxResult;
	Mat mean_b;
	boxFilter(b, mean_b, CV_32FC1, Size(r, r));
	mean_b = mean_b / boxResult;

	//步骤五：生成输出矩阵
	Mat resultMat = mean_a.mul(srcImage) + mean_b;
	return resultMat;
}

//计算暗通道图像矩阵，针对三通道彩色图像
Mat dark_channel(Mat src)//对每个通道取最小值，再最小值滤波
{
	int border = 7;//扩展边界
	std::vector<cv::Mat> rgbChannels(3);
	Mat min_mat(src.size(), CV_8UC1, Scalar(0)), min_mat_expansion;

	split(src, rgbChannels);//分离通道并求每个通道最小值存到min_mat中
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int min_val = 0;
			int val_1, val_2, val_3;
			val_1 = rgbChannels[0].at<uchar>(i, j);
			val_2 = rgbChannels[1].at<uchar>(i, j);
			val_3 = rgbChannels[2].at<uchar>(i, j);

			min_val = std::min(val_1, val_2);
			min_val = std::min(min_val, val_3);
			min_mat.at<uchar>(i, j) = min_val;
		}
	}

	//对图像进行边界扩展操作 因为还要对上述的图像进行最小值滤波
	copyMakeBorder(min_mat, min_mat_expansion, border, border, border, border, BORDER_REPLICATE);

	Mat dark_channel_mat(src.size(), CV_8UC1, Scalar(0));
	for (int m = border; m < min_mat_expansion.rows - border; m++)//遍历非扩展边界
	{
		for (int n = border; n < min_mat_expansion.cols - border; n++)
		{
			Mat imageROI;
			int min_num = 256;
			imageROI = min_mat_expansion(Rect(n - border, m - border, 2 * border + 1, 2 * border + 1));//对每个非扩展边界的像素点最小滤波 大小为 2 * border + 1
			for (int i = 0; i < imageROI.rows; i++)
			{
				for (int j = 0; j < imageROI.cols; j++)
				{
					int val_roi = imageROI.at<uchar>(i, j);
					min_num = std::min(min_num, val_roi);
				}
			}
			dark_channel_mat.at<uchar>(m - border, n - border) = min_num;//将最小值存在dark_channel_mat
		}
	}
	return dark_channel_mat;
}

int calculate_A(Mat src, Mat dark_channel_mat)//计算A 从暗通道中选取最亮区域 对应回原图像中的区域
{
	map<int, Point> pair_data;//存（值，点）对  以及对val进行排序！

	//cout << dark_channel_mat.rows << " " << dark_channel_mat.cols << endl;
	for (int i = 0; i < dark_channel_mat.rows; i++)//遍历按通道图像，构建（值，点）对  将点和对应的数值对应起来
	{
		for (int j = 0; j < dark_channel_mat.cols; j++)
		{
			int val = dark_channel_mat.at<uchar>(i, j);
			Point pt;
			pt.x = j;
			pt.y = i;
			pair_data.insert(make_pair(val, pt));
			//cord.push_back(pt);//对下面for进行优化 错误的；因为存入map的作用是对val进行排序
		}
	}

	vector<Point> cord;//存暗通道的点
	map<int, Point>::iterator iter;//源代码冗余；无误
	for (iter = pair_data.begin(); iter != pair_data.end(); iter++)//用迭代器的方式遍历pair_data 将排序后的点放入cord
	{
		//cout << iter->first << endl;
		cord.push_back(iter->second);
	}

	std::vector<cv::Mat> rgbChannels(3);
	split(src, rgbChannels);
	int max_val = 0;

	////开源代码有错？  原理应该是寻找暗通道中值最大的点对应的原图像的点  
	//for (int m = 0; m < cord.size(); m++)// 这个循环是在寻找原图像中最大值的点？
	//{
	//	Point tmp = cord[m];
	//	int val_1, val_2, val_3;
	//	val_1 = rgbChannels[0].at<uchar>(tmp.y, tmp.x);//
	//	val_2 = rgbChannels[1].at<uchar>(tmp.y, tmp.x);
	//	val_3 = rgbChannels[2].at<uchar>(tmp.y, tmp.x);

	//	max_val = std::max(val_1, val_2);
	//	max_val = std::max(max_val, val_3);
	//}

	//修改开源代码
	Point Max = cord[cord.size() - 1];//暗通道最大值的点
	int val_1, val_2, val_3;
	val_1 = rgbChannels[0].at<uchar>(Max.y, Max.x);//对应原来图像的点
	val_2 = rgbChannels[1].at<uchar>(Max.y, Max.x);
	val_3 = rgbChannels[2].at<uchar>(Max.y, Max.x);
	max_val = max(val_1, max(val_2, val_3));//选取三个通道的最大值返回

	return max_val;
}

//Mat calculate_tx(Mat& src, int A, Mat& dark_channel_mat)//计算t   ？？？
Mat calculate_tx(int A, Mat& dark_channel_mat)//雾气图像I不用闯入？  
{
	Mat dst;
	Mat tx;
	float dark_channel_num;
	dark_channel_num = A / 255.0;//对A归一化

	dark_channel_mat.convertTo(dst, CV_32FC3, 1 / 255.0);//单通道暗通道图像扩展到3通道  每个通道数值相等
	dst = dst / dark_channel_num;

	tx = 1 - 0.95 * dst;

	return tx;
}

//Mat calculate_tx(Mat& src, int A)//计算t 
//{
//	Mat tem = src / A;//  原图像/A
//	Mat dark=dark_channel(tem);//计算暗通道
//
//	//cout << 1 - dark << endl;
//	return 1-dark;
//}


Mat haze_removal_img(Mat& src, int A, Mat& tx)//由雾气图像模型 反解J
{
	Mat result_img(src.rows, src.cols, CV_8UC3);
	vector<Mat> srcChannels(3), resChannels(3);
	split(src, srcChannels);//对雾气图像切分
	split(result_img, resChannels);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			for (int m = 0; m < 3; m++)//遍历每个像素点的每个通道
			{
				int value_num = srcChannels[m].at<uchar>(i, j);//每个通道的值
				float max_t = tx.at<float>(i, j);//对应的折射率
				if (max_t < 0.1) //防止折射率为0作为除数
				{
					max_t = 0.1;
				}
				resChannels[m].at<uchar>(i, j) = (value_num - A) / max_t + A;
			}
		}
	}

	merge(resChannels, result_img);

	return result_img;
}

void App::Q2() {//去雾
	Mat src = imread("./images/haze.png");
	Mat dst;
	cvtColor(src, dst, COLOR_BGR2GRAY);
	Mat dark_channel_mat = dark_channel(src);//计算暗通道图像

	int A = calculate_A(src, dark_channel_mat);//知道I，暗通道图像 可以计算A

	Mat tx = calculate_tx(A, dark_channel_mat);//知道I，暗通道图像，A可以计算t
	//Mat tx = calculate_tx(src, A);

	Mat tx_ = guidedfilter(dst, tx, 30, 0.001);//导向滤波后的tx，dst为引导图像  起优化作用

	Mat haze_removal_image;
	haze_removal_image = haze_removal_img(src, A, tx_);//根据I，t，A反解J
	namedWindow("去雾后的图像");
	namedWindow("原始图像");
	imshow("原始图像", src);
	imshow("去雾后的图像", haze_removal_image);
	waitKey(0);
}

void App::Q3() {
	//思路：使用undistort可以对畸变进行矫正，需要知道cameraMatrix，和distCoeffs  
	//使用calibrateCamera可以求取内参和畸变矩阵，需要知道真实世界的点objectPoints，标定板角点imgsPoints，标定图像的大小imageSize
	//使用findChessboardCorners可以检测标定板角点， objectPoints应该如何测量？

	//用本机拍取的标定板照片求取的是本机的内参和畸变矩阵，对考核任务发布的图片是否能够有效矫正？
	//Mat img = imread("./images/ji.png");
	//Mat dst;
	//Mat cameraMatrix, distCoeffs;//内参和畸变
	//undistort(img, dst, cameraMatrix, distCoeffs);   //pass

	//

}

void Game::Q1() {
	Mat img = imread("./images/zjb.png");
	Mat gray,binary;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	//Mat k = getStructuringElement(1, Size(3, 3));
	//morphologyEx(gray, gray, 2, k,Point(-1,-1),3);//将一些细小边缘腐蚀

	GaussianBlur(gray, gray, Size(5, 5), 0, 0);
	threshold(gray, binary, 50, 255, THRESH_BINARY); 

	vector<vector<Point>> contours;//轮廓 存储检测到的轮廓的向量容器 存储检测到的轮廓的向量容器
	vector<Vec4i> hierarchy;//存储轮廓的层次结构信息
	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());


	//轮廓绘制
	for (int t = 0; t < contours.size(); t++) {
		drawContours(img, contours, t, Scalar(0, 0, 255), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
		imshow("111", img);
		//waitKey(0);
	}

	waitKey(0);
}




//#include "stdio.h"
//#include<iostream> 
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//using namespace std;
//using namespace cv;
//
//const int kThreashold = 180;
//const int kMaxVal = 255;
//const Size kGaussianBlueSize = Size(5, 5);
//const int color = 2;//用于选择判断装甲板的颜色 0红色   2蓝色
//
//int main()
//{
//	//存在问题：
//	//1.装甲板在边缘，只能识别到一个光条
//	//2.蓝色装甲板有时候识别不出轮廓？
//
//	VideoCapture video;
//	video.open("./images/zjbb.mp4");
//	Mat frame;
//	Mat channels[3], binary, Gaussian;
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierarchy;
//
//	Rect boundRect;//存储最小外接矩形
//	while (true)
//	{
//		Rect point_array[50];//存储合格的外接矩阵
//		if (!video.read(frame)) {
//			break;
//		}
//		split(frame, channels);
//
//		threshold(channels[2], binary, kThreashold, kMaxVal, 0);//对r/b进行二值化
//
//		GaussianBlur(binary, Gaussian, kGaussianBlueSize, 0);
//		Mat k = getStructuringElement(1, Size(3, 3));
//		findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//查找轮廓
//
//		int index = 0;
//		for (int i = 0; i < contours.size(); i++) {
//			//box = minAreaRect(Mat(contours[i]));//计算旋转矩阵
//			//box.points(boxPts.data());
//			boundRect = boundingRect(Mat(contours[i]));//计算最小外接矩形
//			rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//左上 右下
//			try
//			{
//				if (double(boundRect.height / boundRect.width) >= 2 && boundRect.height > 36 && boundRect.height > 20) {//满足条件的矩阵保存在point_array
//					point_array[index] = boundRect;
//					index++;
//				}
//			}
//			catch (const char* msg)
//			{
//				cout << printf(msg) << endl;
//				continue;
//			}
//		}
//
//		if (index < 2) {//没检测到合格的轮廓 直接 下一帧
//			imshow("video", frame);
//			cv::waitKey(10);
//			continue;
//		}
//
//		int point_near[2];
//		int min = 10000;
//		for (int i = 0; i < index - 1; i++)//找到面积之差最小的的两个轮廓的索引 
//		{
//			for (int j = i + 1; j < index; j++) {
//				int value = abs(point_array[i].area() - point_array[j].area());
//				if (value < min)
//				{
//					min = value;
//					point_near[0] = i;
//					point_near[1] = j;
//				}
//			}
//		}
//
//		try
//		{
//			Rect rectangle_1 = point_array[point_near[0]];//找到这两个相似的轮廓
//			Rect rectangle_2 = point_array[point_near[1]];
//
//			if (rectangle_2.x == 0 || rectangle_1.x == 0) {
//				throw "not enough points";
//			}
//
//			Point point1 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y);
//			Point point2 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y + rectangle_1.height);
//			Point point3 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y);
//			Point point4 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y + rectangle_2.height);
//			Point p[4] = { point1,point2,point4,point3 };
//
//			cout << p[0] << p[1] << p[2] << p[3] << endl;
//			for (int i = 0; i < 4; i++) {
//				line(frame, p[i % 4], p[(i + 1) % 4], Scalar(255, 255, 255), 2);
//			}
//		}
//		catch (const char* msg)
//		{
//			cout << msg << endl;
//		}
//
//		imshow("video", frame);
//		cv::waitKey(20);
//	}
//	cv::destroyAllWindows();
//	return 0;
//}