#include <opencv2/opencv.hpp>
#include <iostream>
#include <Base.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<algorithm>
#include<set>
#include<map>

using namespace std;
using namespace cv;

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


















///***************************************************APP***************************************************************/
//void App::Q1() {
//	//思路：
//	//一：苹果是红色 对r通道设置阈值做一个筛选  对筛选的结果进行寻找轮廓 在原图中绘制轮廓（效果一般 原因：1.苹果上有阴影 2.苹果的底部色彩不够鲜艳 3.天空区域的r通道数值比较大）
//	
//	//二：转到hsv对红色进行抠图  相较于上一个方法 可以直接排除天空的干扰 但是苹果底部依旧不能很好识别
//
//	Mat image=imread("./images/apple.png");
//
//	//idea1
//	vector<Mat> channels;
//	split(image, channels);//分离
//	
//	Mat r,binary;
//	r = channels[2];
//	GaussianBlur(r, r, Size(13, 13), 4, 4);
//	threshold(r, binary,240 ,255, THRESH_BINARY);//二值化 小于阈值的置0 大于的设置255
//	//adaptiveThreshold(r, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 0);自适应二值化
//
//	Mat k = getStructuringElement(1, Size(3, 3));
//	morphologyEx(binary, binary, 2, k, Point(-1, -1), 5);//开运算 
//
//	morphologyEx(binary, binary, 1, k, Point(-1, -1), 30);//膨胀
//
//	vector<vector<Point>> contours;//存储检测到的轮廓的向量容器  可能会有多个
//	vector<Vec4i> hierarchy;//存储轮廓的层次结构信息
//	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());//第二个参数为检测到的轮廓 
//
//	for (int t = 0; t < contours.size(); t++) {
//		Rect rect = boundingRect(contours[t]);//获取最大外接矩形
//		rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
//		//drawContours(image, contours, t, Scalar(255, 0,0 ), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
//		imshow("原图", image);
//	}
//
//	//idea2
//	//Mat hsv,out;
//	//cvtColor(image, hsv, COLOR_BGR2HSV);
//	//inRange(hsv, Scalar(156, 43, 46), Scalar(180, 255, 255), out);
//	
//	waitKey(0);
//}
//
///***********************去雾相关函数***********************/
////导向滤波，用来优化t(x)，针对单通道
//Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps)
//{
//	//转换源图像信息
//	srcImage.convertTo(srcImage, CV_32FC1, 1 / 255.0);
//	srcClone.convertTo(srcClone, CV_32FC1);
//	int nRows = srcImage.rows;
//	int nCols = srcImage.cols;
//	Mat boxResult;
//	//步骤一：计算均值 //应用盒滤波器计算相关的值
//	boxFilter(Mat::ones(nRows, nCols, srcImage.type()),boxResult, CV_32FC1, Size(r, r));
//
//	//生成导向均值mean_I
//	Mat mean_I;
//	boxFilter(srcImage, mean_I, CV_32FC1, Size(r, r));
//
//	//生成原始均值mean_p
//	Mat mean_p;
//	boxFilter(srcClone, mean_p, CV_32FC1, Size(r, r));
//
//	//生成互相关均值mean_Ip
//	Mat mean_Ip;
//	boxFilter(srcImage.mul(srcClone), mean_Ip,CV_32FC1, Size(r, r));
//
//	//生成自相关均值mean_II
//	Mat mean_II;
//	boxFilter(srcImage.mul(srcImage), mean_II,CV_32FC1, Size(r, r));
//
//	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
//
//	//步骤二：计算相关系数
//	Mat var_I = mean_II - mean_I.mul(mean_I);
//	Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
//
//	//步骤三：计算参数系数a,b
//	Mat a = cov_Ip / (var_I + eps);
//	Mat b = mean_p - a.mul(mean_I);
//
//	//步骤四：计算系数a\b的均值
//	Mat mean_a;
//	boxFilter(a, mean_a, CV_32FC1, Size(r, r));
//	mean_a = mean_a / boxResult;
//	Mat mean_b;
//	boxFilter(b, mean_b, CV_32FC1, Size(r, r));
//	mean_b = mean_b / boxResult;
//
//	//步骤五：生成输出矩阵
//	Mat resultMat = mean_a.mul(srcImage) + mean_b;
//	return resultMat;
//}
//
////计算暗通道图像矩阵，针对三通道彩色图像
//Mat dark_channel(Mat src)//对每个通道取最小值，再最小值滤波
//{
//	int border = 7;//扩展边界
//	std::vector<cv::Mat> rgbChannels(3);
//	Mat min_mat(src.size(), CV_8UC1, Scalar(0)), min_mat_expansion;
//
//	split(src, rgbChannels);//分离通道并求每个通道最小值存到min_mat中
//	for (int i = 0; i < src.rows; i++)
//	{
//		for (int j = 0; j < src.cols; j++)
//		{
//			int min_val = 0;
//			int val_1, val_2, val_3;
//			val_1 = rgbChannels[0].at<uchar>(i, j);
//			val_2 = rgbChannels[1].at<uchar>(i, j);
//			val_3 = rgbChannels[2].at<uchar>(i, j);
//
//			min_val = std::min(val_1, val_2);
//			min_val = std::min(min_val, val_3);
//			min_mat.at<uchar>(i, j) = min_val;
//		}
//	}
//
//	//对图像进行边界扩展操作 因为还要对上述的图像进行最小值滤波
//	copyMakeBorder(min_mat, min_mat_expansion, border, border, border, border, BORDER_REPLICATE);
//
//	Mat dark_channel_mat(src.size(), CV_8UC1, Scalar(0));
//	for (int m = border; m < min_mat_expansion.rows - border; m++)//遍历非扩展边界
//	{
//		for (int n = border; n < min_mat_expansion.cols - border; n++)
//		{
//			Mat imageROI;
//			int min_num = 256;
//			imageROI = min_mat_expansion(Rect(n - border, m - border, 2 * border + 1, 2 * border + 1));//对每个非扩展边界的像素点最小滤波 大小为 2 * border + 1
//			for (int i = 0; i < imageROI.rows; i++)
//			{
//				for (int j = 0; j < imageROI.cols; j++)
//				{
//					int val_roi = imageROI.at<uchar>(i, j);
//					min_num = std::min(min_num, val_roi);
//				}
//			}
//			dark_channel_mat.at<uchar>(m - border, n - border) = min_num;//将最小值存在dark_channel_mat
//		}
//	}
//	return dark_channel_mat;
//}
//
//int calculate_A(Mat src, Mat dark_channel_mat)//计算A 从暗通道中选取最亮区域 对应回原图像中的区域
//{
//	map<int, Point> pair_data;//存（值，点）对  以及对val进行排序！
//
//	//cout << dark_channel_mat.rows << " " << dark_channel_mat.cols << endl;
//	for (int i = 0; i < dark_channel_mat.rows; i++)//遍历按通道图像，构建（值，点）对  将点和对应的数值对应起来
//	{
//		for (int j = 0; j < dark_channel_mat.cols; j++)
//		{
//			int val = dark_channel_mat.at<uchar>(i, j);
//			Point pt;
//			pt.x = j;
//			pt.y = i;
//			pair_data.insert(make_pair(val, pt));
//			//cord.push_back(pt);//对下面for进行优化 错误的；因为存入map的作用是对val进行排序
//		}
//	}
//
//	vector<Point> cord;//存暗通道的点
//	map<int, Point>::iterator iter;//源代码冗余；无误
//	for (iter = pair_data.begin(); iter != pair_data.end(); iter++)//用迭代器的方式遍历pair_data 将排序后的点放入cord
//	{
//		//cout << iter->first << endl;
//		cord.push_back(iter->second);
//	}
//
//	std::vector<cv::Mat> rgbChannels(3);
//	split(src, rgbChannels);
//	int max_val = 0;
//
//	////开源代码有错？  原理应该是寻找暗通道中值最大的点对应的原图像的点  
//	//for (int m = 0; m < cord.size(); m++)// 这个循环是在寻找原图像中最大值的点？
//	//{
//	//	Point tmp = cord[m];
//	//	int val_1, val_2, val_3;
//	//	val_1 = rgbChannels[0].at<uchar>(tmp.y, tmp.x);//
//	//	val_2 = rgbChannels[1].at<uchar>(tmp.y, tmp.x);
//	//	val_3 = rgbChannels[2].at<uchar>(tmp.y, tmp.x);
//
//	//	max_val = std::max(val_1, val_2);
//	//	max_val = std::max(max_val, val_3);
//	//}
//
//	//修改开源代码
//	Point Max = cord[cord.size() - 1];//暗通道最大值的点
//	int val_1, val_2, val_3;
//	val_1 = rgbChannels[0].at<uchar>(Max.y, Max.x);//对应原来图像的点
//	val_2 = rgbChannels[1].at<uchar>(Max.y, Max.x);
//	val_3 = rgbChannels[2].at<uchar>(Max.y, Max.x);
//	max_val = max(val_1, max(val_2, val_3));//选取三个通道的最大值返回
//
//	return max_val;
//}
//
////Mat calculate_tx(Mat& src, int A, Mat& dark_channel_mat)//计算t   ？？？
//Mat calculate_tx(int A, Mat& dark_channel_mat)//雾气图像I不用闯入？  
//{
//	Mat dst;
//	Mat tx;
//	float dark_channel_num;
//	dark_channel_num = A / 255.0;//对A归一化
//
//	dark_channel_mat.convertTo(dst, CV_32FC3, 1 / 255.0);//单通道暗通道图像扩展到3通道  每个通道数值相等
//	dst = dst / dark_channel_num;
//
//	tx = 1 - 0.95 * dst;
//
//	return tx;
//}
//
////Mat calculate_tx(Mat& src, int A)//计算t 
////{
////	Mat tem = src / A;//  原图像/A
////	Mat dark=dark_channel(tem);//计算暗通道
////
////	//cout << 1 - dark << endl;
////	return 1-dark;
////}
//
//
//Mat haze_removal_img(Mat& src, int A, Mat& tx)//由雾气图像模型 反解J
//{
//	Mat result_img(src.rows, src.cols, CV_8UC3);
//	vector<Mat> srcChannels(3), resChannels(3);
//	split(src, srcChannels);//对雾气图像切分
//	split(result_img, resChannels);
//
//	for (int i = 0; i < src.rows; i++)
//	{
//		for (int j = 0; j < src.cols; j++)
//		{
//			for (int m = 0; m < 3; m++)//遍历每个像素点的每个通道
//			{
//				int value_num = srcChannels[m].at<uchar>(i, j);//每个通道的值
//				float max_t = tx.at<float>(i, j);//对应的折射率
//				if (max_t < 0.1) //防止折射率为0作为除数
//				{
//					max_t = 0.1;
//				}
//				resChannels[m].at<uchar>(i, j) = (value_num - A) / max_t + A;
//			}
//		}
//	}
//
//	merge(resChannels, result_img);
//
//	return result_img;
//}
//
//void App::Q2() {//去雾
//	Mat src = imread("./images/haze.png");
//	Mat dst;
//	cvtColor(src, dst, COLOR_BGR2GRAY);
//	Mat dark_channel_mat = dark_channel(src);//计算暗通道图像
//
//	int A = calculate_A(src, dark_channel_mat);//知道I，暗通道图像 可以计算A
//
//	Mat tx = calculate_tx(A, dark_channel_mat);//知道I，暗通道图像，A可以计算t
//	//Mat tx = calculate_tx(src, A);
//
//	Mat tx_ = guidedfilter(dst, tx, 30, 0.001);//导向滤波后的tx，dst为引导图像  起优化作用
//
//	Mat haze_removal_image;
//	haze_removal_image = haze_removal_img(src, A, tx_);//根据I，t，A反解J
//	namedWindow("去雾后的图像");
//	namedWindow("原始图像");
//	imshow("原始图像", src);
//	imshow("去雾后的图像", haze_removal_image);
//	waitKey(0);
//}
//
//void App::Q3() {
//	//思路：使用undistort可以对畸变进行矫正，需要知道cameraMatrix，和distCoeffs  
//	//使用calibrateCamera可以求取内参和畸变矩阵，需要知道真实世界的点objectPoints，标定板角点imgsPoints，标定图像的大小imageSize
//	//使用findChessboardCorners可以检测标定板角点， objectPoints应该如何测量？
//
//	//用本机拍取的标定板照片求取的是本机的内参和畸变矩阵，对考核任务发布的图片是否能够有效矫正？
//	//Mat img = imread("./images/ji.png");
//	//Mat dst;
//	//Mat cameraMatrix, distCoeffs;//内参和畸变
//	//undistort(img, dst, cameraMatrix, distCoeffs);   //pass
//
//	//
//
//}







/****************GAME*******************/
////const int kThreashold = 180;//阈值
////const int kMaxVal = 255;
////const Size kGaussianBlueSize = Size(5, 5);
//
//void Game::Q1p() {//
//	//如果只是通过对轮廓的面积大小对背景光条进行筛选 在比赛中会根据近大远小原则 在靠近装甲板光条的时候是否不适应？
//	//如果是通过对轮廓的长宽比例来筛选的话，小光条和大光条的轮廓比例似乎差不多
//	//如果是通过色彩的阈值来筛选，大小光条的亮度值很接近，无法实现  //pass1
//
//
//	//先进行初筛选 将长宽比小于2 大于10的先 筛选掉 再将剩余的进行就近组合成矩形 再对矩形的长宽比进行筛选 
//	//如何确定两两匹配的光条？  通过矩阵的中心点距离来判断？（距离在比赛中不适用吧？ 近大远小 不可能在某一点固定不动吧？）
//	//面积大小相似？ 不行
//	//中心点的纵坐标？ 能解决匹配到不在一个层面的光条矩形的问题 但是不能解决在同一层面的矩形匹配问题  面积差不多 且角度差不多
//	//查找资料得知可以用旋转矩阵的角度来匹配
//
//	Mat frame = imread("./images/zj.png");
//	Mat channels[3], binary, Gaussian;
//
//	vector<vector<Point>> contours;//轮廓点
//	vector<Vec4i> hierarchy;
//	RotatedRect AreaRect;//存储最小(旋转)外接矩形
//
//	split(frame, channels);
//	//Mat colorimage = channels[0] - channels[2];
//
//	threshold(channels[0], binary, 180, 255, 0);//对b进行二值化
//	GaussianBlur(binary, Gaussian, Size(3,3), 0);//size越小越好
//	findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
//
//	RotatedRect point_array[20];//存筛选后的旋转矩阵
//	Point2f points[4];//存RotatedRect转换的四点坐标
//
//	int index = 0;
//	for (int i = 0; i < contours.size(); i++) {
//		//boundRect = boundingRect(Mat(contours[i]));//计算最小外接矩形
//		AreaRect = minAreaRect(Mat(contours[i]));//计算最小的旋转矩阵 因为要使用angle
//
//		double width = AreaRect.size.width;//宽
//		double heigth = AreaRect.size.height;//高
//		double area = AreaRect.size.area();//面积
//
//		//判断有问题
//		if (area>100) { //先把一些太小的干扰轮廓筛选删除
//			point_array[index] = AreaRect;
//			AreaRect.points(points);
//
//			//rectangle(frame, point_array[index].tl(), point_array[index].br(), Scalar(255, 255, 255), 2, 8, 0);//左上 右下
//			for (int i = 0; i < 4; i++) {
//				line(frame, points[i % 4], points[(i + 1) % 4], Scalar(255, 255, 255), 2);
//			}
//			index++;
//		}
//	}
//
//	
//
//	imshow("frame", frame);
//	waitKey(0);
//
//	//for (int t = 0; t < contours.size(); t++) {
//	//	drawContours(frame, contours, t, Scalar(255, 255, 255), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
//	//	imshow("frame", frame);
//	//	waitKey(0);
//	//}
//}
//
//const bool color = true;//用于选择判断装甲板的颜色 flase红色   true蓝色
//void Game::Q1v() {
//	//思路：分别对红蓝装甲板的r和b通道进行分离并且二值化，然后寻找轮廓
//	//对轮廓进行筛选 绘制出两个灯条矩形所围城的面积 以及 中心点
//
//	//存在问题：
//   //1.只检测到一个光条的时候，追踪会断开  解决：用前几帧作为参考
//   //2.蓝色装甲板有时候识别不出轮廓？ 原因：检测出的部分轮廓高宽比为1.x 在筛选的时候没有使用double类型，将比例1.x的轮廓直接归类为1，导致需要的轮廓没有筛选出来  解决：计算比例时*1.0
//
//	VideoCapture video;
//	String s;
//	int t;
//	if (color == true) {
//		s = "./images/zjbb.mp4";
//		t = 0;
//	}
//	else {
//		s = "./images/zjbr.mp4";
//		t = 2;
//	}
//
//	video.open(s);
//	Mat frame;
//	video >> frame;
//	int fps = video.get(CAP_PROP_FPS);
//	int width = video.get(CAP_PROP_FRAME_WIDTH);
//	int height = video.get(CAP_PROP_FRAME_HEIGHT);
//	//VideoWriter writer("./images/zjbb.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);//保存图像 用于查看哪里识别效果不好
//
//	Mat channels[3], binary, Gaussian;
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierarchy;
//
//	Point prePoint;//上一次打击点
//	Point hitPoint;//当前打击点
//
//	namedWindow("video", WINDOW_NORMAL);
//
//	Rect boundRect;//存储最小外接矩形
//	while (true)
//	{
//		Rect point_array[50];//存储合格的外接矩阵
//		if (!video.read(frame)) {
//			break;
//		}
//		split(frame, channels);
//		threshold(channels[t], binary, 180, 255, 0);//对r进行二值化
//
//		GaussianBlur(binary, Gaussian, Size(5,5), 0);
//		//Mat k = getStructuringElement(1, Size(3, 3));
//		findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
//
//		int index = 0;
//		for (int i = 0; i < contours.size(); i++) {
//			//box = minAreaRect(Mat(contours[i]));//计算旋转矩阵
//			//box.points(boxPts.data());
//			boundRect = boundingRect(Mat(contours[i]));//计算最小外接矩形
//			rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//左上 右下
//			
//			try
//			{
//				//问题所在  原因：没有*1.0 导致是一个整数 比例为1.x的矩形被舍去末尾变为1  小于1.5  导致有效轮廓被过滤掉
//				if (double(boundRect.height*1.0 / boundRect.width) > 1.5 && boundRect.width > 10 && boundRect.height > 20) {//满足条件的矩阵保存在point_array
//					point_array[index] = boundRect;
//					index++;
//					//cout << "满足的比例：" << double(boundRect.height*1.0 / boundRect.width) << " 宽：" << boundRect.width <<" 高："<< boundRect.height << endl;
//				}
//				//else {//输出不满足条件的轮廓看看 用于test
//				//	//cout << "不满足的比例：" << double(boundRect.height*1.0 / boundRect.width) << " 宽：" << boundRect.width << " 高：" << boundRect.height << endl;
//				//}
//			}
//			catch (const char* msg)
//			{
//				cout << printf(msg) << endl;
//				continue;
//			}
//		}
//
//		if (index < 2) {//没检测到合格的轮廓 直接 下一帧  
//			circle(frame, prePoint, 5, Scalar(255, 255, 255), -1);//若是只检测到1个边框或者没有边框 则打击前面几帧的点 直到再次被检测到
//			imshow("video", frame);
//			//writer.write(frame);
//			cv::waitKey(10);
//			//cout << "no" << endl;
//			cout <<"prePoint:" << prePoint << endl;
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
//			hitPoint.x = (point3.x - point1.x) / 2 + point1.x;//矩形中心点
//			hitPoint.y = (point4.y - point3.y) / 2 + point1.y;
//			prePoint = hitPoint;
//			cout << "hitPoint:" << hitPoint << endl;
//
//			circle(frame, hitPoint, 5, Scalar(255, 255, 255), -1);
//			//cout << p[0] << p[1] << p[2] << p[3] << endl;
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
//		//writer.write(frame);
//		//cout << "yes" << endl;
//		cv::waitKey(20);
//	}
//	cv::destroyAllWindows();
//
//	waitKey(0);
//}
//
////红色能量机关识别
//bool colorr = false; //false 红色  true蓝色
//void Game::Q2r(){
//	//问题：
//	//1.识别蓝色的效果远远不如红色  通过可视化二值化图发现阈值效果不理想 
//	//2.红色效果也变差了？？？ 原因：顺手调了阈值。。。
//
//	namedWindow("image", WINDOW_NORMAL);
//	//namedWindow("mask", WINDOW_NORMAL);
//	namedWindow("binaryImage", WINDOW_NORMAL);
//	
//	//存放通道
//	Mat Channels[3];
//	Mat image;
//	vector<Point2f> centerPoint;
//
//	String s= "./images/2_red.mp4";
//
//	VideoCapture capture(s);
//	int fps = capture.get(CAP_PROP_FPS);
//	int width = capture.get(CAP_PROP_FRAME_WIDTH);
//	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
//	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);
//	VideoWriter writer1("./images/x_red.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
//	VideoWriter writer2("./images/x_red_bi.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), false);
//
//	while (true)
//	{
//		double startTime = cv::getTickCount();
//		if (!capture.read(image)) {
//			break;
//		}
//
//		split(image, Channels);
//		Mat colorimage = Channels[2] - Channels[0];//红色
//
//		Mat binaryImage;
//		threshold(colorimage, binaryImage, 110, 255, THRESH_BINARY);//红色最佳110
//		//
//
//		vector<vector<Point>>contours;//轮廓数组
//		vector<Vec4i>hierarchy; //一个参数
//		Point2i center; //用来存放找到的目标的中心坐标
//		//提取所有轮廓并建立网状轮廓结构
//		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
//
//		//记录每个轮廓的子轮廓的个数 如果子轮廓为3则为目标扇
//		int contour[30] = { 0 };
//		for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
//		{
//			if (hierarchy[i][3] != -1) //有父轮廓 
//			{
//				contour[hierarchy[i][3]]++; //对该父轮廓进行记录   目的是寻找父轮廓为3的轮廓 
//			}
//		}
//
//		for (int i = 0; i < 30; i++) {
//			if (contour[i] >= 3) {//目标扇叶
//				//drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8);
//				//imshow("image", image);//符合预先设想
//				//waitKey(0);
//
//				//对目标扇叶绘制大矩形 
//				RotatedRect rect0 = minAreaRect(contours[i]);
//				Point2f vertices0[4];
//				rect0.points(vertices0);
//
//				//再找出目标扇叶的两个子轮廓 方便缩小范围 实现精确打击
//				int child1 = hierarchy[i][2];//目标扇叶的的第一个子轮廓
//				int child2, child3;
//				if (hierarchy[child1][0] != -1) {//目标扇叶的的第一个子轮廓的同级轮廓（上一个或者下一个） 也就是目标扇叶的第二个子轮廓  
//					child2 = hierarchy[child1][0];
//				}
//				else {
//					child2 = hierarchy[child1][1];
//				}
//
//				if (hierarchy[child2][0] != -1) {//同理
//					child3 = hierarchy[child2][0];
//				}
//				else {
//					child3 = hierarchy[child2][1];
//				}
//
//				//drawContours(binaryImage, contours, child1, Scalar(255, 255, 255), 2, 8);
//				//drawContours(binaryImage, contours, child2, Scalar(255, 255, 255), 2, 8);
//				//drawContours(binaryImage, contours, child3, Scalar(255, 255, 255), 2, 8);
//
//				vector<Point> points;//三个子轮廓所包含所有的点  用这些点拟合一个圆 圆心就是要打击的点
//				points.insert(points.end(), contours[child1].begin(), contours[child1].end());
//				points.insert(points.end(), contours[child2].begin(), contours[child2].end());
//				points.insert(points.end(), contours[child3].begin(), contours[child3].end());
//
//				//拟合圆
//				Point2f center;
//				float radius;
//				minEnclosingCircle(points, center, radius);
//				circle(image, center, 8, Scalar(255, 255, 255), -1, LINE_AA);
//				circle(binaryImage, center, 6, Scalar(255, 255, 255), -1, LINE_AA);
//				cout << center << endl;
//				
//				//绘制小矩形
//				//RotatedRect rect1 = minAreaRect(points);
//				// 获取最小面积矩形的四个顶点
//				//Point2f vertices1[4];
//				//rect1.points(vertices1);
//
//				for (int i = 0; i < 4; i++) {
//					//line(image, vertices1[i % 4], vertices1[(i + 1) % 4], Scalar(255, 255, 255),2);
//					//line(binaryImage, vertices1[i % 4], vertices1[(i + 1) % 4], Scalar(255, 255, 255), 2);
//					line(image, vertices0[i % 4], vertices0[(i + 1) % 4], Scalar(255, 255, 255),2);
//					line(binaryImage, vertices0[i % 4], vertices0[(i + 1) % 4], Scalar(255, 255, 255), 2);
//				}
//				//circle(image, rect1.center, 5, Scalar(255, 255, 255), -1);
//				//centerPoint.push_back(rect1.center);
//				//cout<<rect1.center<<endl;
//			}
//		}
//
//		double endTime = cv::getTickCount();
//		double totalTime = (endTime - startTime) / cv::getTickFrequency();
//
//		//writer1.write(image);
//		//writer2.write(binaryImage);
//		imshow("image", image);
//		imshow("binaryImage", binaryImage);
//		cout << totalTime << endl;//识别红色的效率远远高于识别蓝色的效率？？？
//		//imshow("mask", mask);
//		waitKey(1);
//	}
//
//	cout << "finish" << endl;
//	waitKey(0);
//}
//
////蓝色能量机关识别
//void Game::Q2b() {
//	namedWindow("image", WINDOW_NORMAL);
//	namedWindow("mask", WINDOW_NORMAL);
//	namedWindow("binaryImage", WINDOW_NORMAL);
//
//	Mat binaryImage;
//	Mat Channels[3];
//
//	Mat image;
//	vector<Point2f> centerPoint;
//
//	String s= "./images/1_blue.mp4";
//	VideoCapture capture(s);
//	int fps = capture.get(CAP_PROP_FPS);
//	int width = capture.get(CAP_PROP_FRAME_WIDTH);
//	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
//	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);
//	VideoWriter writer1("./images/x_blue.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
//	VideoWriter writer2("./images/x_blue_bi.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), false);
//
//	while (true)
//	{
//		double startTime = cv::getTickCount();
//
//		if (!capture.read(image)){
//			break;
//		}
//
//		split(image, Channels);
//		threshold(Channels[0], binaryImage, 245, 255, THRESH_BINARY);
//		Mat k = getStructuringElement(1, Size(3, 3));
//		morphologyEx(binaryImage, binaryImage, 3, k, Point(-1, -1), 3);//复杂度过高，换一种方式提取蓝色特征吧。。。
//
//		vector<vector<Point>>contours;//轮廓数组
//		vector<Vec4i>hierarchy; //一个参数
//		Point2i center; //用来存放找到的目标的中心坐标
//		//提取所有轮廓并建立网状轮廓结构
//		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
//
//		//记录每个轮廓的子轮廓的个数 如果子轮廓为5则为目标扇
//		int contour[30] = { 0 };
//		for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
//		{
//			if (hierarchy[i][3] != -1) //有父轮廓 
//			{
//				contour[hierarchy[i][3]]++; //对该父轮廓进行记录   目的是寻找父轮廓为3的轮廓 
//			}
//		}
//
//		for (int i = 0; i < 30; i++) {
//			if (contour[i] >= 5) {//目标扇叶
//
//				//对目标扇叶绘制矩形
//				RotatedRect rect0 = minAreaRect(contours[i]);
//				Point2f vertices0[4];
//				rect0.points(vertices0);
//
//				//再找出目标扇叶的子轮廓 方便缩小范围 实现精确打击
//				int child1 = hierarchy[i][2];//目标扇叶的的第一个子轮廓
//				int child2, child3, child4, child5;;
//				if (hierarchy[child1][0] != -1) {//目标扇叶的的第一个子轮廓的同级轮廓（上一个或者下一个） 也就是目标扇叶的第二个子轮廓  
//					child2 = hierarchy[child1][0];
//				}
//				else {
//					child2 = hierarchy[child1][1];
//				}
//
//				if (hierarchy[child2][0] != -1) {//同理
//					child3 = hierarchy[child2][0];
//				}
//				else {
//					child3 = hierarchy[child2][1];
//				}
//
//				if (hierarchy[child3][0] != -1) {//同理
//					child4 = hierarchy[child3][0];
//				}
//				else {
//					child4 = hierarchy[child3][1];
//				}
//
//				if (hierarchy[child4][0] != -1) {//同理
//					child5 = hierarchy[child4][0];
//				}
//				else {
//					child5 = hierarchy[child4][1];
//				}
//
//				//drawContours(binaryImage, contours, child1, Scalar(255, 255, 255), 2, 8);
//				//drawContours(binaryImage, contours, child2, Scalar(255, 255, 255), 2, 8);
//				//drawContours(binaryImage, contours, child3, Scalar(255, 255, 255), 2, 8);
//				//drawContours(binaryImage, contours, child4, Scalar(255, 255, 255), 2, 8);
//				//drawContours(binaryImage, contours, child5, Scalar(255, 255, 255), 2, 8);
//
//				vector<Point> points;//三个子轮廓所包含所有的点
//				points.insert(points.end(), contours[child1].begin(), contours[child1].end());
//				points.insert(points.end(), contours[child2].begin(), contours[child2].end());
//				points.insert(points.end(), contours[child3].begin(), contours[child3].end());
//				points.insert(points.end(), contours[child4].begin(), contours[child4].end());
//				points.insert(points.end(), contours[child5].begin(), contours[child5].end());
//
//				Point2f center;
//				float radius;
//				minEnclosingCircle(points, center, radius);
//				circle(image, center, 8, Scalar(255, 255, 255), -1, LINE_AA);
//				circle(binaryImage, center, 6, Scalar(255, 255, 255), -1, LINE_AA);
//				//cout << center << endl;
//
//				for (int i = 0; i < 4; i++) {
//					line(image, vertices0[i % 4], vertices0[(i + 1) % 4], Scalar(255, 255, 255), 2, LINE_AA);
//					line(binaryImage, vertices0[i % 4], vertices0[(i + 1) % 4], Scalar(255, 255, 255), 2, LINE_AA);
//				}
//			}
//		}
//
//		writer1.write(image);
//		writer2.write(binaryImage);
//		imshow("image", image);
//		imshow("binaryImage", binaryImage);
//
//		double endTime = cv::getTickCount();
//		double totalTime = (endTime - startTime) / cv::getTickFrequency();
//
//		cout << totalTime << endl;//效率太低了....
//		waitKey(1);
//	}
//	cout << "finish" << endl;
//	waitKey(0);
//}
//
////picture test
//void Game::Q2() {
//	Mat image = imread("./images/dblue2.png");
//
//	//Mat Channels[3];
//	//split(image, Channels);
//	//Mat colorimage = Channels[0] - Channels[2];
//	//Mat binaryImage; 
//	//threshold(colorimage, binaryImage, 140, 255, THRESH_BINARY);
//	//Mat k = getStructuringElement(1, Size(3, 3));//操作后效果更加不好
//	//morphologyEx(binaryImage, binaryImage,3, k,Point(-1,-1));  //pass
//
///*	Mat hsv,out, binaryImage;
//	cvtColor(image, hsv, COLOR_RGB2HSV);
//	inRange(hsv, Scalar(0, 0, 221), Scalar(180, 30, 255), out);*///扣不出来？？？
//
//	Mat binaryImage;
//	Mat Channels[3];
//	split(image, Channels);
//	Mat colorimage = Channels[0] - Channels[1];
//
//	threshold(colorimage, binaryImage, 70, 255, THRESH_BINARY);
//	//Mat k = getStructuringElement(1, Size(3, 3));
//	//morphologyEx(binaryImage, binaryImage, 3, k,Point(-1,-1),2); //pass
//	//morphologyEx(binaryImage, binaryImage, 0, k, Point(-1, -1));
//
//	vector<vector<Point>>contours;//轮廓数组
//	vector<Vec4i>hierarchy; //一个参数
//	Point2i center; //用来存放找到的目标的中心坐标
//	//提取所有轮廓并建立网状轮廓结构
//	findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
//
//	for (int t = 0; t < contours.size(); t++) {
//		drawContours(image, contours, t, Scalar(0, 255, 0), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
//		imshow("image", image);
//		waitKey(0);
//	}
//
//	//寻找有五个子轮廓的轮廓 就是要击打的扇叶
//	int contour[30] = { 0 };
//	for (int i = 0; i < contours.size(); i++)//遍历检测的所有轮廓
//	{
//		if (hierarchy[i][3] != -1) //有父轮廓 
//		{
//			contour[hierarchy[i][3]]++; //对该父轮廓进行记录  目的是寻找父轮廓为3的轮廓 
//		}
//	}
//
//
//	for (int i = 0; i < 30; i++) {
//		if (contour[i] >= 5) {//目标扇叶
//			drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8);
//			imshow("image", image);//符合预先设想
//			waitKey(0);
//
//			////再找出目标扇叶的两个子轮廓 方便缩小范围 实现精确打击
//			int child1 = hierarchy[i][2];//目标扇叶的的第一个子轮廓
//			int child2, child3;
//			if (hierarchy[child1][0] != -1) {//目标扇叶的的第一个子轮廓的同级轮廓（上一个或者下一个） 也就是目标扇叶的第二个子轮廓  
//				child2 = hierarchy[child1][0];
//			}
//			else {
//				child2 = hierarchy[child1][1];
//			}
//
//			if (hierarchy[child2][0] != -1) {//同理
//				child3 = hierarchy[child2][0];
//			}
//			else {
//				child3 = hierarchy[child2][1];
//			}
//
//			drawContours(image, contours, child1, Scalar(255, 255, 255), 2, 8);
//			imshow("image", image);
//			waitKey(0);
//
//			drawContours(image, contours, child2, Scalar(255, 255, 255), 2, 8);
//			imshow("image", image);
//			waitKey(0);
//
//			drawContours(image, contours, child3, Scalar(255, 255, 255), 2, 8);
//			imshow("image", image);
//			waitKey(0);
//		}
//	}
//	//Mat blueImage = Channels.at(0) - Channels.at(2);//蓝色
//	waitKey(0);
//}
//
//
//
//
////用于参考筛选轮廓
////vector<vector<Point>> contours;
////vector<Vec4i> hierarchy;
////double maxArea = -1;
////int maxId;
////findContours(dilateImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
////for (int i = 0; i < contours.size(); i++) {
////	vector<Point>points;
////	double area = contourArea(contours[i]);
////	/*面积排除噪声*/
////	if (area < 20 || area>10000)
////		continue;
////	/*找到没有父轮廓的轮廓*/
////	if (hierarchy[i][3] >= 0 && hierarchy[i][3] < contours.size())
////		continue;
////	/*找没子轮廓的*/
////	if (hierarchy[i][2] >= 0 && hierarchy[i][2] < contours.size())
////		continue;
////	/*找面积最大的轮廓*/
////	if (maxArea <= area)
////	{
////		maxArea = area;
////		maxId = i;
////	}
////	/*控制误差范围*/
////	if (area <= maxArea + 50 && area >= maxArea - 50) {
////		maxArea = area;
////		maxId = i;
////	}
////}
////if (maxId >= 0 && maxId < contours.size()) {
////	/*画出需打部位轮廓*/
////	drawContours(test, contours, maxId, Scalar(0, 255, 255), 1, 8);
////}
//
//
//
//
//
//
//
//
////对bar的理解
////Mat img;
////Mat mask;
////void onTrackbar(int value, void* userdata)
////{
////	//imshow("img2", img);
////	int red_value = getTrackbarPos("Red", "Image");//
////	int green_value = getTrackbarPos("Green", "Image");
////	int blue_value = getTrackbarPos("Blue", "Image");
////
////	Mat dst(img.size(), img.type());
////	// 获取滑动条的当前值
////	cout << red_value<< " " << green_value<<" " << blue_value << endl;
////	// 更新图像的三个通道的像素值
////	dst = Scalar(red_value, green_value, blue_value);
////	//imshow("dst", dst);//没问题
////
////	Mat add(img.size(), img.type());//加权结果
////	addWeighted(dst, 0.5, img, 0.5, 0, add);
////	//imshow("add", add);
////
////	Mat imgc = img.clone();
////	add.copyTo(imgc, mask);//原先的方案会改变img（原始图像）
////	imshow("Image", imgc);
////}
////
////void Game::Dog() {
////	img = imread("./images/dog.jpg");
////	Mat channels[3];
////	mask = Mat(img.size(),img.type());
////	//画板
////
////	split(img, channels);
////	threshold(channels[0], mask, 150, 255, THRESH_BINARY);//提取要变色的图像
////
////	int r = 0;
////	int b = 0;
////	int g = 0;
////	namedWindow("Image", WINDOW_NORMAL);
////	resizeWindow("Image", 500, 500);
////
////	//namedWindow("dst", WINDOW_NORMAL);
////	//resizeWindow("dst", 500, 500);
////
////	//namedWindow("add", WINDOW_NORMAL);
////	//resizeWindow("add", 500, 500);
////
////
////	//对dst进行变色 再赋值回原图像
////	createTrackbar("Red", "Image", &r, 255, onTrackbar);
////	createTrackbar("Green", "Image", &b, 255, onTrackbar);
////	createTrackbar("Blue", "Image", &g, 255, onTrackbar);
////
////	waitKey(0);
////}



