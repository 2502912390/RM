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

	imshow("img", img);
	imshow("ans", ans);
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
	VideoCapture cap = *(VideoCapture*)userData;//void转VideoCapture
	double newExposure = exposure / 100.0;  //newExposure单位是秒  映射到[0,1]之间
	cap.set(CAP_PROP_EXPOSURE, newExposure);//设置曝光度
}

void  Base::Q2() {
	VideoCapture capture(0);//打开摄像头

	if (!capture.isOpened()) {  // 检查摄像头是否正确打开
		cout << "Could not open the camera." << endl;
		return;
	}

	//获取摄像头信息
	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);//获取视频总帧率
	int exposureTime = capture.get(cv::CAP_PROP_EXPOSURE);//获取曝光时间 0
	cout << "fps" << fps << " " << "宽" << width << " " << "高" << height << " " << "总帧率 " << numFrame << "曝光时间" << exposureTime<<endl;

	VideoWriter writer("./images/output1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
	//					保存路径		        四字符编码格式                                    true表示输出的视频文件是彩色的
	if (!writer.isOpened()) {
		cout << "error" << endl;
		return;
	}

	namedWindow("显示框", WINDOW_AUTOSIZE);

	int light = 50;//初始值
	//createTrackbar("亮度", "显示框", &light, 255, callBack, 0);//在显示框窗口创建亮度bar
	createTrackbar("亮度", "显示框", &light, 100);//不使用回调函数 最大值设置为100，最多放大两倍亮度
	
	int initialExposure = 10;  // 初始曝光时间
	createTrackbar("曝光", "显示框", &initialExposure, 100, onExposureChange, &capture);//使用回调函数 //有bug？ 摄像头问题？？？
	//createTrackbar("曝光", "显示框", &exposureTime, 100);
	
	while (true)
	{
		//double newExposure = exposureTime / 100.0;  //newExposure单位是秒  映射到[0,1]之间
		//capture.set(CAP_PROP_EXPOSURE, newExposure);//设置曝光度

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
	//Mat picture = imread("./images/bd.png");
	//Size board_size = Size(10, 7);//标定板内角点数目（行 列）
	//Mat gray1;
	//cvtColor(picture, gray1, COLOR_BGR2GRAY);
	//vector<Point2f> img1_points;
	//findChessboardCorners(gray1, board_size, img1_points);//检测图像中棋盘格模式的角点
	//find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));//对初始的角点坐标进行亚像素级别的优化
	//bool pattern = true;
	//drawChessboardCorners(picture, board_size, img1_points, pattern);//绘制检测到的棋盘格角点
	//imshow("标定板角点检测", picture);
	//waitKey(0);
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
				temp.copyTo(image);//不要在原图操作
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);

				//读取当前像素信息
				Vec3b rgb = image.at<Vec3b>(y, x);
				int r = static_cast <int> (rgb[2]);
				int g = static_cast <int> (rgb[1]);
				int b = static_cast <int> (rgb[0]);

				String s = "rgb: " + to_string(r) + "," + to_string(g) + "," + to_string(b) + " " + "position: " + to_string(y) + "," + to_string(x);
				putText(image, s, Point(x-dx, y-dy), 0, 0.5, Scalar(255,0,0), 1, 8, false);//添加文字
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

