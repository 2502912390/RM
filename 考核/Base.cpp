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
	cvtColor(img, hsv, COLOR_BGR2HSV);//ת��hsv����ۺ�ɫ
	//����֪��ɫ��ɫ�ռ�������
	inRange(hsv, Scalar(156, 43, 46), Scalar(180, 255, 255), out1);//��ȡ�������� 
	//inRange(hsv, Scalar(0, 43, 46), Scalar(10, 255, 255), out2);//��ȡ��x

	Mat dst;//out��������� ����ʹ�ÿ������������
	Mat k = getStructuringElement(1, Size(3, 3));
	morphologyEx(out1, dst, 2, k, Point(-1, -1), 5);

	vector<Point2f> points;
	findNonZero(dst, points);//��ȡ���з�0�ĵ�

	vector<Point2f> triangle;//����ɢ�����������
	double area = minEnclosingTriangle(points, triangle);

	Mat ans(img.size(), img.type(),Scalar(0,0,0));

	//����������
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
	cvtColor(img, hsv, COLOR_BGR2HSV);//ת��hsv����ۺ�ɫ
	//����֪��ɫ��ɫ�ռ�������
	inRange(hsv, Scalar(156, 43, 46), Scalar(180, 255, 255), out1);//��ȡ�������� 
	//inRange(hsv, Scalar(0, 43, 46), Scalar(10, 255, 255), out2);//��ȡ��x

	Mat dst;//out��������� ����ʹ�ÿ������������
	Mat k = getStructuringElement(1, Size(3, 3));
	morphologyEx(out1, dst, 2, k,Point(-1,-1),5);

	//���������ն� �����ٽ�����
	morphologyEx(dst, dst, 3, k, Point(-1, -1),20);

	Mat canny;
	Canny(dst, canny, 80, 160, 3, false);

	cvtColor(canny, canny, COLOR_GRAY2BGR);//ת��3ͨ��ͼ�񷽱���ɫ

	Scalar lightBlue(255, 200, 100);//����ǳ��ɫ

	for (int i = 0; i < canny.rows; i++) {//��ÿ��ͨ����ɫ 
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

	imshow("ԭͼ", img);
	imshow("�ڰ�ͼ��", gray);
	imshow("canny", canny);
	imwrite("./images/canny.png",canny);
	waitKey(0);
}

Mat frame;//��ȫ�ֱ����洢

// �ص����������ڴ���bar�����¼�
void onExposureChange(int exposure, void* userData) {
	VideoCapture cap = *(VideoCapture*)userData;//voidתVideoCapture
	double newExposure = exposure / 100.0;  //newExposure��λ����  ӳ�䵽[0,1]֮��
	cap.set(CAP_PROP_EXPOSURE, newExposure);//�����ع��
}

void  Base::Q2() {
	VideoCapture capture(0);//������ͷ

	if (!capture.isOpened()) {  // �������ͷ�Ƿ���ȷ��
		cout << "Could not open the camera." << endl;
		return;
	}

	//��ȡ����ͷ��Ϣ
	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ��֡��
	int exposureTime = capture.get(cv::CAP_PROP_EXPOSURE);//��ȡ�ع�ʱ�� 0
	cout << "fps" << fps << " " << "��" << width << " " << "��" << height << " " << "��֡�� " << numFrame << "�ع�ʱ��" << exposureTime<<endl;

	VideoWriter writer("./images/output1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
	//					����·��		        ���ַ������ʽ                                    true��ʾ�������Ƶ�ļ��ǲ�ɫ��
	if (!writer.isOpened()) {
		cout << "error" << endl;
		return;
	}

	namedWindow("��ʾ��", WINDOW_AUTOSIZE);

	int light = 50;//��ʼֵ
	//createTrackbar("����", "��ʾ��", &light, 255, callBack, 0);//����ʾ�򴰿ڴ�������bar
	createTrackbar("����", "��ʾ��", &light, 100);//��ʹ�ûص����� ���ֵ����Ϊ100�����Ŵ���������
	
	int initialExposure = 10;  // ��ʼ�ع�ʱ��
	createTrackbar("�ع�", "��ʾ��", &initialExposure, 100, onExposureChange, &capture);//ʹ�ûص����� //��bug�� ����ͷ���⣿����
	//createTrackbar("�ع�", "��ʾ��", &exposureTime, 100);
	
	while (true)
	{
		//double newExposure = exposureTime / 100.0;  //newExposure��λ����  ӳ�䵽[0,1]֮��
		//capture.set(CAP_PROP_EXPOSURE, newExposure);//�����ع��

		if (!capture.read(frame)) {
			break;
		}
		
		frame.convertTo(frame, -1,light/50.0, 0);//50/50=1��ʼ�������Ƚ��е��� 
		imshow("��ʾ��", frame);
		writer.write(frame);//��ÿһ֡���б���

		char c = waitKey(5);
		if (c == 'q') {
			break;
		}
	}

	capture.release();//�ͷ���Դ
	writer.release();

	//�Ա궨����нǵ���
	//Mat picture = imread("./images/bd.png");
	//Size board_size = Size(10, 7);//�궨���ڽǵ���Ŀ���� �У�
	//Mat gray1;
	//cvtColor(picture, gray1, COLOR_BGR2GRAY);
	//vector<Point2f> img1_points;
	//findChessboardCorners(gray1, board_size, img1_points);//���ͼ�������̸�ģʽ�Ľǵ�
	//find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));//�Գ�ʼ�Ľǵ�������������ؼ�����Ż�
	//bool pattern = true;
	//drawChessboardCorners(picture, board_size, img1_points, pattern);//���Ƽ�⵽�����̸�ǵ�
	//imshow("�궨��ǵ���", picture);
	//waitKey(0);
}

Point sp(-1, -1);//��ʼ
Point ep(-1, -1);//����
Mat temp;
//int event����ʾ��ǰ������¼����ͣ������� EVENT_MOUSEMOVE���ƶ���,EVENT_LBUTTONDOWN��������£�,EVENT_LBUTTONUP(̧��)
//int x �� int y����ʾ��ǰ����¼�����������λ�á�
//int flags����ʾ��ǰ������¼��ĸ��ӱ�־��Ϣ��������갴��״̬�ȡ�
//void* param���� setMouseCallback ���������һ���������������ڴ��ݶ���Ĳ������ص�������
static void on_draw(int event, int x, int y, int flags, void* userdata) {
	Mat image = *((Mat*)userdata);//תmat����ָ�� ������
	if (event == EVENT_LBUTTONDOWN) {//���¼�¼��ǰλ������
		sp.x = x;
		sp.y = y;
	}
	else if (event == EVENT_LBUTTONUP) {
		ep.x = x;//̧���¼�������
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;

		if (dx > 0 && dy > 0) {
			Rect box(sp.x, sp.y, dx, dy);//������ʼ��ĩrect

			temp.copyTo(image);
			Mat ROI = image(box);

			imshow("ROI����", ROI);//��ʾ image ͼ���е� box ��������
			imwrite("./images/smallmimi.png", ROI);

			rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);//��ԭͼ���ƾ���
			imshow("������", image);
			
			cout << "��������" << ROI.rows / 2 << " " << ROI.cols / 2 << endl;

			// 
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;//��
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);//��Ҫ��ԭͼ����
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);

				//��ȡ��ǰ������Ϣ
				Vec3b rgb = image.at<Vec3b>(y, x);
				int r = static_cast <int> (rgb[2]);
				int g = static_cast <int> (rgb[1]);
				int b = static_cast <int> (rgb[0]);

				String s = "rgb: " + to_string(r) + "," + to_string(g) + "," + to_string(b) + " " + "position: " + to_string(y) + "," + to_string(x);
				putText(image, s, Point(x-dx, y-dy), 0, 0.5, Scalar(255,0,0), 1, 8, false);//�������
				imshow("������", image);
			}
		}
	}
}

void Base::Q3() {
	Mat cat = imread("./images/mimi.png");
	namedWindow("������", WINDOW_AUTOSIZE);
	setMouseCallback("������", on_draw, (void*)(&cat));

	imshow("������", cat);
	temp = cat.clone();
	waitKey(0);
}

