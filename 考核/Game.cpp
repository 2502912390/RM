
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

//const int kThreashold = 180;//��ֵ
//const int kMaxVal = 255;
//const Size kGaussianBlueSize = Size(5, 5);

#define PI 3.1415926

void Game::Q1p() {//
	
	Mat frame = imread("./images/zj.png");
	Mat channels[3], binary, Gaussian;

	vector<vector<Point>> contours;//������
	vector<Vec4i> hierarchy;
	RotatedRect AreaRect;//�洢������С(��ת)��Ӿ���

	split(frame, channels);
	//Mat colorimage = channels[0] - channels[2];

	threshold(channels[0], binary, 180, 255, 0);//��b���ж�ֵ��
	GaussianBlur(binary, Gaussian, Size(3, 3), 0);
	findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	RotatedRect light[20];//��ɸѡ�����ת����
	Point2f points[4];//��RotatedRectת�����ĵ�����

	int index = 0;//���ɸѡ���������±�
	
	for (int i = 0; i < contours.size(); i++) {
		//boundRect = boundingRect(Mat(contours[i]));//������С��Ӿ���
		AreaRect = minAreaRect(Mat(contours[i]));//������С����ת���� ��ΪҪʹ��angle
		double area = AreaRect.size.area();//���

		//�ж�������  ���
		if (area > 100) { //�Ȱ�һЩ̫С�ĸ�������ɸѡɾ��
			light[index] = AreaRect;
			//AreaRect.points(points);
			//rectangle(frame, point_array[index].tl(), point_array[index].br(), Scalar(255, 255, 255), 2, 8, 0);//���� ����
			//for (int i = 0; i < 4; i++) {
			//	line(frame, points[i % 4], points[(i + 1) % 4], Scalar(255, 255, 255), 2);
			//}
			index++;
		}
	}

	//imshow("1", frame);
	//waitKey(0);

	if (index < 1) {//С��һ��
		return;
	}
	//vector<RotatedRect> armor_final; //������յľ���
	RotatedRect armor;// �����������������ɵľ���
	double angle_dif;//�����ǶȵĲ�ֵ
	double hight_dif;//�߶Ȳ�
	double area_diff;//�����

	for (int i = 0; i < index; i++) {
		for (int j = i + 1; j < index; j++) {
			angle_dif = abs(light[i].angle - light[j].angle);//�ǶȲ�
			hight_dif = abs(light[i].center.y - light[j].center.y);//�߶Ȳ�
			area_diff = abs(light[i].size.area() - light[i].size.area());//�����

			if (angle_dif < 4&& hight_dif<20&& area_diff<50) {
				//cout << light[i].angle << " " << light[j].angle<<endl;

				light[i].points(points);
				/************TEST************/
				//for (int i = 0; i < 4; i++) {//��������1
				//	line(frame, points[i % 4], points[(i + 1) % 4], Scalar(0, 0, 255), 2);
				//}
				//imshow("frame", frame);
				////cout<<"1����" << light[i].size.width << " ��" << light[i].size.height << endl;
				//waitKey(0);
				//
				//light[j].points(points);
				//for (int i = 0; i < 4; i++) {//��������2
				//	line(frame, points[i % 4], points[(i + 1) % 4], Scalar(0, 0, 255), 2);
				//}
				//imshow("frame", frame);
				////cout << "2����" << light[j].size.width << " ��" << light[j].size.height << endl;
				//waitKey(0);
				/************TEST**********/
				
				armor.center.x = (light[i].center.x + light[j].center.x) / 2; // װ�����ĵ�x����
				armor.center.y = (light[i].center.y + light[j].center.y) / 2; // װ�����ĵ�y����
				
				//bug������ ��� ԭ����Ԥ��ĺͼ��㶨��ĳ����ǲ�һ����
				double height, width;//�����Ķ�Ϊ�ߣ��̵Ķ�Ϊ��
				if (light[j].size.width > light[j].size.height) {//ע���ĸ��Ǹ� �ĸ��ǿ� 
					armor.size.width = light[j].size.width;
					armor.size.height = abs(light[i].center.x - light[j].center.x); // װ�װ�ĸ߶�
				}
				else {
					armor.size.width = abs(light[i].center.x - light[j].center.x);
					armor.size.height = light[j].size.height;
				}

				armor.angle = (light[i].angle + light[j].angle) / 2;
				//cout <<"���" << armor.size.area() << endl;
				//cout << "�߿����" << armor.size.height / armor.size.width << endl;

				//cout << light[i].center.x << " " << light[j].center.x << endl;
				//cout <<"װ�װ���" << abs(light[i].center.x - light[j].center.x) << endl;

				//Point point1 = Point(light[i].center.x, light[i].center.y - light[i].size.height / 2);
				//Point point2 = Point(light[i].center.x, light[i].center.y + light[i].size.height / 2);
				//Point point3 = Point(light[j].center.x, light[j].center.y - light[j].size.height / 2);
				//Point point4 = Point(light[j].center.x, light[j].center.y + light[j].size.height / 2);
				//Point points[4] = { point1,point2,point4,point3 };

				if (armor.size.area() < 5000) {//�����ɸѡ�ɡ����� ���Ľ�������
					armor.points(points);
					for (int i = 0; i < 4; i++) {//������������ɵľ��ο�
						line(frame, points[i % 4], points[(i + 1) % 4], Scalar(0, 0, 255), 2);
					}
					circle(frame, armor.center, 2, Scalar(255, 255, 255), -1);//û����
					cout <<"������������" << armor.center << endl;
					imshow("frame", frame);
					//waitKey(0);
					//armor_final.push_back(armor); // ������Ҫ��ľ��ηŽ�������
				}
			}
		}
	}
	
	waitKey(0);
}

//����Ӷ�Ŀ���ѡ����Ŀǰֻ�ܼ��������������׷��ʧ���жϣ������ü�֡û��⵽װ�װ壩
const bool color = true;//����ѡ���ж�װ�װ����ɫ flase��ɫ   true��ɫ
void Game::Q1v() {
	//˼·���ֱ�Ժ���װ�װ��r��bͨ�����з��벢�Ҷ�ֵ����Ȼ��Ѱ������
	//����������ɸѡ ���Ƴ���������������Χ�ǵ���� �Լ� ���ĵ�

	//�������⣺
   //1.ֻ��⵽һ��������ʱ��׷�ٻ�Ͽ�  �������ǰ��֡��Ϊ�ο�
   //2.��ɫװ�װ���ʱ��ʶ�𲻳������� ԭ�򣺼����Ĳ��������߿��Ϊ1.x ��ɸѡ��ʱ��û��ʹ��double���ͣ�������1.x������ֱ�ӹ���Ϊ1��������Ҫ������û��ɸѡ����  ������������ʱ*1.0

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
	VideoWriter writer("./images/blue.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);//����ͼ�� ���ڲ鿴����ʶ��Ч������

	Mat channels[3], binary, Gaussian;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Point prePoint;//��һ�δ����
	Point hitPoint;//��ǰ�����

	namedWindow("video", WINDOW_NORMAL);

	Rect boundRect;//�洢��С��Ӿ���
	while (true)
	{
		double startTime = cv::getTickCount();

		Rect point_array[50];//�洢�ϸ����Ӿ���  �����Ͼ�����
		if (!video.read(frame)) {
			break;
		}
		split(frame, channels);
		threshold(channels[t], binary, 180, 255, 0);//���ж�ֵ��

		GaussianBlur(binary, Gaussian, Size(5, 5), 0);
		//Mat k = getStructuringElement(1, Size(3, 3));
		findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

		int index = 0;
		for (int i = 0; i < contours.size(); i++) {
			//box = minAreaRect(Mat(contours[i]));//������ת����
			//box.points(boxPts.data());
			boundRect = boundingRect(Mat(contours[i]));//������С��Ӿ���

			//��������  ԭ��û��*1.0 ������һ������ ����Ϊ1.x�ľ��α���ȥĩβ��Ϊ1  С��1.5  ������Ч���������˵�
			if (double(boundRect.height * 1.0 / boundRect.width) > 1.5 && boundRect.width > 10 && boundRect.height > 20) {//���������ľ��󱣴���point_array
				point_array[index] = boundRect;
				index++;
				rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//���ƾ��� ���� ����   
				//cout << "����ı�����" << double(boundRect.height*1.0 / boundRect.width) << " ��" << boundRect.width <<" �ߣ�"<< boundRect.height << endl;
			}
			//else {//����������������������� ����test
			//	//cout << "������ı�����" << double(boundRect.height*1.0 / boundRect.width) << " ��" << boundRect.width << " �ߣ�" << boundRect.height << endl;
			//}
		}

		if (index < 2) {//û��⵽�ϸ������ ֱ�� ��һ֡  
			circle(frame, prePoint, 5, Scalar(255, 255, 255), -1);//����ֻ��⵽1���߿����û�б߿� ����ǰ�漸֡�ĵ� ֱ���ٴα���⵽
			imshow("video", frame);
			//writer.write(frame);
			cv::waitKey(10);
			//cout << "no" << endl;
			cout << "prePoint:" << prePoint << endl;
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

		hitPoint.x = (point3.x - point1.x) / 2 + point1.x;//�������ĵ�
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
		cout << "����ʱ��" << totalTime << endl;

		imshow("video", frame);
		writer.write(frame);
		//cout << "yes" << endl;
		cv::waitKey(20);
	}
	cv::destroyAllWindows();

	waitKey(0);
}

//��ɫ��������ʶ��
bool colorr = false; //false ��ɫ  true��ɫ
void Game::Q2r() {
	namedWindow("image", WINDOW_NORMAL);
	//namedWindow("mask", WINDOW_NORMAL);
	namedWindow("binaryImage", WINDOW_NORMAL);

	//���ͨ��
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

	double pretheta=0;//ǰһ֡�ĽǶ�

	while (true)
	{
		double startTime = cv::getTickCount();
		if (!capture.read(image)) {
			break;
		}

		split(image, Channels);
		Mat colorimage = Channels[2] - Channels[0];//��ɫ

		Mat binaryImage;
		threshold(colorimage, binaryImage, 110, 255, THRESH_BINARY);//��ɫ���110

		vector<vector<Point>>contours;//��������
		vector<Vec4i>hierarchy; //һ������
		//Point2i center; //��������ҵ���Ŀ�����������
		//��ȡ����������������״�����ṹ
		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

		//��¼ÿ���������������ĸ��� ���������Ϊ3��ΪĿ����     
		int contour[30] = { 0 };
		int min_area = 10000;
		int r_id = -1;//Ҫ����Բ�ĵ��������±�
		for (int i = 0; i < contours.size(); i++)//����������������
		{
			if (hierarchy[i][3] != -1) //�и����� 
			{
				contour[hierarchy[i][3]]++; //�Ըø��������м�¼   Ŀ����Ѱ��ֵΪ3������ 
			}
			double area = contourArea(contours[i]);

			//û�������� Ҳû�и����� �����Ҫ���� ΪԲ��  �����е����⡣���� ��������
			if (hierarchy[i][2] == -1 && hierarchy[i][3] == -1&& (area > 20 && area < 1000)) {
				//���þ�������� �������Ϊ�����β���Բ��
				Rect rect = boundingRect(contours[i]);
				if (double(rect.height * 1.0 / rect.width) < 1.2 && double(rect.height * 1.0 / rect.width) > 0.8) {//Բ��λ�ýӽ�������
					if (area <= min_area) {//��Ҫ�������С��
						min_area = area;
						r_id = i;
					}
				}
			}
		}

		Point2f newcenter;//RԲ��
		float newradius;//�뾶

		Point2f center;//�����Բ��
		float radius;

		if (r_id > 0 && r_id < contours.size()) {//�ں���Χ��
			minEnclosingCircle(Mat(contours[r_id]), newcenter, newradius);//���������������С���Բ�İ뾶��Բ��
			circle(image, newcenter, newradius, Scalar(0, 255, 255), 5, 8,0);
		}
		
		for (int i = 0; i < 30; i++) {
			if (contour[i] >= 3) {//Ŀ����Ҷ
				//drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8);
				//imshow("image", image);//����Ԥ������
				//waitKey(0);

				//��Ŀ����Ҷ���ƴ���� 
				RotatedRect rect0 = minAreaRect(contours[i]);
				Point2f vertices0[4];
				rect0.points(vertices0);

				//���ҳ�Ŀ����Ҷ������������ ������С��Χ ʵ�־�ȷ���
				int child1 = hierarchy[i][2];//Ŀ����Ҷ�ĵĵ�һ��������
				int child2, child3;
				if (hierarchy[child1][0] != -1) {//Ŀ����Ҷ�ĵĵ�һ����������ͬ����������һ��������һ���� Ҳ����Ŀ����Ҷ�ĵڶ���������  
					child2 = hierarchy[child1][0];
				}
				else {
					child2 = hierarchy[child1][1];
				}

				if (hierarchy[child2][0] != -1) {//ͬ��
					child3 = hierarchy[child2][0];
				}
				else {
					child3 = hierarchy[child2][1];
				}

				//drawContours(binaryImage, contours, child1, Scalar(255, 255, 255), 2, 8);
				//drawContours(binaryImage, contours, child2, Scalar(255, 255, 255), 2, 8);
				//drawContours(binaryImage, contours, child3, Scalar(255, 255, 255), 2, 8);

				vector<Point> points;//�������������������еĵ�  ����Щ�����һ��Բ Բ�ľ���Ҫ����ĵ�
				points.insert(points.end(), contours[child1].begin(), contours[child1].end());
				points.insert(points.end(), contours[child2].begin(), contours[child2].end());
				points.insert(points.end(), contours[child3].begin(), contours[child3].end());

				//���Բ
				minEnclosingCircle(points, center, radius);
				circle(image, center, 8, Scalar(255, 255, 255), -1, LINE_AA);
				circle(binaryImage, center, 6, Scalar(255, 255, 255), -1, LINE_AA);
				cout << center << endl;

				//����С����
				//RotatedRect rect1 = minAreaRect(points);
				// ��ȡ��С������ε��ĸ�����
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

		//����Ӽ������� ֻ��ĳһ����ﵽһ����������Ϊ�Ǹ÷��򣬷�ֹ���֡���֡�����
		////ע�� ��һ��ֵ��y
		//cout << "һ�����޵�" << (atan2(1, 1) * (180 / 3.1415926)) << " " << (atan2(1, -1) * (180 / 3.1415926)) << endl;//�ԽǶ���ʾ
		//cout << "�������޵�" << (atan2(-1, -1) * (180 / 3.1415926)) << " " << (atan2(-1, 1) * (180 / 3.1415926)) << endl;
		//�Դ�������ļ��㼫����ϵ�ĽǶ� ������֮֡��ĽǶȲ�����ж���˳ʱ�뻹����ʱ��
		//��R��Բ��Ϊ����ԭ��
		double dx = center.x - newcenter.x;
		double dy = center.y - newcenter.y;
		double p = sqrt(dx * dx + dy * dy);//б��
		double theta = atan2(dy, dx) * (180 / PI);//����ɽǶ�
		if (dy > 0) {//ֻ�ڵ�һ/�������ж�
			if (pretheta != 0) {
				double dtheta = theta - pretheta;//��ǰ-ǰһ֡  Ϊʲô��������ǰ֡���ǰһ֡��С������ ����Ӧ�ô�Ŷ԰�������
				cout << "dx " << dx << "dy " << dy;
				cout << "��ǰ�Ƕ�" << theta << " ǰһ֡�Ƕ�" << pretheta << " �ǶȲ�" << dtheta << endl;
				if (dtheta > 0) {//��ʱ��
					//cout << "��ʱ��" << endl;
					cout << "˳ʱ��" << endl;
				}
				else {
					//cout << "˳ʱ��" << endl;
					cout << "��ʱ��" << endl;
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
		cout << "����ʱ��" << totalTime << endl;//ʶ���ɫ��Ч��ԶԶ����ʶ����ɫ��Ч�ʣ�����
		//imshow("mask", mask);
		waitKey(1);
	}
	waitKey(0);
}

//��ɫ��������ʶ��
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
		morphologyEx(binaryImage, binaryImage, 3, k, Point(-1, -1), 3);//���Ӷȹ��ߣ���һ�ַ�ʽ��ȡ��ɫ�����ɡ�����

		vector<vector<Point>>contours;//��������
		vector<Vec4i>hierarchy; //һ������
		Point2i center; //��������ҵ���Ŀ�����������
		//��ȡ����������������״�����ṹ
		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

		//��¼ÿ���������������ĸ��� ���������Ϊ5��ΪĿ����
		int contour[30] = { 0 };
		for (int i = 0; i < contours.size(); i++)//����������������
		{
			if (hierarchy[i][3] != -1) //�и����� 
			{
				contour[hierarchy[i][3]]++; //�Ըø��������м�¼   Ŀ����Ѱ�Ҹ�����Ϊ3������ 
			}
		}

		for (int i = 0; i < 30; i++) {
			if (contour[i] >= 5) {//Ŀ����Ҷ

				//��Ŀ����Ҷ���ƾ���
				RotatedRect rect0 = minAreaRect(contours[i]);
				Point2f vertices0[4];
				rect0.points(vertices0);

				//���ҳ�Ŀ����Ҷ�������� ������С��Χ ʵ�־�ȷ���
				int child1 = hierarchy[i][2];//Ŀ����Ҷ�ĵĵ�һ��������
				int child2, child3, child4, child5;;
				if (hierarchy[child1][0] != -1) {//Ŀ����Ҷ�ĵĵ�һ����������ͬ����������һ��������һ���� Ҳ����Ŀ����Ҷ�ĵڶ���������  
					child2 = hierarchy[child1][0];
				}
				else {
					child2 = hierarchy[child1][1];
				}

				if (hierarchy[child2][0] != -1) {//ͬ��
					child3 = hierarchy[child2][0];
				}
				else {
					child3 = hierarchy[child2][1];
				}

				if (hierarchy[child3][0] != -1) {//ͬ��
					child4 = hierarchy[child3][0];
				}
				else {
					child4 = hierarchy[child3][1];
				}

				if (hierarchy[child4][0] != -1) {//ͬ��
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

				vector<Point> points;//�������������������еĵ�
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

		cout << totalTime << endl;//Ч��̫����....
		waitKey(1);
	}
	cout << "finish" << endl;
	waitKey(0);
}

//picture test  ���ڲ���
void Game::test1() {
	Mat image = imread("./images/dred.png");

	//Mat Channels[3];
	//split(image, Channels);
	//Mat colorimage = Channels[0] - Channels[2];
	//Mat binaryImage; 
	//threshold(colorimage, binaryImage, 140, 255, THRESH_BINARY);
	//Mat k = getStructuringElement(1, Size(3, 3));//������Ч�����Ӳ���
	//morphologyEx(binaryImage, binaryImage,3, k,Point(-1,-1));  //pass

/*	Mat hsv,out, binaryImage;
	cvtColor(image, hsv, COLOR_RGB2HSV);
	inRange(hsv, Scalar(0, 0, 221), Scalar(180, 30, 255), out);*///�۲�����������

	Mat Channels[3];
	split(image, Channels);
	Mat colorimage = Channels[2] - Channels[0];//��ɫ

	Mat binaryImage;
	threshold(colorimage, binaryImage, 110, 255, THRESH_BINARY);//��ɫ���110

	vector<vector<Point>>contours;//��������
	vector<Vec4i>hierarchy; //һ������
	Point2i center; //��������ҵ���Ŀ�����������
	//��ȡ����������������״�����ṹ
	findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

	//��¼ÿ���������������ĸ��� ���������Ϊ3��ΪĿ����     
	int contour[30] = { 0 };
	int min_area = 10000;
	int r_id = 0;
	for (int i = 0; i < contours.size(); i++)//����������������
	{
		//if (hierarchy[i][3] != -1) //�и����� 
		//{
		//	contour[hierarchy[i][3]]++; //�Ըø��������м�¼   Ŀ����Ѱ�Ҹ�����Ϊ3������ 
		//}
		double area = contourArea(contours[i]);
		//û�������� Ҳû�и����� �����Ҫ���� ΪԲ��  �����е����⡣����
		if (hierarchy[i][2] == -1 && hierarchy[i][3] == -1 && (area > 20 && area < 1000)) {
			//���þ�������� �������Ϊ�����β���Բ��
			Rect rect = boundingRect(contours[i]);
			if (double(rect.height * 1.0 / rect.width) < 1.2 && double(rect.height * 1.0 / rect.width) > 0.8) {//Բ��λ�ýӽ�������
				cout << double(rect.height * 1.0 / rect.width) << endl;
				drawContours(image, contours, i, Scalar(0, 255, 255), 2, 8);
				imshow("image", image);
				waitKey(0);
				if (area <= min_area) {//��Ҫ�������С��
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
	//	drawContours(image, contours, t, Scalar(0, 255, 0), 2, 8);//t��ָ��Ҫ���Ƶ����������� ����-1 ������������
	//	imshow("image", image);
	//	waitKey(0);
	//}

	////Ѱ������������������� ����Ҫ�������Ҷ
	//int contour[30] = { 0 };
	//for (int i = 0; i < contours.size(); i++)//����������������
	//{
	//	if (hierarchy[i][3] != -1) //�и����� 
	//	{
	//		contour[hierarchy[i][3]]++; //�Ըø��������м�¼  Ŀ����Ѱ�Ҹ�����Ϊ3������ 
	//	}
	//}


	//for (int i = 0; i < 30; i++) {
	//	if (contour[i] >= 5) {//Ŀ����Ҷ
	//		drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8);
	//		imshow("image", image);//����Ԥ������
	//		waitKey(0);

	//		////���ҳ�Ŀ����Ҷ������������ ������С��Χ ʵ�־�ȷ���
	//		int child1 = hierarchy[i][2];//Ŀ����Ҷ�ĵĵ�һ��������
	//		int child2, child3;
	//		if (hierarchy[child1][0] != -1) {//Ŀ����Ҷ�ĵĵ�һ����������ͬ����������һ��������һ���� Ҳ����Ŀ����Ҷ�ĵڶ���������  
	//			child2 = hierarchy[child1][0];
	//		}
	//		else {
	//			child2 = hierarchy[child1][1];
	//		}

	//		if (hierarchy[child2][0] != -1) {//ͬ��
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
	////Mat blueImage = Channels.at(0) - Channels.at(2);//��ɫ
	//waitKey(0);
}


//��bar�����
Mat img;
Mat mask;
void onTrackbar(int value, void* userdata)
{
	//imshow("img2", img);
	int red_value = getTrackbarPos("Red", "Image");//
	int green_value = getTrackbarPos("Green", "Image");
	int blue_value = getTrackbarPos("Blue", "Image");

	Mat dst(img.size(), img.type());
	// ��ȡ�������ĵ�ǰֵ
	cout << red_value << " " << green_value << " " << blue_value << endl;
	// ����ͼ�������ͨ��������ֵ
	dst = Scalar(red_value, green_value, blue_value);
	//imshow("dst", dst);//û����

	Mat add(img.size(), img.type());//��Ȩ���
	addWeighted(dst, 0.5, img, 0.5, 0, add);
	//imshow("add", add);

	Mat imgc = img.clone();
	add.copyTo(imgc, mask);//ԭ�ȵķ�����ı�img��ԭʼͼ��
	imshow("Image", imgc);
}

void Game::Dog() {
	Mat img = imread("./images/dog2.png");
	Mat mask = Mat(img.size(), img.type());

	//img = imread("./images/dog2.jpg");
	//mask = Mat(img.size(), img.type());

	Mat channels[3];

	split(img, channels);
	threshold(channels[0], mask, 100, 255, THRESH_BINARY);//��ȡҪ��ɫ��ͼ��
	//imshow("mask", mask);

	int r = 0;
	int b = 0;
	int g = 0;
	namedWindow("Image", WINDOW_NORMAL);
	resizeWindow("Image", 500, 500);

	//��dst���б�ɫ �ٸ�ֵ��ԭͼ��
	createTrackbar("Red", "Image", &r, 255, onTrackbar);
	createTrackbar("Green", "Image", &b, 255, onTrackbar);
	createTrackbar("Blue", "Image", &g, 255, onTrackbar);

	waitKey(0);
}

	
//void Game::test2() {
//	VideoCapture cap("./images/2_red.mp4");
//	Mat image;
//	//colorrr = blue;/*��ɫ*/
//	for (;;) {
//		cap.read(image);
//		/*��ʼ����ͼ��*/
//		clock_t start = clock();
//		/*�ı��С�����֡��*/
//		//Mat image= imread("./images/dred.png");
//		resize(image, image, Size(image.cols * 0.35, image.rows * 0.35));
//
//		/*����Ч��չʾ*/
//		Mat test;
//		image.copyTo(test);
//		///////////////////////////////////////////////�ⲿ�ֲ�����Ҫ�����Լ���///////////////////////////////////////////////////////////
//		///*��������ŷ���ͨ�����ͼ��*/
//		//vector<Mat> imgChannels;
//		//split(image, imgChannels);
//		//
//		////��ɫ
//		//Mat midimage = imgChannels.at(2) - imgChannels.at(0);
//		//imshow("1", midimage);
//		//
//		///*��ɫ*/
//		////Mat blueImage = imgChannels.at(0) - imgChannels.at(2);
//		//Mat binaryImage;
//		//Mat binaryImagecricle;
//		///*��ֵ��*/
//		//threshold(midimage, binaryImagecricle, 170, 255, THRESH_BINARY);
//		//threshold(midimage, binaryImage, 90, 255, THRESH_BINARY);
//		///*��ʴ����*/
//		Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
//		//Mat dstImage;
//		//erode(binaryImagecricle, dstImage, element);
//		
//		Mat Channels[3];
//		split(image, Channels);
//		Mat colorimage = Channels[2] - Channels[0];//��ɫ
//		Mat dstImage, binaryImage;
//		threshold(colorimage, dstImage, 110, 255, THRESH_BINARY);
//		threshold(colorimage, binaryImage, 110, 255, THRESH_BINARY);
//
//		///////////////////////////////////////////////////////////////////////////////////////////////////////////
//		/*�ҵ�Բ���˶���Բ�ġ���R*/
//		state = retain; //state �Ƿ�ƫ��  ����
//		vector<vector<Point>> outlines;//����
//		vector<Vec4i> hierarchies;
//		findContours(dstImage, outlines, hierarchies, RETR_TREE, CHAIN_APPROX_NONE);
//		//for (int t = 0; t < outlines.size(); t++) {//Բ�Ŀ�����û����������û�и�������
//		//	drawContours(image, outlines, t, Scalar(0, 255, 0), 2, 8);//t��ָ��Ҫ���Ƶ����������� ����-1 ������������
//		//	imshow("image", image);
//		//	waitKey(0);
//		//}
//
//		for (int i = 0; i < outlines.size(); i++) {
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
//
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
//		}
//
//		/*��ֹminId���ڷ�Χ�ڱ���*/
//		if (minId >= 0 && minId < outlines.size())
//		{
//			/*�����Բ���ҵ�Բ��*/
//			minEnclosingCircle(Mat(outlines[minId]), newcenter, newradius); //����õ�����С���Բ����������Ͱ뾶
//			/*��С���������*/
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
//		}///////////////////////////////////////////////Բ�������������ķ�ʽ�ҵ���
//
//		/*���Ͳ���*/
//		element = getStructuringElement(0, Size(3, 3));
//		//Mat dilateImage;
//		/*dilate���һ�����������ʹ���*/
//		//dilate(binaryImage, dilateImage, element, Point(-1, -1), 2);
//		/*��������*/
//		vector<vector<Point>> contours;
//		vector<Vec4i> hierarchy;
//		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
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
//		}
//
//		if (maxId >= 0 && maxId < contours.size()) {
//			/*�����*/
//			Moments rect;
//			rect = moments(contours[maxId], false);
//			/*�������ľ�:*/
//			Point2f rectmid;
//			rectmid = Point2f(rect.m10 / rect.m00, rect.m01 / rect.m00);
//			/*�������λ����*/
//			drawContours(test, contours, maxId, Scalar(0, 255, 255), 1, 8);
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
//
//			/*����������λ���ĵ�*/
//			circle(test, rectMid, 1, Scalar(0, 255, 255), -1, 8, 0);
//			/*2��1����������λ,���*/
//			/*��һ����*/
//			if (rectMid.x >= center.x && rectMid.y <= center.y) {
//				target = Point2f(center.x + (rectMid.x - center.x) * multiple, center.y - (center.y - rectMid.y) * multiple);
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
//		}
//		else {
//			//continue;
//		}
//
//
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
//		/*Ԥ��*/
//		if (runNumber > pointSetnumeber) {
//
//			int i = pointNumber - 1;//ȡ���µĵ����ٶ�
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
//			/*ȡ����ĵ㣬���ٶȣ�����ٶ�*/
//			speed = distance(pointSet[number1], pointSet[number2]) * frame;
//			speedSet[0] = speed;
//			speed = distance(pointSet[number3], pointSet[number4]) * frame;
//			speedSet[1] = speed;
//			acceleratedSpeed = fabs((speedSet[0] - speedSet[1]) * frame);
//
//			/* X = V0T + 1 / 2AT'2��ͨ�����빫ʽ����Ԥ��Ĵ������� */  //������
//			predictdistance = (4.5 * speedSet[0] / frame) + 1 / 2 * acceleratedSpeed / frame / frame * 18;
//
//			/*���Ԥ��ʱx, y������ֵ�ı�ֵ*/
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
//		imshow("2", binaryImage);
//		runNumber++;
//		clock_t end = clock();
//		frame = 1 / (double)(end - start) * CLOCKS_PER_SEC;
//		cout << frame << "֡" << endl;
//		waitKey(1);
//	}
//		return;
//}


//�������˲�
//void Game::test2() {
//	VideoCapture cap("F:/rm/������Ԥ��/Windmills-master/wind.mp4");
//	Mat image, binary;
//	int stateNum = 4;
//	int measureNum = 2;
//
//	KalmanFilter KF(stateNum, measureNum, 0);
//	//Mat processNoise(stateNum, 1, CV_32F);
//	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
//	KF.transitionMatrix = (Mat_<float>(stateNum, stateNum) << 1, 0, 1, 0,//A ״̬ת�ƾ���
//		0, 1, 0, 1,
//		0, 0, 1, 0,
//		0, 0, 0, 1);
//
//	//����û�����ÿ��ƾ���B��Ĭ��Ϊ��
//	setIdentity(KF.measurementMatrix);//H=[1,0,0,0;0,1,0,0] ��������
//	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));//Q��˹����������λ��
//	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//R��˹����������λ��
//	setIdentity(KF.errorCovPost, Scalar::all(1));//P����������Э������󣬳�ʼ��Ϊ��λ��
//	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));//��ʼ��״̬Ϊ���ֵ
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
//		threshold(image, image, 80, 255, THRESH_BINARY);        //��ֵҪ�Լ���
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
//				if (middle > 60) {//�������ҲҪ����ʵ�������,��ͼ��ߴ������Զ���йء�
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


