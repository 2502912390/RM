
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

void Game::Q1p() {//
	//���ֻ��ͨ���������������С�Ա�����������ɸѡ �ڱ����л���ݽ���ԶСԭ�� �ڿ���װ�װ������ʱ���Ƿ���Ӧ��
	//�����ͨ���������ĳ��������ɸѡ�Ļ���С�����ʹ���������������ƺ����
	//�����ͨ��ɫ�ʵ���ֵ��ɸѡ����С����������ֵ�ܽӽ����޷�ʵ��  //pass1


	//�Ƚ��г�ɸѡ �������С��2 ����10���� ɸѡ�� �ٽ�ʣ��Ľ��оͽ���ϳɾ��� �ٶԾ��εĳ���Ƚ���ɸѡ 
	//���ȷ������ƥ��Ĺ�����  ͨ����������ĵ�������жϣ��������ڱ����в����ðɣ� ����ԶС ��������ĳһ��̶������ɣ���
	//�����С���ƣ� ����
	//���ĵ�������ꣿ �ܽ��ƥ�䵽����һ������Ĺ������ε����� ���ǲ��ܽ����ͬһ����ľ���ƥ������  ������ �ҽǶȲ��
	//�������ϵ�֪��������ת����ĽǶ���ƥ��

	Mat frame = imread("./images/zj.png");
	Mat channels[3], binary, Gaussian;

	vector<vector<Point>> contours;//������
	vector<Vec4i> hierarchy;
	RotatedRect AreaRect;//�洢��С(��ת)��Ӿ���

	split(frame, channels);
	//Mat colorimage = channels[0] - channels[2];

	threshold(channels[0], binary, 180, 255, 0);//��b���ж�ֵ��
	GaussianBlur(binary, Gaussian, Size(3, 3), 0);//sizeԽСԽ��
	findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	RotatedRect point_array[20];//��ɸѡ�����ת����
	Point2f points[4];//��RotatedRectת�����ĵ�����

	int index = 0;
	for (int i = 0; i < contours.size(); i++) {
		//boundRect = boundingRect(Mat(contours[i]));//������С��Ӿ���
		AreaRect = minAreaRect(Mat(contours[i]));//������С����ת���� ��ΪҪʹ��angle

		double width = AreaRect.size.width;//��
		double heigth = AreaRect.size.height;//��
		double area = AreaRect.size.area();//���

		//�ж�������
		if (area > 100) { //�Ȱ�һЩ̫С�ĸ�������ɸѡɾ��
			point_array[index] = AreaRect;
			AreaRect.points(points);

			//rectangle(frame, point_array[index].tl(), point_array[index].br(), Scalar(255, 255, 255), 2, 8, 0);//���� ����
			for (int i = 0; i < 4; i++) {
				line(frame, points[i % 4], points[(i + 1) % 4], Scalar(255, 255, 255), 2);
			}
			index++;
		}
	}

	imshow("frame", frame);
	waitKey(0);

	//for (int t = 0; t < contours.size(); t++) {
	//	drawContours(frame, contours, t, Scalar(255, 255, 255), 2, 8);//t��ָ��Ҫ���Ƶ����������� ����-1 ������������
	//	imshow("frame", frame);
	//	waitKey(0);
	//}
}

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
	//VideoWriter writer("./images/zjbb.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);//����ͼ�� ���ڲ鿴����ʶ��Ч������

	Mat channels[3], binary, Gaussian;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Point prePoint;//��һ�δ����
	Point hitPoint;//��ǰ�����

	namedWindow("video", WINDOW_NORMAL);

	Rect boundRect;//�洢��С��Ӿ���
	while (true)
	{
		Rect point_array[50];//�洢�ϸ����Ӿ���
		if (!video.read(frame)) {
			break;
		}
		split(frame, channels);
		threshold(channels[t], binary, 180, 255, 0);//��r���ж�ֵ��

		GaussianBlur(binary, Gaussian, Size(5, 5), 0);
		//Mat k = getStructuringElement(1, Size(3, 3));
		findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

		int index = 0;
		for (int i = 0; i < contours.size(); i++) {
			//box = minAreaRect(Mat(contours[i]));//������ת����
			//box.points(boxPts.data());
			boundRect = boundingRect(Mat(contours[i]));//������С��Ӿ���
			rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//���� ����

			try
			{
				//��������  ԭ��û��*1.0 ������һ������ ����Ϊ1.x�ľ��α���ȥĩβ��Ϊ1  С��1.5  ������Ч���������˵�
				if (double(boundRect.height * 1.0 / boundRect.width) > 1.5 && boundRect.width > 10 && boundRect.height > 20) {//���������ľ��󱣴���point_array
					point_array[index] = boundRect;
					index++;
					//cout << "����ı�����" << double(boundRect.height*1.0 / boundRect.width) << " ��" << boundRect.width <<" �ߣ�"<< boundRect.height << endl;
				}
				//else {//����������������������� ����test
				//	//cout << "������ı�����" << double(boundRect.height*1.0 / boundRect.width) << " ��" << boundRect.width << " �ߣ�" << boundRect.height << endl;
				//}
			}
			catch (const char* msg)
			{
				cout << printf(msg) << endl;
				continue;
			}
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

			hitPoint.x = (point3.x - point1.x) / 2 + point1.x;//�������ĵ�
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

//��ɫ��������ʶ��
bool colorr = false; //false ��ɫ  true��ɫ
void Game::Q2r() {
	//���⣺
	//1.ʶ����ɫ��Ч��ԶԶ�����ɫ  ͨ�����ӻ���ֵ��ͼ������ֵЧ�������� 
	//2.��ɫЧ��Ҳ����ˣ����� ԭ��˳�ֵ�����ֵ������

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
		//

		vector<vector<Point>>contours;//��������
		vector<Vec4i>hierarchy; //һ������
		Point2i center; //��������ҵ���Ŀ�����������
		//��ȡ����������������״�����ṹ
		findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

		//��¼ÿ���������������ĸ��� ���������Ϊ3��ΪĿ����
		int contour[30] = { 0 };
		for (int i = 0; i < contours.size(); i++)//����������������
		{
			if (hierarchy[i][3] != -1) //�и����� 
			{
				contour[hierarchy[i][3]]++; //�Ըø��������м�¼   Ŀ����Ѱ�Ҹ�����Ϊ3������ 
			}
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
				Point2f center;
				float radius;
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

		double endTime = cv::getTickCount();
		double totalTime = (endTime - startTime) / cv::getTickFrequency();

		//writer1.write(image);
		//writer2.write(binaryImage);
		imshow("image", image);
		imshow("binaryImage", binaryImage);
		cout << totalTime << endl;//ʶ���ɫ��Ч��ԶԶ����ʶ����ɫ��Ч�ʣ�����
		//imshow("mask", mask);
		waitKey(1);
	}

	cout << "finish" << endl;
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

//picture test
void Game::Q2() {
	Mat image = imread("./images/dblue2.png");

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

	Mat binaryImage;
	Mat Channels[3];
	split(image, Channels);
	Mat colorimage = Channels[0] - Channels[1];

	threshold(colorimage, binaryImage, 70, 255, THRESH_BINARY);
	//Mat k = getStructuringElement(1, Size(3, 3));
	//morphologyEx(binaryImage, binaryImage, 3, k,Point(-1,-1),2); //pass
	//morphologyEx(binaryImage, binaryImage, 0, k, Point(-1, -1));

	vector<vector<Point>>contours;//��������
	vector<Vec4i>hierarchy; //һ������
	Point2i center; //��������ҵ���Ŀ�����������
	//��ȡ����������������״�����ṹ
	findContours(binaryImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

	for (int t = 0; t < contours.size(); t++) {
		drawContours(image, contours, t, Scalar(0, 255, 0), 2, 8);//t��ָ��Ҫ���Ƶ����������� ����-1 ������������
		imshow("image", image);
		waitKey(0);
	}

	//Ѱ������������������� ����Ҫ�������Ҷ
	int contour[30] = { 0 };
	for (int i = 0; i < contours.size(); i++)//����������������
	{
		if (hierarchy[i][3] != -1) //�и����� 
		{
			contour[hierarchy[i][3]]++; //�Ըø��������м�¼  Ŀ����Ѱ�Ҹ�����Ϊ3������ 
		}
	}


	for (int i = 0; i < 30; i++) {
		if (contour[i] >= 5) {//Ŀ����Ҷ
			drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8);
			imshow("image", image);//����Ԥ������
			waitKey(0);

			////���ҳ�Ŀ����Ҷ������������ ������С��Χ ʵ�־�ȷ���
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
	//Mat blueImage = Channels.at(0) - Channels.at(2);//��ɫ
	waitKey(0);
}




//���ڲο�ɸѡ����
//vector<vector<Point>> contours;
//vector<Vec4i> hierarchy;
//double maxArea = -1;
//int maxId;
//findContours(dilateImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
//for (int i = 0; i < contours.size(); i++) {
//	vector<Point>points;
//	double area = contourArea(contours[i]);
//	/*����ų�����*/
//	if (area < 20 || area>10000)
//		continue;
//	/*�ҵ�û�и�����������*/
//	if (hierarchy[i][3] >= 0 && hierarchy[i][3] < contours.size())
//		continue;
//	/*��û��������*/
//	if (hierarchy[i][2] >= 0 && hierarchy[i][2] < contours.size())
//		continue;
//	/*�������������*/
//	if (maxArea <= area)
//	{
//		maxArea = area;
//		maxId = i;
//	}
//	/*������Χ*/
//	if (area <= maxArea + 50 && area >= maxArea - 50) {
//		maxArea = area;
//		maxId = i;
//	}
//}
//if (maxId >= 0 && maxId < contours.size()) {
//	/*�������λ����*/
//	drawContours(test, contours, maxId, Scalar(0, 255, 255), 1, 8);
//}








//��bar�����
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
//	// ��ȡ�������ĵ�ǰֵ
//	cout << red_value<< " " << green_value<<" " << blue_value << endl;
//	// ����ͼ�������ͨ��������ֵ
//	dst = Scalar(red_value, green_value, blue_value);
//	//imshow("dst", dst);//û����
//
//	Mat add(img.size(), img.type());//��Ȩ���
//	addWeighted(dst, 0.5, img, 0.5, 0, add);
//	//imshow("add", add);
//
//	Mat imgc = img.clone();
//	add.copyTo(imgc, mask);//ԭ�ȵķ�����ı�img��ԭʼͼ��
//	imshow("Image", imgc);
//}
//
//void Game::Dog() {
//	img = imread("./images/dog.jpg");
//	Mat channels[3];
//	mask = Mat(img.size(),img.type());
//	//����
//
//	split(img, channels);
//	threshold(channels[0], mask, 150, 255, THRESH_BINARY);//��ȡҪ��ɫ��ͼ��
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
//	//��dst���б�ɫ �ٸ�ֵ��ԭͼ��
//	createTrackbar("Red", "Image", &r, 255, onTrackbar);
//	createTrackbar("Green", "Image", &b, 255, onTrackbar);
//	createTrackbar("Blue", "Image", &g, 255, onTrackbar);
//
//	waitKey(0);
//}

