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
	VideoCapture* cap = (VideoCapture*)userData;//voidתVideoCapture
	double newExposure = exposure / 100.0;  // ��barֵӳ�䵽[0, 1]�ķ�Χ
	cap->set(CAP_PROP_EXPOSURE, newExposure);//�����ع��
}

void  Base::Q2() {
	VideoCapture capture(0);

	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ��֡��
	cout << "fps" << fps << " " << "��" << width << " " << "��" << height << " " << "��֡��" << numFrame << endl;

	VideoWriter writer("./images/output1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
	//					����·��		        ���ַ������ʽ                                    true��ʾ�������Ƶ�ļ��ǲ�ɫ��
	
	if (!capture.isOpened()) {  // �������ͷ�Ƿ���ȷ��
		cout << "Could not open the camera." << endl;
		return;
	}

	if (!writer.isOpened()) {
		cout << "error" << endl;
		return;
	}
	
	namedWindow("��ʾ��", WINDOW_AUTOSIZE);

	int light = 50;//��ʼֵ
	//createTrackbar("����", "��ʾ��", &light, 255, callBack, 0);//����ʾ�򴰿ڴ�������bar
	createTrackbar("����", "��ʾ��", &light, 100);//��ʹ�ûص����� ���ֵ����Ϊ100�����Ŵ���������
	
	int initialExposure = 50;  // ��ʼ�ع��ֵ
	createTrackbar("�ع�", "��ʾ��", &initialExposure, 100, onExposureChange, &capture);//ʹ�ûص����� //��bug�� ����ͷ���⣿����
	
	while (true)
	{
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
	Mat picture = imread("./images/bd.png");
	Size board_size = Size(10, 7);//�궨���ڽǵ���Ŀ���� �У�
	Mat gray1;
	cvtColor(picture, gray1, COLOR_BGR2GRAY);
	vector<Point2f> img1_points;
	findChessboardCorners(gray1, board_size, img1_points);//���ͼ�������̸�ģʽ�Ľǵ�
	find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));//�Գ�ʼ�Ľǵ�������������ؼ�����Ż�
	bool pattern = true;
	drawChessboardCorners(picture, board_size, img1_points, pattern);//���Ƽ�⵽�����̸�ǵ�
	imshow("�궨��ǵ���", picture);
	waitKey(0);
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
				temp.copyTo(image);//����ԭͼ����
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);

				//��ȡ��ǰ������Ϣ
				Vec3b rgb = image.at<Vec3b>(y, x);
				int r = static_cast <int> (rgb[2]);
				int g = static_cast <int> (rgb[1]);
				int b = static_cast <int> (rgb[0]);

				String s = "rgb: " + to_string(r) + "," + to_string(g) + "," + to_string(b) + " " + "position: " + to_string(y) + "," + to_string(x);
				putText(image, s, Point(x-dx, y-dy), 0, 0.5, Scalar(255,0,0), 1, 8, false);
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

/***************************************************APP***************************************************************/
void App::Q1() {
	//˼·��
	//һ��ƻ���Ǻ�ɫ ��rͨ��������ֵ��һ��ɸѡ  ��ɸѡ�Ľ������Ѱ������ ��ԭͼ�л���������Ч��һ�� ԭ��1.ƻ��������Ӱ 2.ƻ���ĵײ�ɫ�ʲ������� 3.��������rͨ����ֵ�Ƚϴ�
	
	//����ת��hsv�Ժ�ɫ���п�ͼ  �������һ������ ����ֱ���ų���յĸ��� ����ƻ���ײ����ɲ��ܺܺ�ʶ��

	Mat image=imread("./images/apple.png");

	//idea1
	vector<Mat> channels;
	split(image, channels);//����
	
	Mat r,binary;
	r = channels[2];
	GaussianBlur(r, r, Size(13, 13), 4, 4);
	threshold(r, binary,240 ,255, THRESH_BINARY);//��ֵ�� С����ֵ����0 ���ڵ�����255
	//adaptiveThreshold(r, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 0);����Ӧ��ֵ��

	Mat k = getStructuringElement(1, Size(3, 3));
	morphologyEx(binary, binary, 2, k, Point(-1, -1), 5);//������ 

	morphologyEx(binary, binary, 1, k, Point(-1, -1), 30);//����

	vector<vector<Point>> contours;//�洢��⵽����������������  ���ܻ��ж��
	vector<Vec4i> hierarchy;//�洢�����Ĳ�νṹ��Ϣ
	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());//�ڶ�������Ϊ��⵽������ 

	for (int t = 0; t < contours.size(); t++) {
		Rect rect = boundingRect(contours[t]);//��ȡ�����Ӿ���
		rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
		//drawContours(image, contours, t, Scalar(255, 0,0 ), 2, 8);//t��ָ��Ҫ���Ƶ����������� ����-1 ������������
		imshow("ԭͼ", image);
	}

	//idea2
	//Mat hsv,out;
	//cvtColor(image, hsv, COLOR_BGR2HSV);
	//inRange(hsv, Scalar(156, 43, 46), Scalar(180, 255, 255), out);
	
	waitKey(0);
}

/***********************ȥ����غ���***********************/
//�����˲��������Ż�t(x)����Ե�ͨ��
Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps)
{
	//ת��Դͼ����Ϣ
	srcImage.convertTo(srcImage, CV_32FC1, 1 / 255.0);
	srcClone.convertTo(srcClone, CV_32FC1);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	Mat boxResult;
	//����һ�������ֵ //Ӧ�ú��˲���������ص�ֵ
	boxFilter(Mat::ones(nRows, nCols, srcImage.type()),boxResult, CV_32FC1, Size(r, r));

	//���ɵ����ֵmean_I
	Mat mean_I;
	boxFilter(srcImage, mean_I, CV_32FC1, Size(r, r));

	//����ԭʼ��ֵmean_p
	Mat mean_p;
	boxFilter(srcClone, mean_p, CV_32FC1, Size(r, r));

	//���ɻ���ؾ�ֵmean_Ip
	Mat mean_Ip;
	boxFilter(srcImage.mul(srcClone), mean_Ip,CV_32FC1, Size(r, r));

	//��������ؾ�ֵmean_II
	Mat mean_II;
	boxFilter(srcImage.mul(srcImage), mean_II,CV_32FC1, Size(r, r));

	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//��������������ϵ��
	Mat var_I = mean_II - mean_I.mul(mean_I);
	Mat var_Ip = mean_Ip - mean_I.mul(mean_p);

	//���������������ϵ��a,b
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);

	//�����ģ�����ϵ��a\b�ľ�ֵ
	Mat mean_a;
	boxFilter(a, mean_a, CV_32FC1, Size(r, r));
	mean_a = mean_a / boxResult;
	Mat mean_b;
	boxFilter(b, mean_b, CV_32FC1, Size(r, r));
	mean_b = mean_b / boxResult;

	//�����壺�����������
	Mat resultMat = mean_a.mul(srcImage) + mean_b;
	return resultMat;
}

//���㰵ͨ��ͼ����������ͨ����ɫͼ��
Mat dark_channel(Mat src)//��ÿ��ͨ��ȡ��Сֵ������Сֵ�˲�
{
	int border = 7;//��չ�߽�
	std::vector<cv::Mat> rgbChannels(3);
	Mat min_mat(src.size(), CV_8UC1, Scalar(0)), min_mat_expansion;

	split(src, rgbChannels);//����ͨ������ÿ��ͨ����Сֵ�浽min_mat��
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

	//��ͼ����б߽���չ���� ��Ϊ��Ҫ��������ͼ�������Сֵ�˲�
	copyMakeBorder(min_mat, min_mat_expansion, border, border, border, border, BORDER_REPLICATE);

	Mat dark_channel_mat(src.size(), CV_8UC1, Scalar(0));
	for (int m = border; m < min_mat_expansion.rows - border; m++)//��������չ�߽�
	{
		for (int n = border; n < min_mat_expansion.cols - border; n++)
		{
			Mat imageROI;
			int min_num = 256;
			imageROI = min_mat_expansion(Rect(n - border, m - border, 2 * border + 1, 2 * border + 1));//��ÿ������չ�߽�����ص���С�˲� ��СΪ 2 * border + 1
			for (int i = 0; i < imageROI.rows; i++)
			{
				for (int j = 0; j < imageROI.cols; j++)
				{
					int val_roi = imageROI.at<uchar>(i, j);
					min_num = std::min(min_num, val_roi);
				}
			}
			dark_channel_mat.at<uchar>(m - border, n - border) = min_num;//����Сֵ����dark_channel_mat
		}
	}
	return dark_channel_mat;
}

int calculate_A(Mat src, Mat dark_channel_mat)//����A �Ӱ�ͨ����ѡȡ�������� ��Ӧ��ԭͼ���е�����
{
	map<int, Point> pair_data;//�棨ֵ���㣩��  �Լ���val��������

	//cout << dark_channel_mat.rows << " " << dark_channel_mat.cols << endl;
	for (int i = 0; i < dark_channel_mat.rows; i++)//������ͨ��ͼ�񣬹�����ֵ���㣩��  ����Ͷ�Ӧ����ֵ��Ӧ����
	{
		for (int j = 0; j < dark_channel_mat.cols; j++)
		{
			int val = dark_channel_mat.at<uchar>(i, j);
			Point pt;
			pt.x = j;
			pt.y = i;
			pair_data.insert(make_pair(val, pt));
			//cord.push_back(pt);//������for�����Ż� ����ģ���Ϊ����map�������Ƕ�val��������
		}
	}

	vector<Point> cord;//�氵ͨ���ĵ�
	map<int, Point>::iterator iter;//Դ�������ࣻ����
	for (iter = pair_data.begin(); iter != pair_data.end(); iter++)//�õ������ķ�ʽ����pair_data �������ĵ����cord
	{
		//cout << iter->first << endl;
		cord.push_back(iter->second);
	}

	std::vector<cv::Mat> rgbChannels(3);
	split(src, rgbChannels);
	int max_val = 0;

	////��Դ�����д�  ԭ��Ӧ����Ѱ�Ұ�ͨ����ֵ���ĵ��Ӧ��ԭͼ��ĵ�  
	//for (int m = 0; m < cord.size(); m++)// ���ѭ������Ѱ��ԭͼ�������ֵ�ĵ㣿
	//{
	//	Point tmp = cord[m];
	//	int val_1, val_2, val_3;
	//	val_1 = rgbChannels[0].at<uchar>(tmp.y, tmp.x);//
	//	val_2 = rgbChannels[1].at<uchar>(tmp.y, tmp.x);
	//	val_3 = rgbChannels[2].at<uchar>(tmp.y, tmp.x);

	//	max_val = std::max(val_1, val_2);
	//	max_val = std::max(max_val, val_3);
	//}

	//�޸Ŀ�Դ����
	Point Max = cord[cord.size() - 1];//��ͨ�����ֵ�ĵ�
	int val_1, val_2, val_3;
	val_1 = rgbChannels[0].at<uchar>(Max.y, Max.x);//��Ӧԭ��ͼ��ĵ�
	val_2 = rgbChannels[1].at<uchar>(Max.y, Max.x);
	val_3 = rgbChannels[2].at<uchar>(Max.y, Max.x);
	max_val = max(val_1, max(val_2, val_3));//ѡȡ����ͨ�������ֵ����

	return max_val;
}

//Mat calculate_tx(Mat& src, int A, Mat& dark_channel_mat)//����t   ������
Mat calculate_tx(int A, Mat& dark_channel_mat)//����ͼ��I���ô��룿  
{
	Mat dst;
	Mat tx;
	float dark_channel_num;
	dark_channel_num = A / 255.0;//��A��һ��

	dark_channel_mat.convertTo(dst, CV_32FC3, 1 / 255.0);//��ͨ����ͨ��ͼ����չ��3ͨ��  ÿ��ͨ����ֵ���
	dst = dst / dark_channel_num;

	tx = 1 - 0.95 * dst;

	return tx;
}

//Mat calculate_tx(Mat& src, int A)//����t 
//{
//	Mat tem = src / A;//  ԭͼ��/A
//	Mat dark=dark_channel(tem);//���㰵ͨ��
//
//	//cout << 1 - dark << endl;
//	return 1-dark;
//}


Mat haze_removal_img(Mat& src, int A, Mat& tx)//������ͼ��ģ�� ����J
{
	Mat result_img(src.rows, src.cols, CV_8UC3);
	vector<Mat> srcChannels(3), resChannels(3);
	split(src, srcChannels);//������ͼ���з�
	split(result_img, resChannels);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			for (int m = 0; m < 3; m++)//����ÿ�����ص��ÿ��ͨ��
			{
				int value_num = srcChannels[m].at<uchar>(i, j);//ÿ��ͨ����ֵ
				float max_t = tx.at<float>(i, j);//��Ӧ��������
				if (max_t < 0.1) //��ֹ������Ϊ0��Ϊ����
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

void App::Q2() {//ȥ��
	Mat src = imread("./images/haze.png");
	Mat dst;
	cvtColor(src, dst, COLOR_BGR2GRAY);
	Mat dark_channel_mat = dark_channel(src);//���㰵ͨ��ͼ��

	int A = calculate_A(src, dark_channel_mat);//֪��I����ͨ��ͼ�� ���Լ���A

	Mat tx = calculate_tx(A, dark_channel_mat);//֪��I����ͨ��ͼ��A���Լ���t
	//Mat tx = calculate_tx(src, A);

	Mat tx_ = guidedfilter(dst, tx, 30, 0.001);//�����˲����tx��dstΪ����ͼ��  ���Ż�����

	Mat haze_removal_image;
	haze_removal_image = haze_removal_img(src, A, tx_);//����I��t��A����J
	namedWindow("ȥ����ͼ��");
	namedWindow("ԭʼͼ��");
	imshow("ԭʼͼ��", src);
	imshow("ȥ����ͼ��", haze_removal_image);
	waitKey(0);
}

void App::Q3() {
	//˼·��ʹ��undistort���ԶԻ�����н�������Ҫ֪��cameraMatrix����distCoeffs  
	//ʹ��calibrateCamera������ȡ�ڲκͻ��������Ҫ֪����ʵ����ĵ�objectPoints���궨��ǵ�imgsPoints���궨ͼ��Ĵ�СimageSize
	//ʹ��findChessboardCorners���Լ��궨��ǵ㣬 objectPointsӦ����β�����

	//�ñ�����ȡ�ı궨����Ƭ��ȡ���Ǳ������ڲκͻ�����󣬶Կ������񷢲���ͼƬ�Ƿ��ܹ���Ч������
	//Mat img = imread("./images/ji.png");
	//Mat dst;
	//Mat cameraMatrix, distCoeffs;//�ڲκͻ���
	//undistort(img, dst, cameraMatrix, distCoeffs);   //pass

	//

}

void Game::Q1() {
	Mat img = imread("./images/zjb.png");
	Mat gray,binary;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	//Mat k = getStructuringElement(1, Size(3, 3));
	//morphologyEx(gray, gray, 2, k,Point(-1,-1),3);//��һЩϸС��Ե��ʴ

	GaussianBlur(gray, gray, Size(5, 5), 0, 0);
	threshold(gray, binary, 50, 255, THRESH_BINARY); 

	vector<vector<Point>> contours;//���� �洢��⵽���������������� �洢��⵽����������������
	vector<Vec4i> hierarchy;//�洢�����Ĳ�νṹ��Ϣ
	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());


	//��������
	for (int t = 0; t < contours.size(); t++) {
		drawContours(img, contours, t, Scalar(0, 0, 255), 2, 8);//t��ָ��Ҫ���Ƶ����������� ����-1 ������������
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
//const int color = 2;//����ѡ���ж�װ�װ����ɫ 0��ɫ   2��ɫ
//
//int main()
//{
//	//�������⣺
//	//1.װ�װ��ڱ�Ե��ֻ��ʶ��һ������
//	//2.��ɫװ�װ���ʱ��ʶ�𲻳�������
//
//	VideoCapture video;
//	video.open("./images/zjbb.mp4");
//	Mat frame;
//	Mat channels[3], binary, Gaussian;
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierarchy;
//
//	Rect boundRect;//�洢��С��Ӿ���
//	while (true)
//	{
//		Rect point_array[50];//�洢�ϸ����Ӿ���
//		if (!video.read(frame)) {
//			break;
//		}
//		split(frame, channels);
//
//		threshold(channels[2], binary, kThreashold, kMaxVal, 0);//��r/b���ж�ֵ��
//
//		GaussianBlur(binary, Gaussian, kGaussianBlueSize, 0);
//		Mat k = getStructuringElement(1, Size(3, 3));
//		findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//��������
//
//		int index = 0;
//		for (int i = 0; i < contours.size(); i++) {
//			//box = minAreaRect(Mat(contours[i]));//������ת����
//			//box.points(boxPts.data());
//			boundRect = boundingRect(Mat(contours[i]));//������С��Ӿ���
//			rectangle(frame, boundRect.tl(), boundRect.br(), (255, 255, 255), 2, 8, 0);//���� ����
//			try
//			{
//				if (double(boundRect.height / boundRect.width) >= 2 && boundRect.height > 36 && boundRect.height > 20) {//���������ľ��󱣴���point_array
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
//		if (index < 2) {//û��⵽�ϸ������ ֱ�� ��һ֡
//			imshow("video", frame);
//			cv::waitKey(10);
//			continue;
//		}
//
//		int point_near[2];
//		int min = 10000;
//		for (int i = 0; i < index - 1; i++)//�ҵ����֮����С�ĵ��������������� 
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
//			Rect rectangle_1 = point_array[point_near[0]];//�ҵ����������Ƶ�����
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