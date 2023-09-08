#include <opencv2/opencv.hpp>
#include <iostream>
#include <All.h>

using namespace std;
using namespace cv;

int main() {
	Base bs;
	//bs.Q3();

	App app;
	app.Q2();

	return 0;
}
















//#define FUNCTION_H   
//
//#include<opencv2/core/core.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
//#include<opencv2/highgui/highgui.hpp>
//#include<iostream>
//#include<map>
//
//using namespace std;
//using namespace cv;
////�����˲��������Ż�t(x)����Ե�ͨ��
//Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps)
//{
//	//ת��Դͼ����Ϣ
//	srcImage.convertTo(srcImage, CV_32FC1, 1 / 255.0);
//	srcClone.convertTo(srcClone, CV_32FC1);
//	int nRows = srcImage.rows;
//	int nCols = srcImage.cols;
//	Mat boxResult;
//	//����һ�������ֵ
//	boxFilter(Mat::ones(nRows, nCols, srcImage.type()),
//		boxResult, CV_32FC1, Size(r, r));
//	//���ɵ����ֵmean_I
//	Mat mean_I;
//	boxFilter(srcImage, mean_I, CV_32FC1, Size(r, r));
//	//����ԭʼ��ֵmean_p
//	Mat mean_p;
//	boxFilter(srcClone, mean_p, CV_32FC1, Size(r, r));
//	//���ɻ���ؾ�ֵmean_Ip
//	Mat mean_Ip;
//	boxFilter(srcImage.mul(srcClone), mean_Ip,
//		CV_32FC1, Size(r, r));
//	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
//	//��������ؾ�ֵmean_II
//	Mat mean_II;
//	//Ӧ�ú��˲���������ص�ֵ
//	boxFilter(srcImage.mul(srcImage), mean_II,
//		CV_32FC1, Size(r, r));
//	//��������������ϵ��
//	Mat var_I = mean_II - mean_I.mul(mean_I);
//	Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
//	//���������������ϵ��a,b
//	Mat a = cov_Ip / (var_I + eps);
//	Mat b = mean_p - a.mul(mean_I);
//	//�����ģ�����ϵ��a\b�ľ�ֵ
//	Mat mean_a;
//	boxFilter(a, mean_a, CV_32FC1, Size(r, r));
//	mean_a = mean_a / boxResult;
//	Mat mean_b;
//	boxFilter(b, mean_b, CV_32FC1, Size(r, r));
//	mean_b = mean_b / boxResult;
//	//�����壺�����������
//	Mat resultMat = mean_a.mul(srcImage) + mean_b;
//	return resultMat;
//}
//
////���㰵ͨ��ͼ����������ͨ����ɫͼ��
//Mat dark_channel(Mat src)//��ÿ��ͨ��ȡ��Сֵ������Сֵ�˲�
//{
//	int border = 7;//��չ�߽�
//	std::vector<cv::Mat> rgbChannels(3);
//	Mat min_mat(src.size(), CV_8UC1, Scalar(0)), min_mat_expansion;
//	
//	split(src, rgbChannels);//����ͨ������ÿ��ͨ����Сֵ�浽min_mat��
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
//	//��ͼ����б߽���չ���� ��Ϊ��Ҫ��������ͼ�������Сֵ�˲�
//	copyMakeBorder(min_mat, min_mat_expansion, border, border, border, border, BORDER_REPLICATE);
//
//	Mat dark_channel_mat(src.size(), CV_8UC1, Scalar(0));
//	for (int m = border; m < min_mat_expansion.rows - border; m++)//��������չ�߽�
//	{
//		for (int n = border; n < min_mat_expansion.cols - border; n++)
//		{
//			Mat imageROI;
//			int min_num = 256;
//			imageROI = min_mat_expansion(Rect(n - border, m - border, 2 * border + 1, 2 * border + 1));//��ÿ������չ�߽�����ص���С�˲� ��СΪ 2 * border + 1
//			for (int i = 0; i < imageROI.rows; i++)
//			{
//				for (int j = 0; j < imageROI.cols; j++)
//				{
//					int val_roi = imageROI.at<uchar>(i, j);
//					min_num = std::min(min_num, val_roi);
//				}
//			}
//			dark_channel_mat.at<uchar>(m - border, n - border) = min_num;//����Сֵ����dark_channel_mat
//		}
//	}
//	return dark_channel_mat;
//}
//
//
//int calculate_A(Mat src, Mat dark_channel_mat)//����A �Ӱ�ͨ����ѡȡ�������� ��Ӧ��ԭͼ���е�����
//{
//	map<int, Point> pair_data;//�棨ֵ���㣩��  �Լ���val��������
//	
//	//cout << dark_channel_mat.rows << " " << dark_channel_mat.cols << endl;
//	for (int i = 0; i < dark_channel_mat.rows; i++)//������ͨ��ͼ�񣬹�����ֵ���㣩��  ����Ͷ�Ӧ����ֵ��Ӧ����
//	{
//		for (int j = 0; j < dark_channel_mat.cols; j++)
//		{
//			int val = dark_channel_mat.at<uchar>(i, j);
//			Point pt;
//			pt.x = j;
//			pt.y = i;
//			pair_data.insert(make_pair(val, pt));
//			//cord.push_back(pt);//������for�����Ż� ����ģ���Ϊ����map�������Ƕ�val��������
//		}
//	}
//
//	vector<Point> cord;//�氵ͨ���ĵ�
//	map<int, Point>::iterator iter;//Դ�������ࣻ����
//	for (iter = pair_data.begin(); iter != pair_data.end(); iter++)//�õ������ķ�ʽ����pair_data �������ĵ����cord
//	{
//		//cout << iter->first << endl;
//		cord.push_back(iter->second);
//	}
//
//	std::vector<cv::Mat> rgbChannels(3);
//	split(src, rgbChannels);
//	int max_val = 0;
//
//	////��Դ�����д�  ԭ��Ӧ����Ѱ�Ұ�ͨ����ֵ���ĵ��Ӧ��ԭͼ��ĵ�  
//	//for (int m = 0; m < cord.size(); m++)// ���ѭ������Ѱ��ԭͼ�������ֵ�ĵ㣿
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
//	//�޸Ŀ�Դ����
//	Point Max = cord[cord.size() - 1];//��ͨ�����ֵ�ĵ�
//	int val_1, val_2, val_3;
//	val_1 = rgbChannels[0].at<uchar>(Max.y, Max.x);//��Ӧԭ��ͼ��ĵ�
//	val_2 = rgbChannels[1].at<uchar>(Max.y, Max.x);
//	val_3 = rgbChannels[2].at<uchar>(Max.y, Max.x);
//	max_val = max(val_1, max(val_2, val_3));//ѡȡ����ͨ�������ֵ����
//
//	return max_val;
//}
//
//
////Mat calculate_tx(Mat& src, int A, Mat& dark_channel_mat)//����t   ������
//Mat calculate_tx(int A, Mat& dark_channel_mat)//����ͼ��I���ô��룿  
//{
//	Mat dst;
//	Mat tx;
//	float dark_channel_num;
//	dark_channel_num = A / 255.0;//��A��һ��
//
//	dark_channel_mat.convertTo(dst, CV_32FC3, 1 / 255.0);//��ͨ����ͨ��ͼ����չ��3ͨ��  ÿ��ͨ����ֵ���
//	dst = dst / dark_channel_num;
//
//	tx = 1 - 0.95 * dst;
//
//	return tx;
//}
//
////Mat calculate_tx(Mat& src, int A)//����t 
////{
////	Mat tem = src / A;//  ԭͼ��/A
////	Mat dark=dark_channel(tem);//���㰵ͨ��
////
////	//cout << 1 - dark << endl;
////	return 1-dark;
////}
//
//
//Mat haze_removal_img(Mat& src, int A, Mat& tx)//������ͼ��ģ�� ����J
//{
//	
//	Mat result_img(src.rows, src.cols, CV_8UC3);
//	vector<Mat> srcChannels(3), resChannels(3);
//	split(src, srcChannels);//������ͼ���з�
//	split(result_img, resChannels);
//	
//	for (int i = 0; i < src.rows; i++)
//	{
//		for (int j = 0; j < src.cols; j++)
//		{
//			for (int m = 0; m < 3; m++)//����ÿ�����ص��ÿ��ͨ��
//			{
//				int value_num = srcChannels[m].at<uchar>(i, j);//ÿ��ͨ����ֵ
//				float max_t = tx.at<float>(i, j);//��Ӧ��������
//				if (max_t < 0.1) //��ֹ������Ϊ0��Ϊ����
//				{
//					max_t = 0.1;
//				}
//				resChannels[m].at<uchar>(i, j) = (value_num - A) / max_t + A;
//			}
//		}
//	}
//
//	
//	merge(resChannels, result_img);
//
//	return result_img;
//}
//
//
//#include<opencv2/core/core.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
//#include<opencv2/highgui/highgui.hpp>
//#include<iostream>
//#include<algorithm>
//#include<set>
//#include<map>
//
//using namespace std;
//using namespace cv;
////�����˲���
//
//int main()
//{
//	Mat src = imread("./images/haze.png");
//	Mat dst;
//	cvtColor(src, dst, COLOR_BGR2GRAY);
//	Mat dark_channel_mat = dark_channel(src);//���㰵ͨ��ͼ��
//
//	int A = calculate_A(src, dark_channel_mat);//֪��I����ͨ��ͼ�� ���Լ���A
//
//	Mat tx = calculate_tx(A, dark_channel_mat);//֪��I����ͨ��ͼ��A���Լ���t
//	//Mat tx = calculate_tx(src, A);
//
//	Mat tx_ = guidedfilter(dst, tx, 30, 0.001);//�����˲����tx��dstΪ����ͼ��  ���Ż�����
//
//	Mat haze_removal_image;
//	haze_removal_image = haze_removal_img(src, A, tx_);//����I��t��A����J
//	namedWindow("ȥ����ͼ��", 0);
//	namedWindow("ԭʼͼ��", 0);
//	imshow("ԭʼͼ��", src);
//	imshow("ȥ����ͼ��", haze_removal_image);
//	waitKey(0);
//	return 0;
//}

