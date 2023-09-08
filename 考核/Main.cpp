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
////导向滤波，用来优化t(x)，针对单通道
//Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps)
//{
//	//转换源图像信息
//	srcImage.convertTo(srcImage, CV_32FC1, 1 / 255.0);
//	srcClone.convertTo(srcClone, CV_32FC1);
//	int nRows = srcImage.rows;
//	int nCols = srcImage.cols;
//	Mat boxResult;
//	//步骤一：计算均值
//	boxFilter(Mat::ones(nRows, nCols, srcImage.type()),
//		boxResult, CV_32FC1, Size(r, r));
//	//生成导向均值mean_I
//	Mat mean_I;
//	boxFilter(srcImage, mean_I, CV_32FC1, Size(r, r));
//	//生成原始均值mean_p
//	Mat mean_p;
//	boxFilter(srcClone, mean_p, CV_32FC1, Size(r, r));
//	//生成互相关均值mean_Ip
//	Mat mean_Ip;
//	boxFilter(srcImage.mul(srcClone), mean_Ip,
//		CV_32FC1, Size(r, r));
//	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
//	//生成自相关均值mean_II
//	Mat mean_II;
//	//应用盒滤波器计算相关的值
//	boxFilter(srcImage.mul(srcImage), mean_II,
//		CV_32FC1, Size(r, r));
//	//步骤二：计算相关系数
//	Mat var_I = mean_II - mean_I.mul(mean_I);
//	Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
//	//步骤三：计算参数系数a,b
//	Mat a = cov_Ip / (var_I + eps);
//	Mat b = mean_p - a.mul(mean_I);
//	//步骤四：计算系数a\b的均值
//	Mat mean_a;
//	boxFilter(a, mean_a, CV_32FC1, Size(r, r));
//	mean_a = mean_a / boxResult;
//	Mat mean_b;
//	boxFilter(b, mean_b, CV_32FC1, Size(r, r));
//	mean_b = mean_b / boxResult;
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
//	
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
////导向滤波器
//
//int main()
//{
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
//	namedWindow("去雾后的图像", 0);
//	namedWindow("原始图像", 0);
//	imshow("原始图像", src);
//	imshow("去雾后的图像", haze_removal_image);
//	waitKey(0);
//	return 0;
//}

