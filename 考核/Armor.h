#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef struct MyRect {
	int Id;//数字
	bool Color;//颜色
	Point2f center;//中心坐标  
	double hight;//高
	double width;//宽
}MyRect;

class Armor {
public:
	MyRect re;
	double Central_Point();//计算装甲板中心坐标
	double Diagonal();//计算装甲板对角线长度(保留两位小数))
	void Armor_Point();//输出装甲板4点坐标的函数 左上角坐标开始顺时针输出
	void Armor_Colour();//输出装甲板颜色的功能
};
