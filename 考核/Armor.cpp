#include <opencv2/opencv.hpp>
#include <iostream>
#include <Armor.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<algorithm>
#include<set>
#include<map>
#include <cmath>

using namespace std;
using namespace cv;

void Armor::setId(int id) {//设置Id
	re.Id = id;
}
void Armor::setColor(bool color) {//设置颜色
	re.Color = color;
}
void Armor::setHight(double height) {//设置高
	re.Hight = height;
}
void Armor::setWidth(double width) {//设置宽
	re.Width = width;
}
void Armor::Central_Point(Point2f Left) {//设置装甲板中心坐标
	re.Center.x = Left.x + re.Width / 2;
	re.Center.y = Left.y + re.Hight / 2;
}

double Armor::Diagonal() {//获取装甲板对角线长度(保留两位小数))
	return sqrt(re.Hight * re.Hight + re.Width * re.Width);
}
Point2f Armor::getCenter() {//获取中心点
	return re.Center;
}
int Armor::getId() {//获取id
	return re.Id;
}

void Armor::Armor_Colour() {//输出装甲板颜色的功能
	if (re.Color == 0) {
		cout << "颜色：蓝" << endl;
	}
	else {
		cout << "颜色：红" << endl;
	}
}
void Armor::Armor_Point() {//输出装甲板4点坐标的函数 左上角坐标开始顺时针输出
	cout << "(" << (int)(re.Center.x - re.Width / 2) << "," << (int)(re.Center.y - re.Hight / 2) << ")";//--
	cout << "(" << (int)(re.Center.x + re.Width / 2) << "," << (int)(re.Center.y - re.Hight / 2) << ")";//+-
	cout << "(" << (int)(re.Center.x + re.Width / 2) << "," << (int)(re.Center.y + re.Hight / 2) << ")";//++
	cout << "(" << (int)(re.Center.x - re.Width / 2) << "," << (int)(re.Center.y + re.Hight / 2) << ")";//-+
}



