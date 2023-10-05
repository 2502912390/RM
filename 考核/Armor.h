#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef struct MyRect {
	int Id;//数字
	bool Color;//颜色
	Point2f Center;//中心坐标  
	double Hight;//高
	double Width;//宽
}MyRect;

class Armor {
private:
	MyRect re;//设置为私有

public:
	Armor() {//构造函数 
		re.Id = 0;
		re.Color = true;
		re.Center.x = 0;
		re.Center.y = 0;
		re.Hight = 0.0;
		re.Width = 0.0;
	}
	~Armor() {//析构函数 因为没有开辟到堆区数据 默认即可

	}
		
	void setId(int id);//设置id
	void setColor(bool color);//设置颜色
	void setHight(double hight);//设置高度
	void setWidth(double width);//设置宽度
	void Central_Point(Point2f Left);//设置装甲板中心

	Point2f getCenter();//获取装甲板中心
	double Diagonal();//获取装甲板对角线长度(保留两位小数))
	int getId();//获取ID

	void Armor_Colour();//输出装甲板颜色的功能
	void Armor_Point();//输出装甲板4点坐标的函数 左上角坐标开始顺时针输出
};
