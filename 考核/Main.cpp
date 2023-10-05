#include <opencv2/opencv.hpp>
#include <iostream>
#include <Base.h>
#include <Armor.h>
#include <Game.h>
#include <App.h>
#include <cmath>

using namespace std;
using namespace cv;

int main() {
	/********************运行说明***********************/
	/********************若想查看题目的效果只需实例化出一个类，调用不同的函数即可********************/
	//Base bs;
	//bs.Q1();//演示基础题的第一题
	//bs.Q3();//演示基础题的第三题  注意：每次最好只调用一个函数


	//Base bs;
	//bs.Q3();

	//App app;
	//app.Q3();

	Game gm;
	//gm.Q1p();
	gm.Dog();


	//C++应用题
	/****************将上面所有代码注释即可运行********************/
	//Armor armor;
	//Point2f left;//左上角
	//Point2f center;//中心
	//int id;
	//bool color;
	//double width;
	//double height;
	//double length;

	//cin >> id >> color;//输入数字ID和颜色
	//armor.setId(id);//设置id
	//armor.setColor(color);//设置颜色

	//cin >> left.x >> left.y >> width >> height;//输入左上角点坐标和宽与高
	//armor.setWidth(width);//设置宽
	//armor.setHight(height);//设置高
	//armor.Central_Point(left);//设置中心
	//
	//id = armor.getId();//获取id
	//cout << "ID: " << id;
	//armor.Armor_Colour();//输出数字ID和颜色

	//center = armor.getCenter();//获取中心
	//length = armor.Diagonal();//获取对角线长
	//cout << "(" << center.x << "," << center.y << ")" << "长度：" << fixed << setprecision(2) << length << endl;//输出中心点坐标和长度
	//armor.Armor_Point();//输出四个点坐标
	//return 0;
}
