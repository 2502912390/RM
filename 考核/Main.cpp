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
	/********************����˵��***********************/
	/********************����鿴��Ŀ��Ч��ֻ��ʵ������һ���࣬���ò�ͬ�ĺ�������********************/
	//Base bs;
	//bs.Q1();//��ʾ������ĵ�һ��
	//bs.Q3();//��ʾ������ĵ�����  ע�⣺ÿ�����ֻ����һ������


	//Base bs;
	//bs.Q3();

	//App app;
	//app.Q3();

	Game gm;
	//gm.Q1p();
	gm.Dog();


	//C++Ӧ����
	/****************���������д���ע�ͼ�������********************/
	//Armor armor;
	//Point2f left;//���Ͻ�
	//Point2f center;//����
	//int id;
	//bool color;
	//double width;
	//double height;
	//double length;

	//cin >> id >> color;//��������ID����ɫ
	//armor.setId(id);//����id
	//armor.setColor(color);//������ɫ

	//cin >> left.x >> left.y >> width >> height;//�������Ͻǵ�����Ϳ����
	//armor.setWidth(width);//���ÿ�
	//armor.setHight(height);//���ø�
	//armor.Central_Point(left);//��������
	//
	//id = armor.getId();//��ȡid
	//cout << "ID: " << id;
	//armor.Armor_Colour();//�������ID����ɫ

	//center = armor.getCenter();//��ȡ����
	//length = armor.Diagonal();//��ȡ�Խ��߳�
	//cout << "(" << center.x << "," << center.y << ")" << "���ȣ�" << fixed << setprecision(2) << length << endl;//������ĵ�����ͳ���
	//armor.Armor_Point();//����ĸ�������
	//return 0;
}
