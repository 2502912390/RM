#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef struct MyRect {
	int Id;//����
	bool Color;//��ɫ
	Point2f center;//��������  
	double hight;//��
	double width;//��
}MyRect;

class Armor {
public:
	MyRect re;
	double Central_Point();//����װ�װ���������
	double Diagonal();//����װ�װ�Խ��߳���(������λС��))
	void Armor_Point();//���װ�װ�4������ĺ��� ���Ͻ����꿪ʼ˳ʱ�����
	void Armor_Colour();//���װ�װ���ɫ�Ĺ���
};
