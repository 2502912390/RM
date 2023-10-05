#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef struct MyRect {
	int Id;//����
	bool Color;//��ɫ
	Point2f Center;//��������  
	double Hight;//��
	double Width;//��
}MyRect;

class Armor {
private:
	MyRect re;//����Ϊ˽��

public:
	Armor() {//���캯�� 
		re.Id = 0;
		re.Color = true;
		re.Center.x = 0;
		re.Center.y = 0;
		re.Hight = 0.0;
		re.Width = 0.0;
	}
	~Armor() {//�������� ��Ϊû�п��ٵ��������� Ĭ�ϼ���

	}
		
	void setId(int id);//����id
	void setColor(bool color);//������ɫ
	void setHight(double hight);//���ø߶�
	void setWidth(double width);//���ÿ��
	void Central_Point(Point2f Left);//����װ�װ�����

	Point2f getCenter();//��ȡװ�װ�����
	double Diagonal();//��ȡװ�װ�Խ��߳���(������λС��))
	int getId();//��ȡID

	void Armor_Colour();//���װ�װ���ɫ�Ĺ���
	void Armor_Point();//���װ�װ�4������ĺ��� ���Ͻ����꿪ʼ˳ʱ�����
};
