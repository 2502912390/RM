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

void Armor::setId(int id) {//����Id
	re.Id = id;
}
void Armor::setColor(bool color) {//������ɫ
	re.Color = color;
}
void Armor::setHight(double height) {//���ø�
	re.Hight = height;
}
void Armor::setWidth(double width) {//���ÿ�
	re.Width = width;
}
void Armor::Central_Point(Point2f Left) {//����װ�װ���������
	re.Center.x = Left.x + re.Width / 2;
	re.Center.y = Left.y + re.Hight / 2;
}

double Armor::Diagonal() {//��ȡװ�װ�Խ��߳���(������λС��))
	return sqrt(re.Hight * re.Hight + re.Width * re.Width);
}
Point2f Armor::getCenter() {//��ȡ���ĵ�
	return re.Center;
}
int Armor::getId() {//��ȡid
	return re.Id;
}

void Armor::Armor_Colour() {//���װ�װ���ɫ�Ĺ���
	if (re.Color == 0) {
		cout << "��ɫ����" << endl;
	}
	else {
		cout << "��ɫ����" << endl;
	}
}
void Armor::Armor_Point() {//���װ�װ�4������ĺ��� ���Ͻ����꿪ʼ˳ʱ�����
	cout << "(" << (int)(re.Center.x - re.Width / 2) << "," << (int)(re.Center.y - re.Hight / 2) << ")";//--
	cout << "(" << (int)(re.Center.x + re.Width / 2) << "," << (int)(re.Center.y - re.Hight / 2) << ")";//+-
	cout << "(" << (int)(re.Center.x + re.Width / 2) << "," << (int)(re.Center.y + re.Hight / 2) << ")";//++
	cout << "(" << (int)(re.Center.x - re.Width / 2) << "," << (int)(re.Center.y + re.Hight / 2) << ")";//-+
}



