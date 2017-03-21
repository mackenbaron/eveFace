#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include  <opencv2\opencv.hpp>
#include "face_detection.h"
#include "face_identification.h"
#include "common.h"
#include "facedetect-dll.h"
#include <thread>
#include <mutex>
#include <vector>

#pragma comment(lib,"libfacedetect.lib")



using namespace std;
using namespace seeta;

mutex m;


struct feats
{
  
	float * feat;  // �ֶ������ڴ�
	string name;

};

struct faces
{
	cv::Mat face;    // ����
	string name;
	cv::Rect faceRect;  // λ��
	float *feat; // �ֶ������ڴ�
	void operator =(faces &t)
	{
		t.face.copyTo(this->face);
		this->faceRect.x = t.faceRect.x;
		this->faceRect.y = t.faceRect.y;
		this->faceRect.width = t.faceRect.width;
		this->faceRect.height= t.faceRect.height;
	}

};


cv::VideoCapture videoCap(0);

cv::Mat img,img_gray;

faces  face_temp;

FaceIdentification *face_recognizer;

seeta::ImageData img_data;

cv::Rect face_rect;

int * pResults = NULL;

short * p;

int faceSize = 0;

vector<faces> face_s;
vector<feats> feat_s;

int Flag_CMD_AddFace = 0;
int Flag_CMD_DelFace = 0;
int Flag_processing = 0; // ����ʶ�����ڴ�����
int Flag_faceReading = 0; // �����ɼ���
int exitFlag = 0;
int x, y, w, h;

void face_detection()  // �������
{
	while (!exitFlag)
	{
		videoCap >> img;

		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

		cv::rectangle(img, cv::Rect(10, 10, 200, 60), CV_RGB(20, 20, 20), -1);  // ����������ʾ����

		cv::rectangle(img, cv::Rect(10, 10, 200, 60), CV_RGB(200, 200, 250), 2);

		pResults = facedetect_frontal_tmp((unsigned char*)(img_gray.ptr(0)), img_gray.cols, img_gray.rows, img_gray.step, 1.2f, 5, 24);

		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{ /*-------------- read face -----------------*/
			if (Flag_CMD_AddFace)
			{
				if (*pResults>1)
				{
					cv::putText(img, "too much faces!", cv::Point(15, 30), 1, 1, cv::Scalar(150, 255, 255));
					break;
				}
				else
				{
					cv::putText(img, "add face", cv::Point(15, 30), 1, 1, cv::Scalar(150, 255, 255));
				}
			}
			if (Flag_CMD_DelFace)
			{
				cv::putText(img, "delete face", cv::Point(15, 30), 1, 1, cv::Scalar(150, 255, 255));
			}

			 p = ((short*)(pResults + 1)) + 6 * i;  // ��ȡ������Ϣ

			 x = p[0];
			 y = p[1];
			 w = p[2];
			 h = p[3];

			if (!((x + w)>630 || x<5 || (y + h)>470 || y < 5)) // ��ֹ����Խ��
			{

				face_rect.x = x; face_rect.y = y; face_rect.width = w; face_rect.height = h;

				cv::rectangle(img, face_rect, CV_RGB(0, 100, 200), 2, 8, 0);  // ��������


				if (!Flag_processing) // ���û���ڽ���ʶ����
				{
					
					img(face_rect).copyTo(face_temp.face);
					face_temp.faceRect.x = x;
					face_temp.faceRect.y = y;
					face_temp.faceRect.width = w;
					face_temp.faceRect.height = h;

					faces face;
					face.feat = new float[2048];
					face = face_temp;

					m.lock();
					face_s.push_back(face);
					faceSize = face_s.size();
					m.unlock();
				}

			}
			
			m.lock();
			if (!Flag_faceReading) 
				Flag_faceReading++;
			m.unlock();

		/*----------------------------------------*/
		}
		m.lock();
		if (Flag_faceReading > 0) // �����������
			Flag_faceReading = 2000;  // ����ʶ���̴߳�������
		m.unlock();

		cv::imshow("cap", img); // ��ʾÿһ֡

		cv::waitKey(10);
	}
}


void face_recognize_1()  // ����ʶ��
{
	ImageData faceData;
	float max_feat;
	float temp_feat;

	while (!exitFlag)
	{


		if (Flag_faceReading == 2000 && !Flag_CMD_AddFace&&!Flag_CMD_DelFace) // ���������ȡ��ɲ���û���ڽ���addface��delface
		{


			m.lock();
			Flag_processing = 1; // �������ڴ����־
			m.unlock();

			for (int i = 0; i < face_s.size(); ++i) // ����ÿһ������
			{


				cv::resize(face_s[i].face, face_s[i].face, cv::Size(face_recognizer->crop_height(), face_recognizer->crop_width()));


				faceData.width = face_s[i].face.cols;
				faceData.height = face_s[i].face.rows;
				faceData.num_channels = face_s[i].face.channels();
				faceData.data = face_s[i].face.data;

				face_recognizer->ExtractFeature(faceData, face_s[i].feat); // ����������ȡ������face_s��

				//max_feat = 0;

				for (int j = 0; j < feat_s.size(); ++j) // ��ÿһ���������е��������бȶ�
				{
					temp_feat = face_recognizer->CalcSimilarity(face_s[i].feat, feat_s[j].feat); // �����������������ƶ�

					if (temp_feat > 0.6)
					{
					
						face_s[i].name = feat_s[j].name;
						cv::putText(img, face_s[i].name, cv::Point(face_s[i].faceRect.x, face_s[i].faceRect.y), 1, 1.5, cv::Scalar(100, 200, 255),2);
						cv::putText(img, "warinig", cv::Point(15, 30), 1, 1, cv::Scalar(150, 255, 255));
						break;
					}
				}

			}


			face_s.clear();

			m.lock();
			Flag_processing = 0;
			Flag_faceReading = 0;
			m.unlock();

		}



	}
}


std::string cmd;

/**
 *  addface  //��������
 *  delface  //ɾ������
 *  exit // �˳�
 *
 *
 */
void console()
{
	while (!exitFlag)
	{
		cout << ">>";
		cin >> cmd;
		if (cmd == "addface")
		{
			if (!Flag_CMD_DelFace && !Flag_CMD_AddFace && face_s.size()>0)
			{
				Flag_CMD_AddFace = 1;

				feats feat;
				feat.feat = new float[2048];

				cout << "��������(��ǰ��֧������):\n";
				cin >> feat.name;

				m.lock();

				cv::resize(face_s[0].face, face_s[0].face, cv::Size(face_recognizer->crop_height(), face_recognizer->crop_width()));

				ImageData faceData(face_s[0].face.cols, face_s[0].face.rows, face_s[0].face.channels());

				faceData.data = face_s[0].face.data;

				face_recognizer->ExtractFeature(faceData, feat.feat);

				feat_s.push_back(feat); // �����������

				face_s.clear();

				m.unlock();

				Flag_CMD_AddFace = 0;
				
			}
			else
			{
				cout << "��ǰæ�����Ժ�����" << endl;
			}
			cmd.clear();
		}
		else if (cmd == "delface")
		{
			if (!Flag_CMD_DelFace&&!Flag_CMD_AddFace)
			{
				Flag_CMD_DelFace = 1;
			}
			else
			{
			
				cout << "��ǰæ�����Ժ�����" << endl;
			}
			cmd.clear();
		}
		else if (cmd == "exit")
		{
			exitFlag = 1;
		}
		else if (cmd == "cls")
		{
			system("cls");
		}
		else if (cmd == "help")
		{
			cout << "======================================\nʹ�÷���:\n addface: ���������������"\
				"\n delface: ��������ɾ������"\
				"\n exit: �˳�����"\
				"\n cls: ������� \n======================================" << endl;
			
		}
		else
		{
			cout << "��Ч�����ʹ��help��ȡ����" << endl;
		}
		cmd.clear();
	}

}

int main()
{
	
	face_recognizer = new FaceIdentification("seeta_fr_v1.0.bin");

	cout << "======================================\nʹ�÷���:\n addface: ���������������"\
		"\n delface: ��������ɾ������"\
		"\n exit: �˳�����"\
		"\n cls: ������� \n======================================" << endl;

	thread detec(face_detection);
	thread recog_1(face_recognize_1);


	thread con(console);

	con.join();
	detec.join();
	recog_1.join();



	delete face_recognizer;

	return 0;

	

}

