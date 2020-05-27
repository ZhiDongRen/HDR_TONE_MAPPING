#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include<opencv2/photo.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/core/mat.hpp>
using namespace cv;
using namespace std;
void imagelog(Mat& image, Mat &dst) {
	Mat imageLog(image.size(), CV_32FC3);

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
		
			imageLog.at<Vec3f>(i, j)[0] = log(1 + image.at<Vec3b>(i, j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = log(1 + image.at<Vec3b>(i, j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = log(1 + image.at<Vec3b>(i, j)[2]);
		}
	}
	//normalize to 0~255  
	normalize(imageLog, imageLog, 0, 255, cv::NORM_MINMAX);  
	convertScaleAbs(imageLog, imageLog);

	dst = imageLog.clone();
	//imshow("imagelog", imageLog);
	//waitKey();

}
void laplace(Mat& image, Mat &dst) {
	//Mat imageEnhance;
	//Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	//filter2D(image, imageEnhance, CV_8UC3, kernel);
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(image, dst, -1, kernel);

	//imshow("Laplace ", imageEnhance);
	//waitKey();

}
void hist(Mat& image, Mat &dst) {
	Mat imageRGB[3];
	split(image, imageRGB);
	for (int i = 0; i < 3; i++)
	{
		equalizeHist(imageRGB[i], imageRGB[i]);
	}
	dst = image.clone();
	merge(imageRGB, 3, dst);
	//imshow("Histogram equalization", image);
	//waitKey();
}
void simplemap(Mat&src, Mat& dst)
{
	CV_Assert(src.data);
	CV_Assert(src.depth() != sizeof(float));

	
	float MaxLum = 255.0;
	float Divider, BiasP, NormalY, Interpol;
	             
	Divider = log10f(MaxLum + 1.0f);         
	BiasP = logf(0.85) / log(0.5);             
	
	float NewLum[256];

	for (int i = 0; i < 256; i++)    // Normal tone mapping of every pixel
	{
		NormalY = (i / 255.0) ;                                            //        divided by the world adaptation luminance Lwa
		Interpol = logf(2.0f + powf(NormalY / MaxLum, BiasP) * 8.0f);                //        论文公式（4）右半部分的除数
		NewLum[i] =  (log((i / 255.0) + 1.0f) / Interpol) / Divider*255.0f;                        //        论文公式（4）
	}

	dst = src.clone();

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			(*it)[0] =3*NewLum[((*it)[0])];
			(*it)[1] =3*NewLum[((*it)[1])] ;
			(*it)[2] = 3*NewLum[((*it)[2])];
		}

}
void MyGammaCorrection(Mat& src, Mat& dst, double fixGamma)
	{
		CV_Assert(src.data);
		CV_Assert(src.depth() != sizeof(uchar));
		unsigned char lut[256];
		for (int i = 0; i < 256; i++)
		{
			lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fixGamma) * 255.0f);
		}
		dst = src.clone();
		const int channels = dst.channels();
		switch (channels)
		{
		case 1:
		{
			MatIterator_<uchar> it, end;
			for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
				*it = lut[(*it)];

			break;
		}
		case 3:
		{
			MatIterator_<Vec3b> it, end;
			for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
			{

				(*it)[0] = lut[((*it)[0])];
				(*it)[1] = lut[((*it)[1])];
				(*it)[2] = lut[((*it)[2])];
			}

			break;
		}
		}
	}
Mat1f hsvmap(Mat1f m,  float exposure = 1) {
	Mat1f exp;
	cv::exp((-m) * exposure, exp);
	Mat1f hsvmap = 1.0f - exp;

	return hsvmap;
}
void exposure(Mat& hdr, Mat& dst) {
	Mat3f hsvcolor;
	cvtColor(hdr, hsvcolor, COLOR_RGB2HSV);

	Mat1f hsv[3];
	split(hsvcolor, hsv);

	hsv[2] = hsvmap(hsv[2], 10);

	merge(hsv, 3, hsvcolor);

	//Mat rgb;
	dst = hdr.clone();
	cvtColor(hsvcolor, dst, COLOR_HSV2RGB);

	//return rgb;
}

int main()
	{

		Mat hdr = imread("memorial.hdr", -1);
		
		
		Mat png;
		hdr.convertTo(png, CV_8U, 255);
		//imshow("HDR image", png);
		imwrite("new.png",png);

		Mat image = imread("new.png");
		Mat dst(image.rows, image.cols, image.type());
		simplemap(image, dst);
		normalize(dst,dst,0,255,cv::NORM_MINMAX);
		//imshow("simplemap without gammafix", dst);
		imwrite("new01_norm.png", dst);


		MyGammaCorrection(dst, dst, 1 / 2.2);
		//namedWindow("gammafix", WINDOW_AUTOSIZE);
		//imshow("gammafix", dst);
		//waitKey();
		imwrite("new02_gamafix.png", dst);

		
		imagelog(image, dst);
		imwrite("new03_im_log.png", dst);

		laplace(image, dst);
		imwrite("new04_im_laplace.png", dst);
	
		hist(image, dst);
		imwrite("new05_im_hist.png", dst);
		
		
		//imshow("exposure", exposure(hdr));
		//waitKey();
		exposure(hdr, dst);
		dst.convertTo(png, CV_8U, 255);
		imwrite("new07_hdr_exposure.png", png);

		}


		
	