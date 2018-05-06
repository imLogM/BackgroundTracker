#include "background_alg.h"

background_alg::background_alg(int width = 640, int height = 480, int chan = 1, double resize_fac = 1,  double tresher=0.05){
	if (!this->initialize(width, height, chan, resize_fac, tresher)){ cout << "initialize failed" << endl; }
}

background_alg::~background_alg(){}

bool background_alg::initialize(int width, int height, int chan, double resize_fac, double tresher){
	bal_resize_fac = resize_fac;
	bal_extend = 40* bal_resize_fac;
	channels = chan;
	treshold = tresher;

	contour = 10;
    image_contour = Rect(0, 0, width, height);
	kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3)); 
	kernel_erode_outer = getStructuringElement(MORPH_RECT, Size(13, 13));

	if (channels == 1)
	{
		 W_old = Mat(height, width, CV_16UC1, Scalar(0));
		 Z_old = Mat(height, width, CV_16UC1, Scalar(1));
		 B_old = Mat(height, width, CV_16UC1, Scalar(0));

		 W = Mat(height, width, CV_16UC1, Scalar(1));
		 Z = Mat(height, width, CV_16UC1, Scalar(0));
		return true;
	}

	else if (channels == 3)
	{
		 W_old = Mat(height, width, CV_16UC3, Scalar(0,0,0));
		 Z_old = Mat(height, width, CV_16UC3, Scalar(1,1,1));
		 B_old = Mat(height, width, CV_16UC3, Scalar(0,0,0));

		 W = Mat(height, width, CV_16UC3, Scalar(1, 1, 1));
		 Z = Mat(height, width, CV_16UC3, Scalar(0, 0, 0));
		return true;
	}
	else{
		cout << "the Channel is neither 1 or 3 ,Please check1" << endl;
		return false;
	}

}

bool background_alg::background_isReady(){
	double MaxValue, MinValue;
	cv::minMaxLoc(Z_old, &MinValue,&MaxValue);
	if(MaxValue==0){
		return true;
	}else{
		return false;
	}
}

bool background_alg::background_update(Mat &src, Point topLef, Point botRig){
	Mat ROI_temp;
	Mat img16;
	old_Rectangle.x = topLef.x; old_Rectangle.y = topLef.y;
    old_Rectangle.width = botRig.x - topLef.x;
    old_Rectangle.height = botRig.y - topLef.y;
	topLef = Point(max(topLef.x - bal_extend, 0), max(topLef.y - bal_extend, 0));
	botRig = Point(min(botRig.x + bal_extend, src.cols - 1), min(botRig.y + bal_extend, src.rows - 1));

	if (channels == 1){
		W = Scalar(1);
		ROI_temp = W(Rect(topLef.x, topLef.y, (botRig.x - topLef.x + 1), (botRig.y - topLef.y + 1)));
		ROI_temp = Scalar(0);
		Z = Scalar(0);
		//cout << Rect(topLef.x, topLef.y, (botRig.x - topLef.x + 1), (botRig.y - topLef.y + 1)) << endl;
		ROI_temp = Z(Rect(topLef.x, topLef.y, (botRig.x - topLef.x + 1), (botRig.y - topLef.y + 1)));
		ROI_temp = Scalar(1);
		Z_old = Z&Z_old;

		Mat img_gray;
		cv::cvtColor(src, img_gray, CV_BGR2GRAY);
		img_gray.convertTo(img16, CV_16UC1, 1, 0);

		B_old = (B_old.mul(W_old) + img16.mul(W)) / (W_old + W + Z_old);
		W_old = W_old + W;

		W_old.convertTo(W_old, CV_8UC1, 1, 0);
		W_old.convertTo(W_old, CV_16UC1, 1, 0);
		return true;
	}
	
	else if (channels == 3){
		W = Scalar(1,1,1);
		ROI_temp = W(Rect(topLef.x, topLef.y, (botRig.x - topLef.x + 1), (botRig.y - topLef.y + 1)));
		ROI_temp = Scalar(0,0,0);
		Z = Scalar(0, 0, 0);
		ROI_temp = Z(Rect(topLef.x, topLef.y, (botRig.x - topLef.x + 1), (botRig.y - topLef.y + 1)));
		ROI_temp = Scalar(1,1,1);
		Z_old = Z&Z_old;
		src.convertTo(img16, CV_16UC3, 1, 0);


		B_old = (B_old.mul(W_old) + img16.mul(W)) / (W_old + W + Z_old);
		W_old = W_old + W;

		W_old.convertTo(W_old, CV_8UC3, 1, 0);
		W_old.convertTo(W_old, CV_16UC3, 1, 0);
		return true;
	}
	return false;

}

bool background_alg::back_display(){
	Mat disp;
	if (channels == 1){
		B_old.convertTo(disp, CV_8UC1, 1, 0);
		imshow("background", disp);
		waitKey(1);
		return true;
	}
	else if (channels == 3)
	{
		B_old.convertTo(disp, CV_8UC3, 1, 0);
		imshow("background", disp);
		waitKey(1);
		return true;
	}
	return false;
}

Rect background_alg::object_detect(Mat &src){
	Mat binar;
	Mat background8;
	Mat img8;
	Mat sub;

	if (channels == 1){
		cv::cvtColor(src, img8, CV_BGR2GRAY);
		B_old.convertTo(background8, CV_8UC1, 1, 0);
		sub = abs(img8 - background8);
		binar = sub.clone();
		//imshow("sub", sub);
		//waitKey(5);
	}

	else if (channels == 3){
		B_old.convertTo(background8, CV_8UC3, 1, 0);
		sub = abs(src - background8);
		cv::cvtColor(sub, binar, CV_BGR2GRAY);
	}

	//���ݻҶ�ֱ��ͼ�����ж�̬��ֵ��ֵ��
	adaptive_treshold(binar,binar);
    //imshow("binar",binar);
    //waitKey(5);

	//���е������㣻
	return  (iteration(binar));
}

bool background_alg::adaptive_treshold(Mat &src, Mat &dst){
	static int test_count = 0;
	test_count++;

//	MatND dstHist;
//	int dims = 1;
//	float hranges[2] = { 0, 255 };
//	const float *ranges[1] = { hranges };   // ������ҪΪconst����
//	int size = 256;
//	int channels = 0;
//	calcHist(&src, 1, &channels, Mat(), dstHist, dims, &size, ranges);
//	int ihist = 255;
//	double num = 0;
//	while (1){
//		num += dstHist.at<float>(ihist, 0);
//		if (num / (src.rows*src.cols) > 0.05)break;
//		else ihist--;
//	}
    cv::threshold(src*2, dst, 0, 255, cv::THRESH_OTSU);
	//cv::threshold(src, dst, ihist, 255, cv::THRESH_BINARY);
	//erode(dst, dst, kernel_erode);

	Rect remove_Rect;
	remove_Rect = Rect(max(old_Rectangle.x - contour, 0),
		max(old_Rectangle.y - contour, 0),
		min(old_Rectangle.width + 2 * contour, dst.cols - max(old_Rectangle.x - contour, 0)),
		min(old_Rectangle.height + 2 * contour, dst.rows - max(old_Rectangle.y - contour, 0)));
	Mat remain_ROI = dst(remove_Rect).clone();
	erode(dst, dst, kernel_erode_outer);
	Mat retrive_ROI = dst(remove_Rect);
	retrive_ROI = retrive_ROI + remain_ROI;
	Outer_contour = remove_Rect;
	return true;
}

bool background_alg::center_compute(Mat &src, int *X_proj, int *Y_proj, double &num_cent){
	Point2d centerd;
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) == 255)
			{
				centerd += Point2d(j, i);
				X_proj[j]++;
				Y_proj[i]++;
				num_cent++;
			}
		}
	}
	center = (1 / num_cent)*centerd;

	return true;
}

Rect background_alg::iteration(Mat &src){
	//�����ֵ��ͼ������&ͬʱ����X,Y����ͶӰͼ
	int *X_proj = new int[src.cols];
	int *Y_proj = new int[src.rows];
    for(int i=0;i<src.cols;i++)X_proj[i]=0;
    for(int i=0;i<src.rows;i++)Y_proj[i]=0;
	double num_cent = 0;
	center = Point(0, 0);
	center_compute(src, X_proj, Y_proj, num_cent);

	if (num_cent == 0)return old_Rectangle;

	//�����������ų߶�
	int Result_width = 0;
	int Result_height = 0;
	Point center_i = center;
	double iter_cond_space = 1;
	double iter_cond_space_old = 1;
	double iter_cond_ratio = 0;
	double iter_cond_ratio_old = 0;
	int	   iter_step = 0;
	double iter_pixel = 0;
	float factor;
	int itermax[2] = { src.cols*0.05, src.rows*0.3 };
	int iter_width;
	int iter_height;

	//��X��Y����ֱ������������
	int it_ori_min = 0;
	int it_ori_max = 0;
	//X����
	factor = itermax[0];
	iter_width = itermax[0];
	//iter_width = 0;

	while (iter_cond_ratio < 0.95){
		iter_cond_ratio_old = iter_cond_ratio;
		iter_pixel = 0;
		//iter_width = iter_width + factor;
        iter_width = iter_width + 2;

		it_ori_min = center_i.x - iter_width / 2;
		it_ori_max = center_i.x + iter_width / 2;
        if (it_ori_max >= src.cols - 1)it_ori_max = src.cols - 1;
        if (it_ori_min <= 0)it_ori_min = 0;

		for (int i = it_ori_min; i < it_ori_max; i++)
		{
			iter_pixel += X_proj[i];
		}
		iter_cond_ratio = iter_pixel / num_cent;
//        std::cout << "center_i=" << center_i << std::endl;
//        std::cout << "it_ori_min=" << it_ori_min << std::endl;
//        std::cout << "it_ori_max=" << it_ori_max << std::endl;
//        std::cout << "X_iter_cond_ratio=" << iter_cond_ratio << std::endl;
//        std::cout << "num_cent" << num_cent << std::endl;
//        std::cout << "iter_pixel" << iter_pixel << std::endl;

		iter_step++;

		//factor = itermax[0] - 1.5 * iter_step;
		//if ((iter_cond_ratio - iter_cond_ratio_old<0.02&&iter_cond_ratio>0.9) || factor <= 1)break;
        //if ((iter_cond_ratio - iter_cond_ratio_old<0.02&&iter_cond_ratio>0.9) || iter_step >= 30)break;
        if ( iter_step >= 15) break;
	}
	Result_width = iter_width;



	//Y����
    iter_step=0;
	iter_cond_ratio = 0;
	iter_cond_ratio_old = 0;
	factor = itermax[1];
	iter_height = itermax[1];
	//iter_height = 0;
	while (iter_cond_ratio < 0.97){
		iter_cond_ratio_old = iter_cond_ratio;
		iter_pixel = 0;
		//iter_height = iter_height + factor;
        iter_height = iter_height + 4;

		it_ori_min = center_i.y - iter_height / 2;
		it_ori_max = center_i.y + iter_height / 2;
        if (it_ori_max >= src.rows - 1)it_ori_max = src.rows - 1;
        if (it_ori_min <= 0)it_ori_min = 0;

		for (int i = it_ori_min; i < it_ori_max; i++)
		{
			iter_pixel += Y_proj[i];
		}
		iter_cond_ratio = iter_pixel / num_cent;
//		std::cout << "Y_iter_cond_ratio=" << iter_cond_ratio << std::endl;

		iter_step++;
		//factor = itermax[1] - 3 * iter_step;
		//if ((iter_cond_ratio - iter_cond_ratio_old<0.01&&iter_cond_ratio>0.9) || factor <= 1)break;
        if ( iter_step >= 15) break;
	}
	Result_height = iter_height;

    Rect Result = image_contour & Rect(center_i.x-Result_width/2, center_i.y-Result_height/2, Result_width, Result_height);

	//��ʾ���
	Mat iter_temp = src.clone();
	cv::rectangle(iter_temp, Result, Scalar(255), 1);
	cv::rectangle(iter_temp, Outer_contour, Scalar(255), 2);

	//if (test_count >= 100)cv::rectangle(iter_temp, remove_Rect, Scalar(255, 0, 255), 3);
	//imshow("iter_temp", iter_temp);
	//waitKey(1);
	old_Rectangle = Result;
	delete[] Y_proj;
	delete[] X_proj;
	return Result;

}
