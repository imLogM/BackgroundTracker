#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "highgui.h"
#include "colotracker.h"
#include "region.h"
using namespace cv;
using namespace std;

class background_alg{
private:
	Mat W;
	Mat Z;
	Mat kernel_erode;
	Mat kernel_erode_outer;

	Rect old_Rectangle;
	Rect Outer_contour;
	Rect image_contour;


	Point bal_topLef;
	Point bal_botRight;

	double bal_resize_fac;
	int bal_extend;
	int channels;
	int contour;

	double treshold;
	Point2i center;
	bool adaptive_treshold(Mat &src, Mat &dst);
	bool center_compute(Mat &src, int *X_proj, int *Y_proj, double &num_cent);
	Rect iteration(Mat &src);
public:
	Mat W_old;
	Mat Z_old;
	Mat B_old;



	background_alg(int width, int height, int chan, double resize_fac,double tresher);
	background_alg();
	~background_alg();

	bool initialize(int width, int height, int chan, double resize_fac, double tresher);

	bool background_update(Mat &src, Point topLef, Point botRig);

	bool back_display();

	Rect object_detect(Mat &src);

	bool background_isReady();


};