#include "INCLUDES.h"

struct SkewData {
	double rotation;
	Mat image;
};

class Skew {
public:
	Skew();
	Mat rot(Mat& im, double thetaRad);
	Mat preprocess2(Mat& im);
	Mat preprocess1(Mat& im);
	void hough_transform(Mat& im,Mat& orig,double* skew);
	SkewData GetSkewedImage(string filePath);
	SkewData GetSkewedImage(Mat imageData);
	void ShowBeforeAfter(string filePath);
	~Skew();
};