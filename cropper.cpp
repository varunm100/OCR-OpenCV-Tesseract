#include "cropper.h"

Cropper::Cropper() {

}

Cropper::~Cropper() {

}

void Cropper::DisplayRectangles(BoundingBoxesData dataB, Mat inputImage) {
	for (int i = 0; i < dataB.regions.size(); i++) {
		rectangle(inputImage, dataB.BoundingBoxes[i], CV_RGB(0, 255, 0));  
	}
	cv::namedWindow("MSER Algo", cv::WINDOW_AUTOSIZE );
	imwrite("FinalImage.png", inputImage);
	imshow("mser", inputImage);
	waitKey(0);
}

BoundingBoxesData Cropper::MSERGuassianBin(string imagePath) {
	Mat image = imread(imagePath.c_str(), 0);
	imshow("GreyScaledImage", image);
	waitKey(0);
	//5, 60, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5
	Ptr<MSER> mserObj = MSER::create(5, 100, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5);
	vector <vector<Point> > regions;
	vector <Rect> mser_bbox;
	mserObj->detectRegions(image, regions, mser_bbox);
	BoundingBoxesData rectData = { mser_bbox , regions };
	return rectData;
}

BoundingBoxesData Cropper::MSERGuassianBin(Mat inImage) {
	Mat image = inImage;
	imshow("GreyScaledImage", image);
	waitKey(0);
	Ptr<MSER> mserObj = MSER::create();
	vector <vector<Point> > regions;
	vector <Rect> mser_bbox;
	mserObj->detectRegions(image, regions, mser_bbox);
	BoundingBoxesData rectData = { mser_bbox , regions };
	return rectData;
}