#include "INCLUDES.h"

struct BoundingBoxesData {
	vector<Rect> BoundingBoxes;
	vector <vector<Point> > regions;
};

class Cropper {
public:
	Cropper();
	void DisplayRectangles(BoundingBoxesData dataB, Mat inputImage);
	BoundingBoxesData MSERGuassianBin(string imagePath);
	BoundingBoxesData MSERGuassianBin(Mat inImage);
	~Cropper();
};