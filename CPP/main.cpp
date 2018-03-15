#include "skew.h"
#include "cropper.h"

struct TextData {
	vector<string> Text;
	vector<Rect> BoundingBoxes;
	std::vector<float> Confidence;
};

vector<Rect> Crop(Mat imageData);
TextData getTextFromImage(string ImagePath);
Image increaseDPI(string inputFileName, Geometry geo);


int main() {
	string imgPath = "TestCases/test7.jpeg";
	//string imgPath = "test/scenetext_segmented_word01.jpg";
	//string procImagePath = "TEMP/Proc.jpg";
	//Image temp = increaseDPI(imgPath, Geometry(70,70));
	//temp.write(procImagePath);
	
	//Image resize;
	//resize.read("TestCases/test7.jpeg");
	//resize.resize(Geometry(resize.columns()*2,resize.rows()*2));
	//resize.write("test.tiff");
	
	TextData data = getTextFromImage(imgPath);
	for (int i = 0; i < data.Text.size(); ++i) {
		cout << data.Text[i];
	}
	cout << endl;
	return 0;
}

TextData getTextFromImage(string ImagePath) {
	Skew* skewObj = new Skew();
	tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();
	if (tess->Init(NULL, "eng")) {
        cout << "!!english language file not found!!" << endl;
        exit(1);
    }
	Mat skewedImage = (skewObj->GetSkewedImage(ImagePath)).image;
	Mat gray;
	vector<Rect> TextBoxes = Crop(skewedImage);
	vector<string> TextVec;
	vector<float> confidence;
	string tempFileName = "TEMP/temp.tiff";
	for (int i = 0; i < TextBoxes.size(); ++i) {
		Mat tempImage = (skewObj->GetSkewedImage(skewedImage(TextBoxes[i]))).image;
		cvtColor(tempImage, tempImage, CV_BGR2GRAY);
		threshold(tempImage, tempImage, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
		imwrite(tempFileName, tempImage);
		//Image tempMagick = increaseDPI(tempFileName, Geometry(1000,1000));
		//tempImage = imread(tempFileName.c_str());
		//Pix *FinalProcImage = pixRead(tempFileName.c_str());
		//tess->SetImage(FinalProcImage);
		Image tempMagick;
		tempMagick.read(tempFileName);
		tempMagick.resize(Geometry(tempMagick.columns()*1000,tempMagick.rows()*1000));
		tempMagick.write(tempFileName);
		tempImage = imread(tempFileName);
		cout << "started iteration " << i << endl;

		tess->SetImage((uchar*)tempImage.data, tempImage.size().width, tempImage.size().height, tempImage.channels(), tempImage.step1());
		tess->Recognize(0);
		TextVec.push_back(tess->GetUTF8Text());
		confidence.push_back(tess->MeanTextConf());
		cout << TextVec.back() << endl;
		//imshow(tempFileName, imread(tempFileName.c_str()));
		//waitKey(0);
		remove(tempFileName.c_str());
		bool failed = !std::ifstream(tempFileName.c_str());
		if (!failed) { cout << "FAILED TO DELETE FILE!!!|  " << i << endl; }
	}
	TextData data = { TextVec , TextBoxes , confidence};
	return data;
}

Image increaseDPI(string inputFileName, Geometry geo) {
	Image tempMagick;
	tempMagick.read(inputFileName);
	tempMagick.resolutionUnits(PixelsPerInchResolution);
	tempMagick.density(geo);
	return tempMagick;
}

vector<Rect> Crop(Mat imageData) {
	vector<Rect> BoundingBoxes;
	Mat large = imageData;
	Mat rgb;
	//pyrDown(large, rgb);
	rgb = large;
	Mat small;
	cvtColor(rgb, small, CV_BGR2GRAY);
	Mat grad;
	//10,4
	Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(10,4));
	morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
	Mat bw;
	threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
	Mat connected;
	morphKernel = getStructuringElement(MORPH_RECT, Size(4,2));
	morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
	Mat mask = Mat::zeros(bw.size(), CV_8UC1);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
		Rect rect = boundingRect(contours[idx]);
		Mat maskROI(mask, rect);
		maskROI = Scalar(0, 0, 0);
		drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
		double r = (double)countNonZero(maskROI)/(rect.width*rect.height);
		if (r > .45 && (rect.height > 8 && rect.width > 8)) {
			if ((rect.y-5>0) && (rect.x-5>0)) {
				rect.y-=5;
				rect.x-=5;
			}
			rect+=Size(7,7);
			if (0 <= rect.x
			    && 0 <= rect.width
			    && rect.x + rect.width <= rgb.cols
			    && 0 <= rect.y
			    && 0 <= rect.height
			    && rect.y + rect.height <= rgb.rows){
				BoundingBoxes.push_back(rect);
			} else {
			    rect-=Size(7,7);
			    BoundingBoxes.push_back(rect);
			}
			rectangle(rgb, rect, Scalar(0, 255, 0), 2);
		}
	}
	imwrite("final.png", rgb);
	return BoundingBoxes;
}