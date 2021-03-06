#include "skew.h"
#include "cropper.h"

struct TextData {
	vector<string> Text;
	vector<Rect> BoundingBoxes;
	vector<float> Confidence;
};

vector<Rect> Crop(Mat imageData);
TextData getTextFromImage(string ImagePath);

int main() {
	string imgPath = "TestCases/test14.jpg";
	TextData data = getTextFromImage(imgPath);
	cout << "__Text Output__" << endl;
	for (int i = 0; i < data.Text.size(); ++i) {
		cout << data.Text[i];
	}
	cout << endl;
	return 0;
}

TextData getTextFromImage(string ImagePath) {
	Skew* skewObj = new Skew();
	tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();
	tess->SetPageSegMode(tesseract::PSM_SINGLE_WORD);
	if (tess->Init(NULL, "eng")) {
		cout << "!!english language file not found!!" << endl;
		exit(1);
	}
	imshow("Original Image", imread(ImagePath));
	Mat skewedImage = (skewObj->GetSkewedImage(ImagePath)).image;
	Mat copyTemp = skewedImage.clone();
	imwrite("Deskewed.jpg", copyTemp);
	imshow("Deskewed Image", skewedImage);
	waitKey(0);
	Mat gray;
	vector<Rect> TextBoxes = Crop(copyTemp);
	vector<string> TextVec;
	vector<float> confidence;
	string tempFileName = "TEMP/temp.tiff";
	for (int i = 0; i < TextBoxes.size(); ++i) {
		Mat tempImage = skewedImage(TextBoxes[i]);
		resize(tempImage, tempImage, Size(tempImage.size().width*3,tempImage.size().height*3));
		imwrite(tempFileName, tempImage);
		//string TessCommand = "tesseract " + tempFileName + " " + "TEMP/TXToutput" + " --oem 1 -l eng"; 
		//system(TessCommand.c_str());
		
		string command = "./textcleaner -g -e normalize -f 100	 -o 12 -s 2 " + tempFileName + " " + tempFileName;
		system(command.c_str());
		tempImage = imread(tempFileName.c_str());

		cvtColor(tempImage, tempImage, CV_BGR2GRAY);
		//threshold(tempImage, tempImage, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
		imwrite(tempFileName, tempImage);
		try {	
			tess->SetImage((uchar*)tempImage.data, tempImage.size().width, tempImage.size().height, tempImage.channels(), tempImage.step1());
			tess->Recognize(0);
			TextVec.push_back(tess->GetUTF8Text());
			confidence.push_back(tess->MeanTextConf());
			cout << TextVec.back() << " Confidence: " << confidence.back() << endl;
			imshow(tempFileName, tempImage);
			waitKey(0);
		} catch (const string& TesseractException) {
			cout << "Caught Exception: " << TesseractException << endl;
		}
		remove(tempFileName.c_str());
		bool failed = !std::ifstream(tempFileName.c_str());
		if (!failed) { cout << "FAILED TO DELETE FILE!!!|  " << i << endl; }
		//GetTextFromCroppedImage(skewedImage(TextBoxes[i]).clone());
	}
	TextData data = { TextVec , TextBoxes , confidence };
	return data;
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
	//10,4 - ||Optimal For Now||
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
	imshow("Text Detection", rgb);
	waitKey(0);
	return BoundingBoxes;
}
//g++ *.cpp -std=gnu++11 -o OCR-OpenCV-Tesseract -llept -ltesseract `pkg-config --cflags --libs opencv`