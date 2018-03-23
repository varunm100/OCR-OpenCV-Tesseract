#include "skew.h"
#include "cropper.h"

struct TextData {
	vector<string> Text;
	vector<Rect> BoundingBoxes;
	std::vector<float> Confidence;
};

vector<Rect> Crop(Mat imageData);
TextData getTextFromImage(string ImagePath);

int main() {
	string imgPath = "TestCases/test7.jpeg";;
	TextData data = getTextFromImage(imgPath);
	/*for (int i = 0; i < data.Text.size(); ++i) {
		cout << data.Text[i];
	}
	cout << endl;*/
	return 0;
}

TextData getTextFromImage(string ImagePath) {
	Mat kernel = (Mat_<int>(3, 3) << 0, 1, 0,1, -1, 1,0, 1, 0);
	Skew* skewObj = new Skew();
	tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();
	tess->SetPageSegMode(tesseract::PSM_SINGLE_WORD);
	//tess->setVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>.=");
	if (tess->Init(NULL, "eng+hin")) {
		cout << "!!english language file not found!!" << endl;
		exit(1);
	}
	//Ptr<OCRTesseract> ocr = OCRTesseract::create();
	imshow("Original Image", imread(ImagePath));
	Mat skewedImage = (skewObj->GetSkewedImage(ImagePath)).image;
	Mat copyTemp = skewedImage.clone();
	imshow("Deskewed Image", skewedImage);
	waitKey(0);
	Mat gray;
	vector<Rect> TextBoxes = Crop(copyTemp);
	vector<string> TextVec;
	vector<float> confidence;
	string tempFileName = "TEMP/temp.tiff";
	for (int i = 0; i < TextBoxes.size(); ++i) {
		//Mat tempImage = (skewObj->GetSkewedImage(skewedImage(TextBoxes[i]))).image;
		Mat tempImage = skewedImage(TextBoxes[i]);
		//cvtColor(tempImage, tempImage, CV_BGR2GRAY);
		resize(tempImage, tempImage, Size(tempImage.size().width*3,tempImage.size().height*3));
		imwrite(tempFileName, tempImage);
		string command = "./textcleaner -g -e normalize -f 100	 -o 12 -s 2 " + tempFileName + " " + tempFileName;
		system(command.c_str());
		tempImage = imread(tempFileName.c_str());
		cvtColor(tempImage, tempImage, CV_BGR2GRAY);
		//floodFill(tempImage, Point(10,10), Scalar(255.0, 255.0, 255.0));
		//threshold(tempImage, tempImage, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
		
		tess->SetImage((uchar*)tempImage.data, tempImage.size().width, tempImage.size().height, tempImage.channels(), tempImage.step1());
		tess->Recognize(0);

		string output;
		vector<Rect>   boxes;
		vector<string> words;
		vector<float>  confidences;
		//ocr->run(tempImage, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);
		/*for (int j = 0; j < words.size(); ++j) {
			cout << words[j] << " Confidence: " << confidences[j] << endl;
		}*/

		TextVec.push_back(tess->GetUTF8Text());
		confidence.push_back(tess->MeanTextConf());
		cout << TextVec.back() << " Confidence: " << confidence.back() << endl;
		imshow(tempFileName, tempImage);
		waitKey(0);
		remove(tempFileName.c_str());
		bool failed = !std::ifstream(tempFileName.c_str());
		if (!failed) { cout << "FAILED TO DELETE FILE!!!|  " << i << endl; }
		//GetTextFromCroppedImage(skewedImage(TextBoxes[i]).clone());
	}
	TextData data = { TextVec , TextBoxes , confidence };
	return data;
	//g++ *.cpp -std=gnu++11 -o OCR-OpenCV-Tesseract -llept -ltesseract `pkg-config --cflags --libs opencv`
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
	imshow("Cropped Image", rgb);
	waitKey(0);
	return BoundingBoxes;
}