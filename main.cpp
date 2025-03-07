#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace std;
/***** Utility Function ******/

#define ERR(msg)        \
  cout << msg << endl;  \
  exit(1);              \

void printMatrix(const cv::Mat& image) {
  for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			cout << image.at<float>(i, j) << " ";
		}
		cout << endl;
	}
}

cv::Mat applyGrayScale(cv::Mat image){
	cv::Mat gray_image;
	image.convertTo(image, CV_32F);
	int rows = image.rows;
	int cols = image.cols;
	//preprocess
	vector<cv::Mat> channels;
	cv::split(image, channels);
	gray_image = 0.2126 * channels[2] + 0.7152 * channels[1] + 0.0722 * channels[0];
	gray_image.convertTo(gray_image, CV_8U);
	return gray_image;
}

int main(){
	cout << "OpenCV version: " << CV_VERSION << endl;
	cv::VideoCapture cap(0);
	cv::Mat ref, frame1, frame2;
	if(!cap.isOpened()) {ERR("Camera Issue")}
  cap >> ref;
  if(ref.empty()) {ERR("Empty Reference Frame")}
	cv::flip(ref, ref, -1);
  ref = applyGrayScale(ref);
  cv::imshow("reference", ref);


	while(1){
		cap >> frame1;
		if(frame1.empty()) {ERR("Empty Frame")}
		if(cv::waitKey(1) > 0) {break;}
		cv::flip(frame1, frame2, -1);
		frame2 = applyGrayScale(frame2);
    cv::absdiff(ref, frame2, frame2);
    cv::threshold(frame2, frame2, 50.0, 255, cv::THRESH_BINARY);
    //cv::Sobel(frame2, frame2, CV_16S, 1, 0);
    //vector<cv::Vec4i> lines;
    // Apply Hough Transform
    //HoughLinesP(frame2, lines, 1, CV_PI/180, 100.0, 10, 250);
    // Draw lines on the image
    //for (int i=0; i<lines.size(); i++) {
      //cv::Vec4i l = lines[i];
      //cv::line(frame2, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
    //}
		cv::imshow("frame", frame2);

	}


	cap.release();
	cv::destroyAllWindows();
	return 0;
}
