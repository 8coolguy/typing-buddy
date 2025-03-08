#include "typing_buddy_utility.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>


using namespace std;
#define GAUS_THRESHOLD 30.0
#define GAUS_KSIZE 9
#define ERR(msg)        \
  cout << msg << endl;  \
  exit(1);              \


int main(){
	cout << "OpenCV version: " << CV_VERSION << endl;
	cv::VideoCapture cap(0);
	cv::Mat ref, frame1, frame2, edges, eroded, kmeans;
  vector<vector<cv::Point>> contours;
	if(!cap.isOpened()) {ERR("Camera Issue")}
  cap >> ref;
  if(ref.empty()) {ERR("Empty Reference Frame")}
	cv::flip(ref, ref, -1);
  kmeans = applyKmeansClustering(ref, 3);
  kmeans.convertTo(kmeans, CV_8U);
  ref = applyGrayScale(kmeans);
  cv::GaussianBlur(ref, ref, cv::Size(GAUS_KSIZE,GAUS_KSIZE), GAUS_THRESHOLD);
  cv::Canny(ref, edges, 65, 130, 5);
  cv::imshow("canny", edges);
  //cv::cornerHarris(edges, edges, 31, 17, .05);
  cv::findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
  cv::imshow("canny/hariscorners", edges);

  while(cv::waitKey(1) != 27);
  ERR("Breakpoint to find where the keys are");
	while(1){
		cap >> frame1;
		if(frame1.empty()) {ERR("Empty Frame")}
		if(cv::waitKey(1) == 27) {break;}
		cv::flip(frame1, frame2, -1);
		frame2 = applyGrayScale(frame2);

    cv::GaussianBlur(frame2, frame2, cv::Size(GAUS_KSIZE,GAUS_KSIZE), GAUS_THRESHOLD);
    cv::absdiff(ref, frame2, frame2);
    cv::threshold(frame2, frame2, 10.0, 250.0, cv::THRESH_BINARY);
    erosion(frame2, eroded, cv::MORPH_ELLIPSE, 5);
		cv::imshow("frame2", frame2);
		cv::imshow("eroded", eroded);

	}


	cap.release();
	cv::destroyAllWindows();
	return 0;
}
