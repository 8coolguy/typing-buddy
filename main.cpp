#include "typing_buddy_utility.hpp"
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <vector>


using namespace std;
#define GAUS_THRESHOLD 10.0
#define GAUS_KSIZE 5

void on_trackbar(cv::Size size, vector<vector<cv::Point>> contours, vector<cv::Vec4i> hierarchy){
  cv::Mat cnt_img = cv::Mat::zeros(size, CV_8SC3);
  int idx = 0;
  for( ; idx >= 0; idx = hierarchy[idx][0]){
    cv::Scalar color( rand()&255, rand()&255, rand()&255 );
    cv::drawContours(cnt_img, contours, idx, color, cv::FILLED, 8, hierarchy );
  }
  //cv::polylines(cnt_img, contours, true, cv::Scalar(128, 255, 255), 3, cv::LINE_AA);
  imshow("contours", cnt_img);
}

void findHands(cv::Mat color_image){
  //grab reference image 
  // look at 4 boxes where the color of the hand Issue
  //mark the color and look for that color while looking 
  vector<cv::Mat> channels;
  cv::split(color_image, channels);
  int hist_size = 256;
  float range[] = { 0, 256 }; //the upper boundary is exclusive
  const float* histRange[] = { range };
  bool uniform = true;
  bool accumulate = false;


  cv::Mat r_hist, g_hist, b_hist;
  cv::calcHist(&channels[0], 1, 0, cv::Mat(), b_hist, 1, &hist_size, histRange, uniform, accumulate);
  cv::calcHist(&channels[1], 1, 0, cv::Mat(), g_hist, 1, &hist_size, histRange, uniform, accumulate);
  cv::calcHist(&channels[2], 1, 0, cv::Mat(), r_hist, 1, &hist_size, histRange, uniform, accumulate);

  int hist_w = 512, hist_h = 400;
  int bin_w = hist_w/hist_size;
  cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );


  cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

  for( int i = 1; i < hist_size; i++ ){
    cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - b_hist.at<float>(i-1)),
      cv::Point( bin_w*(i), hist_h - b_hist.at<float>(i)),
      cv::Scalar( 255, 0, 0), 2, 8, 0 );
    cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - g_hist.at<float>(i-1)),
      cv::Point( bin_w*(i), hist_h - g_hist.at<float>(i)),
      cv::Scalar( 0, 255, 0), 2, 8, 0 );
    cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - r_hist.at<float>(i-1) ),
      cv::Point( bin_w*(i), hist_h - r_hist.at<float>(i) ),
      cv::Scalar( 0, 0, 255), 2, 8, 0 );
  }
  imshow("color_image", color_image);
  imshow("calcHist", histImage );

}

void applyCalibration(cv::Mat image_with_hands){
  cout << "Place hands on the five boxes on thre screen." << endl;

}
int main(){
	cout << "OpenCV version: " << CV_VERSION << endl;
	cv::VideoCapture cap(0);
	cv::Mat ref, frame1, frame2, edges, eroded, kmeans, blurred;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
	if(!cap.isOpened()) {ERR("Camera Issue")}
  cap >> ref;
  if(ref.empty()) {ERR("Empty Reference Frame")}
	cv::flip(ref, ref, -1);
  kmeans = applyKmeansClustering(ref, 2, .001);
  kmeans.convertTo(kmeans, CV_8U);
  ref = applyGrayScale(kmeans);
  cv::GaussianBlur(ref, blurred, cv::Size(GAUS_KSIZE,GAUS_KSIZE), GAUS_THRESHOLD);
  cv::Canny(blurred, edges, 30, 200, 5);
  cv::imshow("canny", edges);
  cv::findContours(edges, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  cout << contours.size() << endl;
  cv::imshow("canny/hariscorners", edges);
  on_trackbar(ref.size(), contours, hierarchy);

	while(1){
		cap >> frame1;
		if(frame1.empty()) {ERR("Empty Frame")}
		if(cv::waitKey(1) == 27) {break;}
		cv::flip(frame1, frame1, -1);
    findHands(frame1);
		frame2 = applyGrayScale(frame1);
    cv::GaussianBlur(frame2, frame2, cv::Size(GAUS_KSIZE,GAUS_KSIZE), GAUS_THRESHOLD);
    erosion(frame2, frame2, cv::MORPH_ELLIPSE, 5);
    cv::absdiff(blurred, frame2, frame2);
    cv::threshold(frame2, frame2, 30.0, 200.0, cv::THRESH_BINARY);
	}

	cap.release();
	cv::destroyAllWindows();
	return 0;
}
