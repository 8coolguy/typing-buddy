#include "typing_buddy_utility.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
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
#define CSIZE 50

int last_mouse_click_x = -1;
int last_mouse_click_y = -1;

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

void mouse_callback(int event, int x, int y, int flags, void* u_data){
  if(0 == event) return;
  last_mouse_click_y = y;
  last_mouse_click_x = x;
}

void focus_finger_tip(cv::Mat color_image, cv::Size size){
  if(last_mouse_click_x < 0) return;
  cv::Point pt1(last_mouse_click_x, last_mouse_click_y);
  cv::Point pt2(last_mouse_click_x + size.width, last_mouse_click_y + size.height);
  cv::rectangle(color_image, pt1, pt2, cv::Scalar(0, 255, 0), 8, cv::LINE_8, 0);
}

cv::Mat adjustGamma(cv::Mat input, double gamma){
  cv::Mat output;
  double inv_gamma = 1.0/gamma;
  vector<int> lookup;
  for(int i = 0; i < 256; i++)
    lookup.push_back(uint(pow((i/255.0),inv_gamma) * 255.0));
  cv::LUT(input, lookup, output);
  output.convertTo(output, CV_8U);
  return output;
}

void createHistogram(cv::Mat color_image){
  //grab reference image 
  // look at 4 boxes where the color of the hand Issue
  //mark the color and look for that color while looking 
  //
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
  imshow("calcHist", histImage );
}

cv::Mat create_finger_histogram(cv::Mat original_frame, cv::Size size){
  cv::Mat hsv_image, hist;
  cv::cvtColor(original_frame, hsv_image, cv::COLOR_RGB2HSV);
  cv::Mat finger_hist(size, hsv_image.type());
  //copy image square to box
  for(int row = 0; row < size.height; row++)
    for(int col = 0; col < size.width; col++)
      finger_hist.at<cv::Vec3b>(row, col) = original_frame.at<cv::Vec3b>(row + last_mouse_click_y, col + last_mouse_click_x);
  //FROM opencv documentation:
  int channels[] = {0 ,1};
  int hbins = 30, sbins = 32;
  int hist_size[] = {hbins, sbins};
  // hue varies from 0 to 179, see cvtColor
  float hranges[] = { 0, 180 };
  // saturation varies from 0 (black-gray-white) to
  // 255 (pure spectrum color)
  float sranges[] = { 0, 256 };
  const float* ranges[] = { hranges, sranges };
  cv::calcHist(&finger_hist, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
  cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

  return hist;
}

cv::Mat finger_mask(cv::Mat original_frame, cv::Mat hist){
  cv::Mat hsv_image, backproject, output;
  int channels[] = {0 ,1};
  float hranges[] = { 0, 180 };
  float sranges[] = { 0, 256 };
  const float* ranges[] = { hranges, sranges };
  cv::cvtColor(original_frame, hsv_image, cv::COLOR_RGB2HSV);
  cv::calcBackProject(&original_frame, 1, channels, hist, backproject, ranges, 1);
  cv::Mat shp = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9,9));
  cv::filter2D(backproject, backproject,-1, shp);
  double threshold = cv::threshold(backproject, backproject, 20, 240, cv::THRESH_BINARY);
  cv::imshow("threshold", backproject);
  //cv::bitwise_and(original_frame, backproject, output);
  //cv::imshow("threshold", output);
  return hsv_image;
}

void applyCalibration(cv::Mat image_with_hands, cv::Size size){
  if(last_mouse_click_x < 0) return;
  cv::Mat finger_tip(size, image_with_hands.type()), hsv_image;
  for(int row = 0; row < size.height; row++){
    for(int col = 0; col < size.width; col++){
      finger_tip.at<cv::Vec3b>(row, col) = image_with_hands.at<cv::Vec3b>(row + last_mouse_click_y, col + last_mouse_click_x);
    }
  }
  //cv::cvtColor(image_with_hands, hsv_image, cv::COLOR_RGB2HSV);
  //createHistogram(finger_tip);
  cv::Mat hand_hist = create_finger_histogram(image_with_hands, size);
  cv::Mat mask = finger_mask(image_with_hands, hand_hist);
}
int main(){
	cout << "OpenCV version: " << CV_VERSION << endl;
	cv::VideoCapture cap(0);
	cv::Mat ref, original_frame, annotated_frame, gray_scaled, frame, edges, eroded, kmeans, blurred, gamma_adjusted;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  cv::Size calibration_size(CSIZE, CSIZE);
  cv::namedWindow("color_image");

  cout << "Taking Reference picture of the keyboard" << endl;
	if(!cap.isOpened()) {ERR("Camera Issue")}
  while (ref.empty()) cap >> ref;

	cv::flip(ref, ref, -1);
  kmeans = applyKmeansClustering(ref, 3, .1);
  kmeans.convertTo(kmeans, CV_8U);
  ref = applyGrayScale(kmeans);
  cv::GaussianBlur(ref, blurred, cv::Size(GAUS_KSIZE,GAUS_KSIZE), GAUS_THRESHOLD);
  cv::Canny(blurred, edges, 30, 200, 5);
  cv::findContours(edges, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  cv::setMouseCallback("color_image", mouse_callback, 0);
	while(1){
		cap >> original_frame;
		if(original_frame.empty()) {ERR("Empty Frame")}
    int keyPress = cv::waitKey(1);
		if(keyPress == 27) {break;}
		if(keyPress == 45) {calibration_size.width-=5; calibration_size.height-=5;}
		if(keyPress == 43) {calibration_size.width+=5; calibration_size.height+=5;}
		cv::flip(original_frame, original_frame, -1);
    gamma_adjusted = adjustGamma(original_frame, 0.4);
    cv::copyTo(gamma_adjusted, annotated_frame, cv::Mat());
    focus_finger_tip(annotated_frame, calibration_size);
    cv::imshow("color_image", annotated_frame);
    applyCalibration(gamma_adjusted, calibration_size);
    /*
		gray_scaled = applyGrayScale(original_frame);
    cv::GaussianBlur(gray_scaled, frame, cv::Size(GAUS_KSIZE,GAUS_KSIZE), GAUS_THRESHOLD);
    erosion(frame, frame, cv::MORPH_ELLIPSE, 5);
    cv::absdiff(blurred, frame, frame);
    cv::threshold(frame, frame, 30.0, 200.0, cv::THRESH_BINARY);
    */
	}

	cap.release();
	cv::destroyAllWindows();
	return 0;
}
