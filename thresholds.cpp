#include "typing_buddy_utility.hpp"
#include <iostream>
#include <vector>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
using namespace std;

vector<vector<cv::Point>> contours;
vector<cv::Vec4i> hierarchy;
vector<key> key_vec;
set<int> box_set;

cv::Size size;
cv::Mat reference, kmeans, gray_scale, blurred, edges, cnt_img, annotated;

int gauss_kernel = 3;
int k = 3;
int gauss_thresh = 3;
int canny_t1 = 6;
int canny_t2 = 13;

static void trackbar_callback(int pos, void* userdata){
  cv::GaussianBlur(gray_scale, blurred, cv::Size(gauss_kernel,gauss_kernel), gauss_thresh*10.0);
  cv::Canny(blurred, edges, canny_t1*10.0, canny_t2*10.0, 5);
  cv::findContours(edges, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  cnt_img = cv::Mat::zeros(size, CV_8SC3);
  int idx = 0;
  for( ; idx >= 0; idx = hierarchy[idx][0]){
    cv::Scalar color(128, 255, 255);
    cv::drawContours(cnt_img, contours, idx, color, cv::FILLED, 8, hierarchy );
  }
  cout << "gauss_thresh: " << gauss_thresh << " gauss_kernel: " << gauss_kernel << " t1: " << canny_t1 << " t2: " << canny_t2 << endl;
  cv::copyTo(cnt_img, annotated, cv::Mat());
  imshow("reference adjust", cnt_img);
}

static void mouse_callback(int event, int x, int y, int flags, void*userdata ){
  if(event != 4) return;
  cv::Size size = cnt_img.size();
  if(x > size.width || y > size.height) return;
  cv::Point p(x,y);
  cv::Rect box;
  cv::Mat mask = cv::Mat::zeros(size.height + 2, size.width + 2, CV_8UC1);
  cv::Mat cnt_img8;
  cnt_img.convertTo(cnt_img8, reference.type());
  cv::floodFill(cnt_img8, mask, p, cv::Scalar(255), &box, cv::Scalar(0), cv::Scalar(0), cv::FLOODFILL_MASK_ONLY);
  int pos = box.x * size.width + box.y;
  if (box_set.find(pos) != box_set.end()) return;
  cout << "Press a key to register region:" << endl;
  int keypress = cv::waitKey(0);
  cv::rectangle(annotated, box, cv::Scalar(255,0,0), 10, cv::LINE_8, 0);
  cv::imshow("new with rectangle",annotated);
  box_set.insert(pos);
  key_vec.push_back(key({keypress,box}));
  cout << box.x << SPACE << box.y << SPACE << box.height << SPACE << box.width << SPACE << keypress << endl;
}

int main(){
  //cv::VideoCapture cap(0);
	//if(!cap.isOpened()) {ERR("Camera Issue")}
  //cap >> reference;
  reference = cv::imread("pic.jpg", cv::IMREAD_COLOR);
  if(reference.empty()) {ERR("Empty Reference Frame")}
  size = reference.size();
	cv::flip(reference, reference, -1);
  kmeans = applyKmeansClustering(reference, k, .1);
  kmeans.convertTo(kmeans, CV_8U);
  gray_scale = applyGrayScale(kmeans);

  cv::namedWindow("reference adjust");
  cv::setMouseCallback("reference adjust", mouse_callback);
  cv::createTrackbar("gauss_kernel_trackbar", "reference adjust", &gauss_kernel, 9, trackbar_callback);// only can move to odd numbers
  cv::createTrackbar("gauss_thresh_trackbar", "reference adjust",&gauss_thresh, 25, trackbar_callback);
  trackbar_callback(gauss_thresh, nullptr);
  while(cv::waitKey(1) != 27);
}
