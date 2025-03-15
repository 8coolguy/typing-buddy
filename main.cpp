#include "typing_buddy_utility.hpp"
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
#include <set>
#include <map>
#include <cstdlib>
#include <chrono>


using namespace std;
#define GAUS_THRESHOLD 10.0
#define GAMMA_LEVEL .4
#define GAUS_KSIZE 5
#define CSIZE 50
#define THRESHOLD 25.0
#define EROSION_KSIZE 9
string quote = "with year know since about follow think after head school shes plan under many small present such against now when much before home present";
int last_mouse_click_x = -1;
int last_mouse_click_y = -1;
bool clicked = false;
vector<key> key_vec;
set<int> box_set;
cv::Mat cnt_img, original_frame, annotated,labeled;


map<char, int> keyToFinger = {
    {'a', 0},
    {'q', 0},
    {'z', 0},
    {'w', 1},
    {'s', 1},
    {'x', 1},
    {'e', 2},
    {'d', 2},
    {'c', 2},
    {'r', 3},
    {'f', 3},
    {'v', 3},
    {'t', 3},
    {'g', 3},
    {'b', 3},
    {'y', 4},
    {'h', 4},
    {'n', 4},
    {'u', 4},
    {'j', 4},
    {'m', 4},
    {'i', 5},
    {'k', 5},
    {'o', 6},
    {'l', 6},
    {'p', 7},
};

void mouse_callback(int event, int x, int y, int flags, void* u_data){
  if(clicked && event != cv::EVENT_FLAG_LBUTTON) return;
  last_mouse_click_y = y;
  last_mouse_click_x = x;
}

static void mouse_box_callback(int event, int x, int y, int flags, void*userdata ){
  if(event != cv::EVENT_FLAG_RBUTTON) return;
  cv::Size size = cnt_img.size();
  if(x > size.width || y > size.height) return;
  cv::Point p(x,y);
  cv::Rect box;
  cv::Mat cnt_img8;
  cv::Mat mask = cv::Mat::zeros(size.height + 2, size.width + 2, CV_8UC1);
  cnt_img.convertTo(cnt_img8, original_frame.type());
  cv::floodFill(cnt_img8, mask, p, cv::Scalar(255), &box, cv::Scalar(0), cv::Scalar(0), cv::FLOODFILL_MASK_ONLY);
  int pos = box.x * size.width + box.y;
  if (box_set.find(pos) != box_set.end()) return;
  cout << "Press a key to register region:" << endl;
  int keypress = cv::waitKey(0);
  cv::rectangle(labeled, box, cv::Scalar(255,0,0), 10, cv::LINE_8, 0);
  cv::imshow("labeled",labeled);
  box_set.insert(pos);
  key_vec.push_back(key({keypress,box}));
  cout << box.x << SPACE << box.y << SPACE << box.height << SPACE << box.width << SPACE << keypress << endl;
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
float euclideanDist(cv::Vec3b pixel1, cv::Vec3b pixel2){
  return pow(pow(pixel1[0] - pixel2[0],2) + pow(pixel1[1] - pixel2[1],2) + pow(pixel1[2] - pixel2[2],2),.5);
}

cv::Mat difference(cv::Mat reference, cv::Mat current_frame, int threshold, int new_value){
  cout << reference.type() << SPACE << current_frame.type() << endl;
  cv::Mat mask(reference.size(), reference.type());
  cv::Size size = reference.size();
  for(int r = 0; r < size.width; r++){
    for(int c = 0; c < size.height; c++){
      cv::Vec3f pixel_ref = reference.at<cv::Vec3f>(r,c);
      cv::Vec3f pixel_cur = current_frame.at<cv::Vec3f>(r,c);
      if(euclideanDist(pixel_ref, pixel_cur)> threshold){
        mask.at<cv::Vec3f>(r,c) = pixel_cur;
      }
    }
  }
  return mask;
}

void focus_finger_tip(cv::Mat color_image, cv::Size size){
  if(last_mouse_click_x < 0) return;
  cv::Point pt1(last_mouse_click_x, last_mouse_click_y);
  cv::Point pt2(last_mouse_click_x + size.width, last_mouse_click_y + size.height);
  cv::rectangle(color_image, pt1, pt2, cv::Scalar(0, 255, 0), 8, cv::LINE_8, 0);
  cv::imshow("color_image", color_image);
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

cv::Mat extract_finger(cv::Mat image_with_hands, cv::Size size){
  cv::Mat finger_tip(size, image_with_hands.type());
  for(int row = 0; row < size.height; row++){
    for(int col = 0; col < size.width; col++){
      finger_tip.at<cv::Vec3b>(row, col) = image_with_hands.at<cv::Vec3b>(row + last_mouse_click_y, col + last_mouse_click_x);
    }
  }
  return finger_tip;
}

cv::Mat create_finger_histogram(cv::Mat original_frame, cv::Size size){
  if(last_mouse_click_x < 0) return cv::Mat();
  cv::Mat hsv_image, hist;
  cv::cvtColor(original_frame, hsv_image, cv::COLOR_RGB2HSV);
  cv::imshow("hsv_image", hsv_image);
  cv::Mat finger_hist(size, hsv_image.type());
  //copy image square to box
  for(int row = 0; row < size.height; row++)
    for(int col = 0; col < size.width; col++)
      finger_hist.at<cv::Vec3b>(row, col) = original_frame.at<cv::Vec3b>(row + last_mouse_click_y, col + last_mouse_click_x);
  //FROM opencv documentation:
  int channels[] = {0 ,1, 2};
  int hbins = 15, sbins = 16, vbins = 10;
  int hist_size[] = {hbins, sbins, vbins};
  float hranges[] = { 0, 180};
  float sranges[] = { 0, 256 };
  float vranges[] = {0, 100};
  const float* ranges[] = { hranges, sranges , vranges};
  cv::calcHist(&finger_hist, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
  cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
  return hist;
}

cv::Mat finger_mask(cv::Mat original_frame, cv::Mat hist){
  cv::Mat hsv_image, backproject, output;
  int channels[] = {0 ,1, 2};
  float hranges[] = { 0, 180};
  float sranges[] = { 0, 256 };
  float vranges[] = {0, 100};
  const float* ranges[] = { hranges, sranges , vranges};
  cv::cvtColor(original_frame, hsv_image, cv::COLOR_RGB2HSV);
  cv::calcBackProject(&original_frame, 1, channels, hist, backproject, ranges, 1);
  cv::Mat shp = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9,9));
  cv::filter2D(backproject, backproject,-1, shp);
  double threshold = cv::threshold(backproject, backproject, 20, 255, cv::THRESH_BINARY);
  return backproject;
}

cv::Mat preprocess(cv::Mat frame){
  frame = applyGrayScale(frame);
  cv::GaussianBlur(frame, frame, cv::Size(GAUS_KSIZE,GAUS_KSIZE), GAUS_THRESHOLD);
  frame = adjustGamma(frame, GAMMA_LEVEL);
  return frame;
}

cv::Mat isolate_hands(cv::Mat reference, cv::Mat frame){
  cv::absdiff(frame, reference, frame);
  cv::threshold(frame, frame, THRESHOLD, 255.0, cv::THRESH_BINARY);
  erosion(frame, frame, cv::MORPH_ELLIPSE, EROSION_KSIZE);
  return frame;
}
int determine_finger(){
  return rand() % 8;
}
int main(){
	cout << "OpenCV version: " << CV_VERSION << endl;
	cv::VideoCapture cap(0);
	cv::Mat ref, annotated_frame, gray_scaled, frame, edges, eroded, kmeans, blurred, gamma_adjusted, gamma_adjusted_ref, mask, kmeans_gray_scaled;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  vector<key> key_vec;
  set<int> box_set;
  cv::Size calibration_size(CSIZE, CSIZE);
  cv::namedWindow("color_image");

  cout << "Taking Reference picture of the keyboard" << endl;
	if(!cap.isOpened()) {ERR("Camera Issue")}
  while (ref.empty()) cap >> ref;
	cv::flip(ref, ref, -1);
  cv::copyTo(ref, labeled, cv::Mat());
  kmeans = applyKmeansClustering(ref, 3, .1);
  kmeans.convertTo(kmeans, CV_8U);
  gamma_adjusted_ref = preprocess(kmeans);
  cv::findContours(edges, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  cv::Canny(gamma_adjusted_ref, edges, 30, 200, 5);
  cv::findContours(edges, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  cnt_img = cv::Mat::zeros(ref.size(), CV_8SC3);
  for(int idx = 0 ; idx >= 0; idx = hierarchy[idx][0]){
    cv::Scalar color(255, 255, 255);
    cv::drawContours(cnt_img, contours, idx, color, cv::FILLED, 8, hierarchy );
  }
  cv::imshow("cnt_img", cnt_img);
  cv::imshow("labeled", cnt_img);
  cv::setMouseCallback("labeled", mouse_box_callback, 0);
  cv::waitKey(0);
  cv::setMouseCallback("color_image", mouse_callback, 0);

  int keypress = cv::waitKey(1);
  cv::Mat hand_hist;
  cout << "Click once to record color of finger tips and click eso to finish recording." << endl;
  while(keypress != 27){
    cap >> original_frame;
		if(original_frame.empty()) {ERR("Empty Frame")}
    cv::flip(original_frame, original_frame, -1);
		if(keypress == 45) {calibration_size.width-=5; calibration_size.height-=5;}
		if(keypress == 43) {calibration_size.width+=5; calibration_size.height+=5;}
    cv::imshow("color_image", original_frame);
    focus_finger_tip(original_frame, calibration_size);
    hand_hist = create_finger_histogram(original_frame, calibration_size);
    keypress = cv::waitKey(1);
    clicked = true;
  }
  cout << "Finished Calibration" << endl;
  printMatrix(hand_hist);

  keypress = -1;
  string typed;
  int typed_index = 0;
  cout << "Analysing Typing" << endl;
  cv::Mat game_screen = cv::Mat::zeros(cv::Size(1920,1080), CV_8UC3);
  cv::putText(game_screen, "Type: " + quote, cv::Point(50,100), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 2);
  cv::imshow("display", game_screen);
  int total_inference_time = 0;
  int total = 0;
  int correct = 0;
	while(1){
    keypress = cv::waitKey(0);
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
		cap >> original_frame;
		if(keypress == 27) {break;}
    if(typed_index > quote.size()) break;
		if(original_frame.empty()) {ERR("Empty Frame")}
		cv::flip(original_frame, original_frame, -1);
    gamma_adjusted = preprocess(original_frame);
    mask = isolate_hands(gamma_adjusted_ref, gamma_adjusted);
    cv::copyTo(original_frame, annotated_frame, mask);
    cv::Mat mask2 = finger_mask(original_frame, hand_hist); 
    erosion(mask2, mask2, cv::MORPH_ELLIPSE, 19);
    cv::bitwise_and(original_frame, original_frame, annotated_frame, mask2);
    cv::imshow("mask2", mask2);
    cv::imshow("annotated_frame", annotated_frame);
    cv::imshow("original_frame", original_frame);
    int finger_id = determine_finger();
    if(quote[typed_index] == keypress){
      typed += quote[typed_index];
      if(finger_id == keyToFinger[quote[typed_index]]){
        correct++;
      }
      total++;
      typed_index++;
      chrono::steady_clock::time_point end = chrono::steady_clock::now();
      total_inference_time += chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
    cv::putText(game_screen, "Type: " + quote, cv::Point(50,100), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(game_screen, "Typed: "+typed, cv::Point(50,150), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(game_screen, to_string(correct) +  "/" + to_string(total), cv::Point(50,170), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::imshow("display", game_screen);
    annotated_frame = cv::Mat();
    game_screen = cv::Mat::zeros(cv::Size(1920,1080), CV_8UC3);

	}
  cout << "Your typing score was " << correct << "/" << total << endl;
  cout << "Total Inference Time: " << total_inference_time << endl;
  cout << "Avg. Inference Time: " << total_inference_time/total << endl;
	cap.release();
	cv::destroyAllWindows();
	return 0;
}
