#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
int main(){
	cout << "OpenCV version: " << CV_VERSION << endl;
	cv::VideoCapture cap(1);
	if(!cap.isOpened()){
		cerr << "Cannot open camera" << endl;
		return -1;
	}

	cv::Mat frame;
	while(1){
		cap >> frame;
		if(frame.empty()){
			cerr << "Empty frame" << endl;
			break;
		}
		cv::imshow("frame", frame);
		if(cv::waitKey(1) == 1){
			break;
		}
	}
	cap.release();
	cv::destroyAllWindows();


	return 0;
}
