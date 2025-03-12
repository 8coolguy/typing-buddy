/*
* typing_buddy_utility.hpp
* This file contains the helper functions needed fot the typing-buddy project.
*
*/
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#define ERR(msg)        \
  cout << msg << endl;  \
  exit(1);              \

struct pixel{
	int pos;
	float r;
	float g;
	float b;
};
struct centroid{
	int count;
	float r;
	float g;
	float b;
};
typedef struct pixel pixel;
typedef struct centroid centroid;

void printMatrix(const cv::Mat& image);
cv::Mat applyGrayScale(cv::Mat image);
void erosion(cv::InputArray input, cv::OutputArray output, int erosion_type, int kernel_size);
cv::Mat applyKmeansClustering(cv::Mat image, int k, double sigma);

