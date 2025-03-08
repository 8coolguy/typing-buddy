/*
* typing_buddy_utility.hpp
* This file contains the helper functions needed fot the typing-buddy project.
*
*/
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

void printMatrix(const cv::Mat& image);
cv::Mat applyGrayScale(cv::Mat image);
void erosion(cv::InputArray input, cv::OutputArray output, int erosion_type, int kernel_size);

