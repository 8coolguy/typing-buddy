/*
* typing_buddy_utility.cpp
* This file contains the functions definitions for typing_buddy_utility.
*
*/
#include "typing_buddy_utility.hpp"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <opencv2/core.hpp>


void printMatrix(const cv::Mat& image) {
  for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
      std::cout << image.at<float>(i, j) << " ";
		}
    std::cout << std::endl;
	}
}

cv::Mat applyGrayScale(cv::Mat image){
	cv::Mat gray_image;
	image.convertTo(image, CV_32F);
	int rows = image.rows;
	int cols = image.cols;
	//preprocess
  std::vector<cv::Mat> channels;
	cv::split(image, channels);
	gray_image = 0.2126 * channels[2] + 0.7152 * channels[1] + 0.0722 * channels[0];
	gray_image.convertTo(gray_image, CV_8U);
	return gray_image;
}

void erosion(cv::InputArray input, cv::OutputArray output, int erosion_type, int kernel_size){
  cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(kernel_size, kernel_size));
  cv::erode(input, output, element);
}
