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

void printCentroids(const std::vector<centroid>& centeroids) {
  std::cout << "Centroids:" << centeroids.size() << std::endl;
	for (int i = 0; i < centeroids.size(); ++i) {
    std::cout << centeroids[i].r << " " << centeroids[i].g << " " << centeroids[i].b << " " << centeroids[i].count << std::endl;
	}
}

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

cv::Mat applyKmeansClustering(cv::Mat image, int k){
	image.convertTo(image, CV_32F);
  std::vector<pixel> pixelList;
  std::vector<centroid> centeroids;
	for(int i = 0; i < image.rows; i++){
		for(int j = 0; j < image.cols; j++){
			pixel p;
			p.r = image.at<cv::Vec3f>(i,j)[0];
			p.g = image.at<cv::Vec3f>(i,j)[1];
			p.b = image.at<cv::Vec3f>(i,j)[2];
			p.pos = i * image.cols + j;
			pixelList.push_back(p);
		}
	}
	//select k data points which will be the initial centers of the clusters
	for(int i =  0; i < k; i++){
		int index = rand() % pixelList.size();
		centroid c;
		c.r = pixelList[index].r;
		c.g = pixelList[index].g;
		c.b = pixelList[index].b;
		c.count = 1;
		centeroids.push_back(c);
	}
  std::vector<centroid> new_centroids(k);
	for(int i = 0; i < k; i++){
		new_centroids[i].r = 0;	
		new_centroids[i].g = 0;
		new_centroids[i].b = 0;
		new_centroids[i].count = 0;
	}
	bool converged = false;
	int iteration = 0;
	while(!converged){
		//recalculate the centers of the clusters
		for(int i = 0; i < pixelList.size(); i++){
			int closest_centroid = 0;
			float closest_centroid_distance = 255 * 255 * 255;
			for(int j = 0; j < k; j++){
				float r = pixelList[i].r - centeroids[j].r;
				float g = pixelList[i].g - centeroids[j].g;
				float b = pixelList[i].b - centeroids[j].b;
				float dist = sqrt(r*r + g*g + b*b);
				if(dist < closest_centroid_distance){
					closest_centroid = j;
					closest_centroid_distance = dist;
				}
			}
			new_centroids[closest_centroid].r += pixelList[i].r;
			new_centroids[closest_centroid].g += pixelList[i].g;
			new_centroids[closest_centroid].b += pixelList[i].b;
			new_centroids[closest_centroid].count += 1;
		}

		float error = 0;
		for(int i = 0; i < k; i++){
			new_centroids[i].r /= new_centroids[i].count;
			new_centroids[i].g /= new_centroids[i].count;
			new_centroids[i].b /= new_centroids[i].count;

			float r = centeroids[i].r - new_centroids[i].r;
			float g = centeroids[i].g - new_centroids[i].g;
			float b = centeroids[i].b - new_centroids[i].b;
			error += sqrt(r*r + g*g + b*b);

			centeroids[i].r = new_centroids[i].r;
			centeroids[i].g = new_centroids[i].g;
			centeroids[i].b = new_centroids[i].b;
			centeroids[i].count = 1;
		}
		
		for(int i = 0; i < k; i++){
			new_centroids[i].r = 0;	
			new_centroids[i].g = 0;
			new_centroids[i].b = 0;
			new_centroids[i].count = 0;
		}
    std::cout << "iteration: " << iteration << " error: " << error << std::endl;
		if (error < 0.1) converged = true;
		iteration++;
	}
  printCentroids(centeroids);
	//calculate where each pixel belongs to the centroids
	cv::Mat result(image.size(), image.type());
	for(int i = 0; i < pixelList.size(); i++){
		int closest_centroid = 0;
		float closest_centroid_distance = 255 * 255 * 255;
		for(int j = 0; j < k; j++){
			float r = pixelList[i].r - centeroids[j].r;
			float g = pixelList[i].g - centeroids[j].g;
			float b = pixelList[i].b - centeroids[j].b;
			float dist = sqrt(r*r + g*g + b*b);
			if(dist < closest_centroid_distance){
				closest_centroid = j;
				closest_centroid_distance = dist;
			}
		}
		int row = pixelList[i].pos / image.cols;
		int col = pixelList[i].pos % image.cols;
		result.at<cv::Vec3f>(row, col)[0] = centeroids[closest_centroid].r;
		result.at<cv::Vec3f>(row, col)[1] = centeroids[closest_centroid].g;
		result.at<cv::Vec3f>(row, col)[2] = centeroids[closest_centroid].b;
	}
  //printMatrix(result);
	return result;
}
