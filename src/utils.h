// -------------------------------------------------------------------------------------------
// Saideep Arikontham
// Feb 2025
// Pattern Recognition and Computer Vision - Project 2
// -------------------------------------------------------------------------------------------

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Function declarations
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);


int get_feature_vector_7x7(char img_file[256], std::vector<float> &img_data);
int get_feature_vector_rg_hist(char img_file[256], std::vector<float> &img_data);
int get_feature_vector_two_rgb_hist(char img_file[256], std::vector<float> &img_data);
int get_feature_vector_rgb_texture_hist(char img_file[256], std::vector<float> &img_data);
int get_feature_vector_depth_rgb_hist(char img_file[256], std::vector<float> &img_data);
int get_feature_vector_dnn(char img_file[256], std::vector<float> &img_data);
int get_feature_vector_depth_dnn(char img_file[256], std::vector<float> &img_data);
int get_feature_vector_four_rgb_hist(char img_file[256], std::vector<float> &img_data);

int get_rg_histogram(cv::Mat &src, cv::Mat &hist, const int histsize);
int get_rgb_histogram(cv::Mat &src, cv::MatND &hist, const int histsize);
int get_texture_histogram(cv::Mat &src, cv::MatND &hist, const int histsize);
int get_depth(cv::Mat &src, cv::Mat &dst);
int get_depth_rgb_histogram(cv::Mat &src, cv::MatND &hist, const int histsize);
int get_embedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net);
int get_depth_embedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net);

typedef int (*FeatureVectorWriter)(char*, std::vector<float>&);
int write_feature_vectors(char dirname[256], char feature_file[256], FeatureVectorWriter write_feature_vector);

float sum_of_squared_difference(std::vector<float> &img_data1, std::vector<float> &img_data2);
float histogram_intersection(std::vector<float> &img_data1, std::vector<float> &img_data2);
double chiSquareDistance(std::vector<float> &histA, std::vector<float> &histB);
double cosine_similarity(std::vector<float>& A, std::vector<float>& B);
double computeCosineDistance(std::vector<float>& img_data1, std::vector<float>& img_data2);
float earth_movers_distance(std::vector<float>& img_data1, std::vector<float>& img_data2);

int read_image(char img_file[256], cv::Mat &src);
int display_image(std::string display_name, cv::Mat &img);
int loop_until_q_pressed();
int get_top_N_matches(const std::vector<char*> &filenames, const std::vector<float> &ssd_data, int N, bool ascending = true);
#endif // UTILS_H
