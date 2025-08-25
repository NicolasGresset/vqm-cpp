#pragma once

#include <array>
#include <iterator>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class VQM {
public:
  int frames_number;
  const int FRAMES_PER_BATCH = 6;
  VQM(int frames_number)
      : frames_number(frames_number), RATIO_THRESHOLD_RADIANS{tan(225)} {}

  // apply filters to this batch, compute features, parameter and spatially
  // collapse them

  void compute_batch_parameter(
      std::vector<std::array<cv::Mat, 3>> &original_frames,
      std::vector<std::array<cv::Mat, 3>> &processed_frames);

  void temporally_collapse(void);

  void compute_VQM_score(void);

  const double RATIO_THRESHOLD_RADIANS;

private:
  std::size_t batch_index = 0;
  cv::Mat last_frame;

  static constexpr int Y = 0;
  static constexpr int Cb = 1;
  static constexpr int Cr = 2;

  // mask values were doublechecked
  static constexpr std::array<float, 13> FILTER_MASK = {
      -0.0052625f, -0.0173446f, -0.0427401f, -0.0768961f, -0.0957739f,
      -0.0696751f, 0.0f,        0.0696751f,  0.0957739f,  0.0768961f,
      0.0427401f,  0.0173446f,  0.0052625f};
  const cv::Mat FILTER_KERNEL_H =
      cv::Mat(1, 13, CV_32F, const_cast<float *>(FILTER_MASK.data()));
  const cv::Mat FILTER_KERNEL_V =
      cv::Mat(13, 1, CV_32F, const_cast<float *>(FILTER_MASK.data()));

  void perform_si_hv_bar_filtering(std::vector<std::array<cv::Mat, 3>> &base,
                                   std::vector<cv::Mat> &si,
                                   std::vector<cv::Mat> &hv,
                                   std::vector<cv::Mat> &hv_bar);

  void perform_si_filter(cv::Mat &base_Y, cv::Mat &si, cv::Mat &horizontal,
                         cv::Mat &vertical);

  void perform_ati_filtering(std::vector<std::array<cv::Mat, 3>> &base,
                             std::vector<cv::Mat> &ati);

  const double R_MIN = 20;
  void perform_hv_bar_filter(cv::Mat &si, cv::Mat &horizontal,
                             cv::Mat &vertical, cv::Mat &hv, cv::Mat &hv_bar);
  // results after spatial collapsing
  std::vector<double> si_loss_parameters;
  std::vector<double> si_gain_parameters;
  std::vector<double> hi_loss_parameters;
  std::vector<double> hi_gain_parameters;
  std::vector<double> chroma_spread_parameters;
  std::vector<double> chroma_extreme_parameters;
  std::vector<double> ct_ati_gain_parameters;
};


class ParameterCalculator {

  static double ratio(double original_feature, double processed_feature);
  static double loss(double value);
  static double log(double original_feature, double processed_feature);
  static double gain(double value);

  static double euclidean_dist(double original_feature1,
                               double original_feature2,
                               double processed_feature1,
                               double processed_feature2);
};
