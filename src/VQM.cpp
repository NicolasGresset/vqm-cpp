#include "VQM.hpp"
#include "FeatureExtractor.hpp"
#include <cmath>
#include <cstddef>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


// std::pair<double, double> ChromaExtractor::compute(const cv::Mat &Cb,
//                                                    const cv::Mat &Cr) const {
//   double meanCb = cv::mean(Cb)[0];
//   double meanCr = cv::mean(Cr)[0] * 1.5;

//   return {meanCb, meanCr};
// }

double ParameterCalculator::ratio(double original_feature,
                                  double processed_feature) {
  return (processed_feature - original_feature) / original_feature;
}

double ParameterCalculator::loss(double value) { return value > 0 ? 0 : value; }

double ParameterCalculator::log(double original_feature,
                                double processed_feature) {
  return std::log10(processed_feature / original_feature);
}

double ParameterCalculator::gain(double value) { return value < 0 ? 0 : value; }

double ParameterCalculator::euclidean_dist(double original_feature1,
                                           double original_feature2,
                                           double processed_feature1,
                                           double processed_feature2) {
  double diff1 = (original_feature1 - processed_feature1);
  double diff1_squared = diff1 * diff1;
  double diff2 = original_feature2 - processed_feature2;
  double diff2_squared = diff2 * diff2;
  return std::sqrt(diff1_squared + diff2_squared);
}

void VQM::compute_batch_parameter(
    std::vector<std::array<cv::Mat, 3>> &original_frames,
    std::vector<std::array<cv::Mat, 3>> &processed_frames) {
  // printf("Computing batch parameters, for round %zu\n", batch_index);
  size_t batch_size = original_frames.size();
  std::vector<cv::Mat> original_si_image(batch_size),
      original_hv_image(batch_size), original_hv_bar_image(batch_size);
  perform_si_hv_bar_filtering(original_frames, original_si_image,
                              original_hv_image, original_hv_bar_image);
  std::vector<cv::Mat> processed_si_image(batch_size),
      processed_hv_image(batch_size), processed_hv_bar_image(batch_size);
  perform_si_hv_bar_filtering(processed_frames, processed_si_image,
                              processed_hv_image, processed_hv_bar_image);

  std::vector<cv::Mat> original_ati_frames, processed_ati_frames;
  perform_ati_filtering(original_frames, processed_ati_frames);
  perform_ati_filtering(processed_frames, processed_ati_frames);

  SiExtractor extractor;
  std::pair<cv::Mat, cv::Mat> res = extractor.compute(processed_si_image);
  printf("si feature computed\n");
  std::cout << res.first << std::endl;

  batch_index++;
  return;
}

void VQM::temporally_collapse(void) {}

void VQM::perform_si_hv_bar_filtering(std::vector<std::array<cv::Mat, 3>> &base,
                                      std::vector<cv::Mat> &si,
                                      std::vector<cv::Mat> &hv,
                                      std::vector<cv::Mat> &hv_bar) {
#pragma omp parallel for
  for (size_t i = 0; i < base.size(); i++) {
    cv::Mat horizontal, vertical;
    perform_si_filter(base[i][Y], si[i], horizontal, vertical);
    perform_hv_bar_filter(si[i], horizontal, vertical, hv[i], hv_bar[i]);
  }
}

void VQM::perform_si_filter(cv::Mat &base_Y, cv::Mat &si, cv::Mat &horizontal,
                            cv::Mat &vertical) {
  const cv::Mat ONES_COL = cv::Mat::ones(13, 1, CV_32F);
  const cv::Mat ONES_ROW = cv::Mat::ones(1, 13, CV_32F);

  horizontal.create(base_Y.size(), base_Y.type());
  vertical.create(base_Y.size(), base_Y.type());
  si.create(base_Y.size(), base_Y.type());

  cv::sepFilter2D(base_Y, horizontal, -1, FILTER_KERNEL_H, ONES_COL);
  cv::sepFilter2D(base_Y, vertical, -1, ONES_ROW, FILTER_KERNEL_V);

  cv::Mat h2 = horizontal.mul(horizontal);
  cv::Mat v2 = vertical.mul(vertical);

  cv::sqrt(h2 + v2, si);
}

void VQM::perform_hv_bar_filter(cv::Mat &si, cv::Mat &horizontal,
                                cv::Mat &vertical, cv::Mat &hv,
                                cv::Mat &hv_bar) {

  horizontal = cv::abs(horizontal);
  vertical = cv::abs(vertical);
  cv::Mat ratio;
  cv::divide(min(horizontal, vertical), max(horizontal, vertical), ratio);

  cv::Mat rat_t, base;
  // apply perceptability threshold
  cv::threshold(si, base, R_MIN, -1, cv::THRESH_TOZERO);
  // hv and hv bar criteria
  cv::threshold(ratio, rat_t, RATIO_THRESHOLD_RADIANS, 1, cv::THRESH_BINARY);

  /* rat_t now contains 0 is the threshold is not exceeded, 1 otherwise... */
  hv = base.mul(rat_t);
  cv::subtract(base, hv, hv_bar);
}

void VQM::perform_ati_filtering(std::vector<std::array<cv::Mat, 3>> &base,
                                std::vector<cv::Mat> &ati) {
  ati.resize(base.size());

  if (batch_index == 0) {
    ati[0] = cv::Mat::zeros(base[0][Y].size(), base[0][Y].type());
  } else {
    cv::absdiff(base[0][Y], last_frame, ati[0]);
  }

#pragma omp parallel for
  for (size_t i = 1; i < base.size(); i++) {
    cv::absdiff(base[i][Y], base[i - 1][Y], ati[i]);
  }

  // update last frame for next batch computation
  // THIS IS NOT PARALLEL FRIENDLY FOR MULTI BATCH COMPUTATION
  last_frame = base[base.size() - 1][Y];
}