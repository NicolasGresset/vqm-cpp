#include "FeatureExtractor.hpp"

template <typename ResultContainer, typename BlockResult>
ResultContainer FeatureExtractor<ResultContainer, BlockResult>::compute(
    const std::vector<cv::Mat> &slice) {
  int rows_number = slice[0].rows;
  int cols_number = slice[0].cols;

  ResultContainer result =
      initialize_result(rows_number / block_length, cols_number / block_length);

  // we treat block rows per block rows in order not to create one thread per
  // block
#pragma omp parallel for
  for (int block_y = 0; block_y < rows_number; block_y += block_length) {
    for (int block_x = 0; block_x < cols_number; block_x += block_length) {
      std::vector<float> block_values =
          collect_block_values(slice, block_y, block_x);

      BlockResult block_result = process_block(block_values);
      fill_result(result, block_y / block_length, block_x / block_length,
                  block_result);
    }
  }

  return result;
}

template <typename ResultContainer, typename BlockResult>
std::vector<float>
FeatureExtractor<ResultContainer, BlockResult>::collect_block_values(
    const std::vector<cv::Mat> &slice, int block_y, int block_x) {
  int rows_number = slice[0].rows;
  int cols_number = slice[0].cols;

  std::vector<float> block_values;
  block_values.reserve(block_length * block_length * temporal_depth);

  for (int frame = 0; frame < temporal_depth; frame++) {
    for (int y = 0; y < block_length && block_y + y < rows_number; y++) {
      const float *rowPtr = slice[frame].ptr<float>(block_y + y);
      for (int x = 0; x < block_length && block_x + x < cols_number; x++) {
        block_values.push_back(rowPtr[block_x + x]);
      }
    }
  }

  return block_values;
}

std::pair<float, float>
SiExtractor::process_block(std::vector<float> &block_values) {
  float sum = 0.0;
  for (float value : block_values)
    sum += value;
  float mean = sum / block_values.size();

  float var = 0.0;
  for (float value : block_values) {
    float diff = value - mean;
    var += diff * diff;
  }
  var /= block_values.size();

  float stddev = std::sqrt(var);

  return {apply_perceptibility_threshold(stddev, SI_LOSS_THRESHOLD),
          apply_perceptibility_threshold(stddev, SI_GAIN_THRESHOLD)};
}

void SiExtractor::fill_result(
    std::pair<cv::Mat, cv::Mat> &result, int block_y, int block_x,
    const std::pair<float, float> &block_result) const {
  result.first.at<float>(block_y, block_x) = block_result.first;
  result.second.at<float>(block_y, block_x) = block_result.second;
}

std::pair<cv::Mat, cv::Mat> SiExtractor::initialize_result(int rows,
                                                           int cols) const {
  cv::Mat mat1(rows / block_length, cols / block_length, CV_32F);
  return {mat1, mat1.clone()};
}