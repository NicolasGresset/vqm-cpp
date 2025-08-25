#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

template <typename ResultContainer, typename BlockResult>
class FeatureExtractor {
public:
  int block_length;
  int temporal_depth;
  virtual ~FeatureExtractor() = default;

  FeatureExtractor(int block_length, int temporal_depth)
      : block_length(block_length), temporal_depth(temporal_depth) {}

  // compute a feature for this whole S-T region
  ResultContainer compute(const std::vector<cv::Mat> &slice);
  // overload for Cb and Cr planes
  // virtual std::pair<double, double> compute(const cv::Mat &Cb,
  //                                           const cv::Mat &Cr) const = 0;

protected:
  inline float apply_perceptibility_threshold(float value, float threshold) {
    return (value < threshold ? threshold : value);
  }
  virtual ResultContainer initialize_result(int rows, int cols) const = 0;
  // it must be thread safe, i.e only assigning result at by and bx
  virtual void fill_result(ResultContainer &result, int by, int bx,
                           const BlockResult &block_result) const = 0;
  virtual BlockResult process_block(std::vector<float> &block_values) = 0;

private:
  std::vector<float> collect_block_values(const std::vector<cv::Mat> &slice,
                                          int block_y, int block_x);
};

// class ChromaExtractor : FeatureExtractor {
// public:
//   ChromaExtractor() : FeatureExtractor(8, 1, 12) {}

//   inline double compute(cv::Mat[][3]) {
//     assert(false && "Should not be used in this class");
//   }
//   std::pair<double, double> compute(const cv::Mat &Cb, const cv::Mat &Cr)
//   const;
// };

class SiExtractor : public FeatureExtractor<std::pair<cv::Mat, cv::Mat>,
                                            std::pair<float, float>> {
public:
  SiExtractor() : FeatureExtractor(8, 6) {}

private:
  static constexpr float SI_LOSS_THRESHOLD = 12;
  static constexpr float SI_GAIN_THRESHOLD = 8;

  std::pair<float, float>
  process_block(std::vector<float> &block_values) override;
  void fill_result(std::pair<cv::Mat, cv::Mat> &result, int block_y,
                   int block_x,
                   const std::pair<float, float> &block_result) const override;
  inline std::pair<cv::Mat, cv::Mat> initialize_result(int rows,
                                                       int cols) const override;
};

template class FeatureExtractor<std::pair<cv::Mat, cv::Mat>,
                                std::pair<float, float>>;