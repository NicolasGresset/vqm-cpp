#pragma once
#include <string>

class Calibration {
public:
  // writes the processed calibrated video to output_path
  static void perform_gain_level_offset_calibration(std::string original_path,
                                                    std::string processed_path,
                                                    std::string output_path) {}

  // writes the temporally aligned video to output_path
  static void perform_temporal_alignement(std::string original_path,
                                          std::string processed_path,
                                          std::string output_path) {}
};