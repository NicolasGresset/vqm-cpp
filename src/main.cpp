#include "Calibration.hpp"
#include "Options.hpp"
#include "VQM.hpp"
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <ostream>
#include <vector>

inline void checkOrExit(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "Error: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

std::vector<std::array<cv::Mat, 3>> readBatchFrames(cv::VideoCapture &cap,
                                                    int batch_size) {
  // this can not be parallelized because frames reading is intrinsically
  // sequential in OpenCV
  // therefore we can only treat 6 frames per 6 frames I fear
  std::vector<std::array<cv::Mat, 3>> frames;
  frames.reserve(batch_size); // optimise allocations

  for (int i = 0; i < batch_size; i++) {
    cv::Mat frame;
    if (!cap.read(frame)) {
      break; // no more frames
    }

    std::array<cv::Mat, 3> channels;
    cv::split(frame, channels);
    for (int c = 0; c < 3; ++c) {
      channels[c].convertTo(channels[c], CV_32F);
    }
    frames.push_back(channels);
  }

  // printf("Just read %zu frames\n", frames.size());
  return frames;
}

cv::VideoCapture openVideo(std::string path) {
  cv::VideoCapture capture(path);
  checkOrExit(capture.isOpened(), "Unable to open video : " + path);
  return capture;
}

int main(int argc, char **argv) {

  // read cli arguments (video original path, video processed path)
  Options options(argc, argv);
  options.print();

  cv::VideoCapture original_capture = openVideo(options.original_path);
  cv::VideoCapture processed_capture = openVideo(options.processed_path);
  checkOrExit(original_capture.get(cv::CAP_PROP_FRAME_COUNT) ==
                  processed_capture.get(cv::CAP_PROP_FRAME_COUNT),
              "Both videos have different number of frames, exiting ...");

  // perform calibration
  // TODO: enhance this logic
  std::string processed_calibrated_path = "tmp.mp4";

  if (options.calibration) {
    Calibration::perform_gain_level_offset_calibration(
        options.original_path, options.processed_path,
        processed_calibrated_path);
  }

  if (options.temporal_calibration) {
    std::string input_processed_sequence = options.calibration
                                               ? processed_calibrated_path
                                               : options.processed_path;
    Calibration::perform_gain_level_offset_calibration(
        options.original_path, input_processed_sequence,
        processed_calibrated_path);
  }
  int total_frames =
      static_cast<int>(original_capture.get(cv::CAP_PROP_FRAME_COUNT));
  printf("there are %d total frames\n", total_frames);
  VQM VQM{total_frames};

  std::vector<std::array<cv::Mat, 3>> original_frames;
  std::vector<std::array<cv::Mat, 3>> processed_frames;

  int frames_handled{0};
  while (frames_handled < total_frames) {
    original_frames = readBatchFrames(original_capture, VQM.FRAMES_PER_BATCH);
    processed_frames = readBatchFrames(processed_capture, VQM.FRAMES_PER_BATCH);

    VQM.compute_batch_parameter(original_frames, processed_frames);

    frames_handled += original_frames.size();
  }

  VQM.temporally_collapse();

  // aggregate those 0.2 seconds (= 6 frames) values for the time aggregation
  // parameter specified via CLI??
}
