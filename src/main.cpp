#include "Options.hpp"
#include "VQM.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {

  Options options(argc, argv);

  options.print();
  // read cli arguments (video original path, video processed path)
  //
  // check if it exists indeed
  //
  // video must have same number of frames, check that
  //
  //
  // perform calibration between those streams if a CLI flag was specified :
  // PVR, spatial(unlikely I will implement this), temporal
  // (requires additional flag) ; and gain level offset compensation
  //
  // for each S-T region of each batch (6 frames for T=0.2 seconds, 30 FPS):
  // whole frame, N (corresponding to T) number of frames of both video (both
  // videos can be computed separately), compute feature extraction for each S-T
  // that is, for each feature, we have an array of features values for both
  // videos
  //  this can be done in parallel (for each batch, and for each S-T region
  //  within the batch)
  //
  // then, the parameter can be extracted for each feature by comparing original
  // stream and processed stream
  //
  // afterwards, those parameters can be spatially collapsed (inside S-T
  // regions, typically 6 frames) for each feature
  //
  // then, each feature should be temporally collapsed ; and the duration to
  // colllapse (a multiple of 0.2 seconds (longest T-sequence), maximum duartion
  // of the video, which is also default value) should be determined from CLI
  //
  // then we can output the score to stdout in plain text or store it to json
  // format (why not ?)
  //
  //
}
