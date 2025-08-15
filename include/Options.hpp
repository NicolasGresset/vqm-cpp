#pragma once
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

class Options {
public:
  std::string reference_path;
  std::string processed_path;
  bool temporal_calibration = false;
  bool calibration = false;
  double temporal_collapse_step = 1.0; // expressed in seconds

  Options(int argc, char **argv);
  void print(void) const;

private:
  void printUsage(const char *program_name) const;

  void parseCLI(int argc, char **argv);
};
