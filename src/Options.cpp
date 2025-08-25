#include "Options.hpp"

Options::Options(int argc, char **argv) { parseCLI(argc, argv); }
void Options::printUsage(const char *progName) const {
  std::cerr << "Usage: " << progName
            << " -r <reference> -p <processed> [-t] [-c] [-s <value>]\n";
  std::cerr << "  -r, --reference <path>   Path to reference file (required)\n";
  std::cerr << "  -p, --processed <path>   Path to processed file (required)\n";
  std::cerr << "  -t                       Enable temporal calibration "
               "(optional)\n";
  std::cerr << "  -c                       Enable calibration (PVR & Gain / "
               "Level offset) flag (optional)\n";
  std::cerr << "  -s <value>               Temporal collapse step (float/int, "
               "optional, "
               "default=1.0)\n";
}

void Options::print() const {
  std::cout << "Reference path : " << original_path << "\n";
  std::cout << "Processed path : " << processed_path << "\n";
  std::cout << "Enable temporal calibration : "
            << (temporal_calibration ? "true" : "false") << "\n";
  std::cout << "Enable calibration (PVR & Gain / "
               "Level offset) : "
            << (calibration ? "true" : "false") << "\n";
  std::cout << "Temporal collapse step (-s) : " << temporal_collapse_step
            << "\n";
}

void Options::parseCLI(int argc, char **argv) {
  if (argc < 5) {
    printUsage(argv[0]);
    std::exit(1);
  }

  std::vector<std::string> tokens(argv + 1, argv + argc);

  for (size_t i = 0; i < tokens.size(); ++i) {
    const std::string &token = tokens[i];

    if (token == "-r" || token == "--reference") {
      if (i + 1 >= tokens.size()) {
        std::cerr << "Error: Missing value after " << token << "\n";
        printUsage(argv[0]);
        std::exit(1);
      }
      original_path = tokens[++i];
    } else if (token == "-p" || token == "--processed") {
      if (i + 1 >= tokens.size()) {
        std::cerr << "Error: Missing value after " << token << "\n";
        printUsage(argv[0]);
        std::exit(1);
      }
      processed_path = tokens[++i];
    } else if (token == "-t") {
      temporal_calibration = true;
    } else if (token == "-c") {
      calibration = true;
    } else if (token == "-s") {
      if (i + 1 >= tokens.size()) {
        std::cerr << "Error: Missing value after -s\n";
        printUsage(argv[0]);
        std::exit(1);
      }
      temporal_collapse_step = std::atof(tokens[++i].c_str());
    } else {
      std::cerr << "Error: Unknown argument: " << token << "\n";
      printUsage(argv[0]);
      std::exit(1);
    }
  }

  if (original_path.empty()) {
    std::cerr << "Error: Missing required flag -r / --reference\n";
    printUsage(argv[0]);
    std::exit(1);
  }
  if (processed_path.empty()) {
    std::cerr << "Error: Missing required flag -p / --processed\n";
    printUsage(argv[0]);
    std::exit(1);
  }
}
