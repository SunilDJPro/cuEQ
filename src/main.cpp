#include <iostream>
#include <string>
#include "audio_processor.h"

void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name 
            << " <input_file.wav> <output_file.wav>\n";
  std::cout << "Applies 31-band ISO equalizer with GPU acceleration\n";
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    PrintUsage(argv[0]);
    return 1;
  }
  
  const std::string input_file = argv[1];
  const std::string output_file = argv[2];
  
  // Validate file extensions
  if (input_file.substr(input_file.find_last_of(".") + 1) != "wav" ||
      output_file.substr(output_file.find_last_of(".") + 1) != "wav") {
    std::cerr << "Error: Only .wav files are supported\n";
    return 1;
  }
  
  std::cout << "Processing: " << input_file << " -> " << output_file << "\n";
  
  EQProcessor::AudioProcessor processor;
  if (!processor.ProcessFile(input_file, output_file)) {
    std::cerr << "Error: Failed to process audio file\n";
    return 1;
  }
  
  std::cout << "Processing completed successfully!\n";
  return 0;
}
