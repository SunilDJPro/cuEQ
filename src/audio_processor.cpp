#include "audio_processor.h"
#include "eq_config.h"
#include <iostream>
#include <chrono>

namespace EQProcessor {

AudioProcessor::AudioProcessor()
    : cuEQ_processor_(std::make_unique<cuEQProcessor>()),
      compressor_(nullptr) {}

AudioProcessor::~AudioProcessor() = default;

bool AudioProcessor::ProcessFile(const std::string& input_file, 
                                const std::string& output_file) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Load input audio file
  AudioData audio_data;
  if (!WavIO::ReadWavFile(input_file, &audio_data)) {
    return false;
  }
  
  // Initialize GPU EQ processor
  if (!cuEQ_processor_->Initialize(audio_data.sample_rate)) {
    std::cerr << "Error: Failed to initialize GPU EQ processor\n";
    return false;
  }
  
  // Initialize compressor
  compressor_ = std::make_unique<Compressor>(
      kThresholdDb, kRatio, kAttackMs, kReleaseMs, audio_data.sample_rate);
  
  std::cout << "Applying EQ processing...\n";
  
  // Apply EQ processing
  std::vector<float> processed_samples;
  if (!cuEQ_processor_->ProcessAudio(audio_data.samples, &processed_samples,
                                      audio_data.num_channels)) {
    std::cerr << "Error: EQ processing failed\n";
    return false;
  }
  
  std::cout << "Applying dynamic range compression...\n";
  
  // Apply compression to prevent clipping
  compressor_->ProcessSamples(&processed_samples, audio_data.num_channels);
  
  // Update audio data with processed samples
  audio_data.samples = std::move(processed_samples);
  
  // Save output file
  if (!WavIO::WriteWavFile(output_file, audio_data)) {
    return false;
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  
  std::cout << "Processing completed in " << duration.count() << " ms\n";
  std::cout << "Output saved to: " << output_file << "\n";
  
  return true;
}

}  // namespace EQProcessor