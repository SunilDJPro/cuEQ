//==============================================================================
// src/wav_io.cpp - WITH DEBUG OUTPUT
//==============================================================================
#include "wav_io.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace EQProcessor {

bool WavIO::ReadWavFile(const std::string& filename, AudioData* audio_data) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open input file: " << filename << "\n";
    return false;
  }
  
  // Read WAV header
  WavHeader& header = audio_data->header;
  file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));
  
  if (strncmp(header.chunk_id, "RIFF", 4) != 0 ||
      strncmp(header.format, "WAVE", 4) != 0) {
    std::cerr << "Error: Invalid WAV file format\n";
    return false;
  }
  
  if (header.audio_format != 1) {
    std::cerr << "Error: Only PCM format is supported\n";
    return false;
  }
  
  // Debug WAV header information
  std::cout << "WAV Header Debug:\n";
  std::cout << "  Audio format: " << header.audio_format << "\n";
  std::cout << "  Channels: " << header.num_channels << "\n";
  std::cout << "  Sample rate: " << header.sample_rate << "\n";
  std::cout << "  Bits per sample: " << header.bits_per_sample << "\n";
  std::cout << "  Byte rate: " << header.byte_rate << "\n";
  std::cout << "  Block align: " << header.block_align << "\n";
  std::cout << "  Data size: " << header.subchunk2_size << " bytes\n";
  
  // Read audio data
  const size_t data_size = header.subchunk2_size;
  std::vector<uint8_t> raw_data(data_size);
  file.read(reinterpret_cast<char*>(raw_data.data()), data_size);
  
  // Debug raw data
  std::cout << "WAV Raw Data Debug:\n";
  std::cout << "  Raw data size: " << data_size << " bytes\n";
  std::cout << "  Bytes actually read: " << file.gcount() << "\n";
  
  // Check first few raw bytes
  std::cout << "  First 16 raw bytes: ";
  for (size_t i = 0; i < std::min(data_size, size_t(16)); ++i) {
    std::cout << static_cast<int>(raw_data[i]) << " ";
  }
  std::cout << "\n";
  
  // Convert to float
  ConvertToFloat(raw_data, &audio_data->samples, header.bits_per_sample);
  
  // Debug converted data
  std::cout << "WAV Conversion Debug:\n";
  std::cout << "  Converted samples: " << audio_data->samples.size() << "\n";
  std::cout << "  Expected samples: " << (data_size / (header.bits_per_sample / 8)) << "\n";
  
  // Check sample levels
  float max_sample = 0.0f;
  float min_sample = 0.0f;
  float sum_abs = 0.0f;
  for (size_t i = 0; i < std::min(audio_data->samples.size(), size_t(10000)); ++i) {
    float sample = audio_data->samples[i];
    max_sample = std::max(max_sample, sample);
    min_sample = std::min(min_sample, sample);
    sum_abs += std::abs(sample);
  }
  float avg_abs = sum_abs / std::min(audio_data->samples.size(), size_t(10000));
  
  std::cout << "  Sample range: [" << min_sample << ", " << max_sample << "]\n";
  std::cout << "  Average absolute level: " << avg_abs << "\n";
  
  // Show first few converted samples
  std::cout << "  First 10 samples: ";
  for (size_t i = 0; i < std::min(audio_data->samples.size(), size_t(10)); ++i) {
    std::cout << audio_data->samples[i] << " ";
  }
  std::cout << "\n";
  
  audio_data->num_channels = header.num_channels;
  audio_data->sample_rate = header.sample_rate;
  audio_data->num_samples = audio_data->samples.size() / header.num_channels;
  
  std::cout << "Loaded: " << audio_data->num_samples << " samples, "
            << audio_data->num_channels << " channels, "
            << audio_data->sample_rate << " Hz\n";
  
  // Final validation
  if (audio_data->samples.empty()) {
    std::cerr << "ERROR: No audio samples loaded!\n";
    return false;
  }
  
  if (max_sample == 0.0f && min_sample == 0.0f) {
    std::cerr << "WARNING: All samples are zero - silent audio detected!\n";
  }
  
  return true;
}

bool WavIO::WriteWavFile(const std::string& filename, 
                        const AudioData& audio_data) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot create output file: " << filename << "\n";
    return false;
  }
  
  // Debug output data before writing
  std::cout << "WAV Write Debug:\n";
  std::cout << "  Samples to write: " << audio_data.samples.size() << "\n";
  std::cout << "  Channels: " << audio_data.num_channels << "\n";
  std::cout << "  Sample rate: " << audio_data.sample_rate << "\n";
  std::cout << "  Bits per sample: " << audio_data.header.bits_per_sample << "\n";
  
  // Check output sample levels
  float max_output = 0.0f;
  float min_output = 0.0f;
  float sum_abs_output = 0.0f;
  size_t non_zero_count = 0;
  
  for (size_t i = 0; i < std::min(audio_data.samples.size(), size_t(10000)); ++i) {
    float sample = audio_data.samples[i];
    max_output = std::max(max_output, sample);
    min_output = std::min(min_output, sample);
    sum_abs_output += std::abs(sample);
    if (std::abs(sample) > 1e-6f) {
      non_zero_count++;
    }
  }
  float avg_abs_output = sum_abs_output / std::min(audio_data.samples.size(), size_t(10000));
  
  std::cout << "  Output sample range: [" << min_output << ", " << max_output << "]\n";
  std::cout << "  Output average absolute level: " << avg_abs_output << "\n";
  std::cout << "  Non-zero samples in first 10k: " << non_zero_count << "\n";
  
  // Show first few output samples
  std::cout << "  First 10 output samples: ";
  for (size_t i = 0; i < std::min(audio_data.samples.size(), size_t(10)); ++i) {
    std::cout << audio_data.samples[i] << " ";
  }
  std::cout << "\n";
  
  // Warning if output is silent
  if (max_output == 0.0f && min_output == 0.0f) {
    std::cerr << "WARNING: Output audio is completely silent!\n";
  }
  
  if (non_zero_count == 0) {
    std::cerr << "WARNING: No non-zero samples found in output!\n";
  }
  
  // Convert float samples back to integer format
  std::vector<uint8_t> raw_data;
  ConvertFromFloat(audio_data.samples, &raw_data, 
                  audio_data.header.bits_per_sample);
  
  // Debug converted raw data
  std::cout << "  Converted raw data size: " << raw_data.size() << " bytes\n";
  
  // Check first few converted bytes
  std::cout << "  First 16 converted bytes: ";
  for (size_t i = 0; i < std::min(raw_data.size(), size_t(16)); ++i) {
    std::cout << static_cast<int>(raw_data[i]) << " ";
  }
  std::cout << "\n";
  
  // Update header with correct sizes
  WavHeader header = audio_data.header;
  header.subchunk2_size = raw_data.size();
  header.chunk_size = 36 + header.subchunk2_size;
  
  std::cout << "  Writing header with data size: " << header.subchunk2_size << " bytes\n";
  
  // Write header and data
  file.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));
  file.write(reinterpret_cast<const char*>(raw_data.data()), raw_data.size());
  
  if (!file.good()) {
    std::cerr << "Error: Failed to write WAV file data\n";
    return false;
  }
  
  std::cout << "WAV file written successfully: " << filename << "\n";
  return true;
}

void WavIO::ConvertToFloat(const std::vector<uint8_t>& raw_data,
                          std::vector<float>* float_data,
                          int bits_per_sample) {
  const size_t num_samples = raw_data.size() / (bits_per_sample / 8);
  float_data->resize(num_samples);
  
  std::cout << "Float Conversion Debug:\n";
  std::cout << "  Bits per sample: " << bits_per_sample << "\n";
  std::cout << "  Raw data bytes: " << raw_data.size() << "\n";
  std::cout << "  Samples to convert: " << num_samples << "\n";
  
  if (bits_per_sample == 16) {
    const auto* samples = reinterpret_cast<const int16_t*>(raw_data.data());
    std::cout << "  Converting 16-bit samples\n";
    std::cout << "  First few 16-bit values: ";
    for (size_t i = 0; i < std::min(num_samples, size_t(5)); ++i) {
      std::cout << samples[i] << " ";
    }
    std::cout << "\n";
    
    for (size_t i = 0; i < num_samples; ++i) {
      (*float_data)[i] = static_cast<float>(samples[i]) / 32768.0f;
    }
  } else if (bits_per_sample == 24) {
    std::cout << "  Converting 24-bit samples\n";
    for (size_t i = 0; i < num_samples; ++i) {
      const size_t byte_idx = i * 3;
      int32_t sample = (raw_data[byte_idx + 2] << 16) |
                      (raw_data[byte_idx + 1] << 8) |
                      raw_data[byte_idx];
      if (sample & 0x800000) sample |= 0xFF000000;  // Sign extend
      (*float_data)[i] = static_cast<float>(sample) / 8388608.0f;
    }
  } else if (bits_per_sample == 32) {
    const auto* samples = reinterpret_cast<const int32_t*>(raw_data.data());
    std::cout << "  Converting 32-bit samples\n";
    for (size_t i = 0; i < num_samples; ++i) {
      (*float_data)[i] = static_cast<float>(samples[i]) / 2147483648.0f;
    }
  } else {
    std::cerr << "Error: Unsupported bit depth: " << bits_per_sample << "\n";
    float_data->clear();
    return;
  }
  
  std::cout << "  Conversion completed\n";
}

void WavIO::ConvertFromFloat(const std::vector<float>& float_data,
                           std::vector<uint8_t>* raw_data,
                           int bits_per_sample) {
  const size_t num_samples = float_data.size();
  raw_data->resize(num_samples * (bits_per_sample / 8));
  
  std::cout << "Float to Raw Conversion Debug:\n";
  std::cout << "  Float samples: " << num_samples << "\n";
  std::cout << "  Bits per sample: " << bits_per_sample << "\n";
  std::cout << "  Output raw bytes: " << raw_data->size() << "\n";
  
  if (bits_per_sample == 16) {
    auto* samples = reinterpret_cast<int16_t*>(raw_data->data());
    std::cout << "  Converting to 16-bit\n";
    for (size_t i = 0; i < num_samples; ++i) {
      float clamped = std::clamp(float_data[i], -1.0f, 1.0f);
      samples[i] = static_cast<int16_t>(clamped * 32767.0f);
    }
    
    // Show first few converted values
    std::cout << "  First few 16-bit outputs: ";
    for (size_t i = 0; i < std::min(num_samples, size_t(5)); ++i) {
      std::cout << samples[i] << " ";
    }
    std::cout << "\n";
    
  } else if (bits_per_sample == 24) {
    std::cout << "  Converting to 24-bit\n";
    for (size_t i = 0; i < num_samples; ++i) {
      float clamped = std::clamp(float_data[i], -1.0f, 1.0f);
      int32_t sample = static_cast<int32_t>(clamped * 8388607.0f);
      const size_t byte_idx = i * 3;
      (*raw_data)[byte_idx] = sample & 0xFF;
      (*raw_data)[byte_idx + 1] = (sample >> 8) & 0xFF;
      (*raw_data)[byte_idx + 2] = (sample >> 16) & 0xFF;
    }
  } else if (bits_per_sample == 32) {
    auto* samples = reinterpret_cast<int32_t*>(raw_data->data());
    std::cout << "  Converting to 32-bit\n";
    for (size_t i = 0; i < num_samples; ++i) {
      float clamped = std::clamp(float_data[i], -1.0f, 1.0f);
      samples[i] = static_cast<int32_t>(clamped * 2147483647.0f);
    }
  } else {
    std::cerr << "Error: Unsupported bit depth: " << bits_per_sample << "\n";
    raw_data->clear();
    return;
  }
  
  std::cout << "  Raw conversion completed\n";
}

}  // namespace EQProcessor