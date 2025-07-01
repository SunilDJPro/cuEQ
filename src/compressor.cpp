#include "compressor.h"
#include <cmath>
#include <algorithm>

namespace EQProcessor {

Compressor::Compressor(float threshold_db, float ratio, float attack_ms, 
                      float release_ms, int sample_rate)
    : threshold_linear_(dbToLinear(threshold_db)), ratio_(ratio),
      envelope_(1.0f), sample_rate_(sample_rate) {
  
  // Convert attack/release times to coefficients
  attack_coeff_ = std::exp(-1.0f / (attack_ms * 0.001f * sample_rate));
  release_coeff_ = std::exp(-1.0f / (release_ms * 0.001f * sample_rate));
}

void Compressor::ProcessSamples(std::vector<float>* samples, int num_channels) {
  for (size_t i = 0; i < samples->size(); i += num_channels) {
    // Calculate peak level across all channels
    float peak_level = 0.0f;
    for (int ch = 0; ch < num_channels; ++ch) {
      peak_level = std::max(peak_level, std::abs((*samples)[i + ch]));
    }
    
    // Update envelope follower
    const float target_envelope = peak_level > threshold_linear_ ? peak_level : envelope_;
    const float coeff = peak_level > envelope_ ? attack_coeff_ : release_coeff_;
    envelope_ = target_envelope + (envelope_ - target_envelope) * coeff;
    
    // Calculate gain reduction
    float gain_reduction = 1.0f;
    if (envelope_ > threshold_linear_) {
      const float over_threshold = LinearTodb(envelope_) - LinearTodb(threshold_linear_);
      const float compressed_over = over_threshold / ratio_;
      const float target_level_db = LinearTodb(threshold_linear_) + compressed_over;
      gain_reduction = dbToLinear(target_level_db) / envelope_;
    }
    
    // Apply gain reduction to all channels
    for (int ch = 0; ch < num_channels; ++ch) {
      (*samples)[i + ch] *= gain_reduction;
    }
  }
}

float Compressor::dbToLinear(float db) const {
  return std::pow(10.0f, db / 20.0f);
}

float Compressor::LinearTodb(float linear) const {
  return 20.0f * std::log10(std::max(linear, 1e-6f));
}

}  // namespace EQProcessor