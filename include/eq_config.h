#ifndef EQ_CONFIG_H_
#define EQ_CONFIG_H_

#include <array>

namespace EQProcessor {

// 31-band ISO equalizer configuration
constexpr int kNumEqBands = 31;
constexpr float kMinGainDb = -20.0f;
constexpr float kMaxGainDb = 20.0f;

// ISO center frequencies for 31-band equalizer (Hz)
constexpr std::array<float, kNumEqBands> kIsoCenterFreqs = {
    20.0f, 25.0f, 31.5f, 40.0f, 50.0f, 63.0f, 80.0f, 100.0f, 125.0f, 160.0f,
    200.0f, 250.0f, 315.0f, 400.0f, 500.0f, 630.0f, 800.0f, 1000.0f, 1250.0f,
    1600.0f, 2000.0f, 2500.0f, 3150.0f, 4000.0f, 5000.0f, 6300.0f, 8000.0f,
    10000.0f, 12500.0f, 16000.0f, 20000.0f
};

// EQ gains in dB - Configure your desired EQ curve here
// Example curve with some visible changes for testing
constexpr std::array<float, kNumEqBands> kEqGainsDb = {
    0.0f,   // 20 Hz - bass boost
    1.0f,   // 25 Hz
    2.0f,   // 31.5 Hz
    2.0f,   // 40 Hz
    3.0f,   // 50 Hz
    3.0f,   // 63 Hz
    1.0f,  // 80 Hz
    -2.0f,  // 100 Hz
    -1.0f,   // 125 Hz
    0.0f,   // 160 Hz
    1.0f,   // 200 Hz
    1.5f,   // 250 Hz
    2.0f,   // 315 Hz
    1.0f,  // 400 Hz
    0.0f,   // 500 Hz
    0.0f,   // 630 Hz
    0.0f,   // 800 Hz
    1.5f,   // 1000 Hz
    1.0f,  // 1250 Hz
    0.0f,  // 1600 Hz
    -1.0f,  // 2000 Hz
    0.0f,   // 2500 Hz
    1.0f,   // 3150 Hz
    1.0f,   // 4000 Hz - presence boost
    1.0f,   // 5000 Hz
    1.0f,   // 6300 Hz
    0.0f,   // 8000 Hz
    0.0f,  // 10000 Hz
    1.0f,  // 12500 Hz
    1.0f,   // 16000 Hz
    0.0f    // 20000 Hz
};

// Processing parameters
constexpr int kFftSize = 4096;
constexpr int kOverlapFactor = 4;
constexpr int kHopSize = kFftSize / kOverlapFactor;

// Compressor parameters
constexpr float kThresholdDb = -6.0f;
constexpr float kRatio = 4.0f;
constexpr float kAttackMs = 5.0f;
constexpr float kReleaseMs = 50.0f;

}  // namespace EQProcessor

#endif //EQ_CONFIG_H_