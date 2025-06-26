#ifndef EQ_CONFIG_H_
#define EQ_CONFIG_H_

#include <array>

namespace eq_x {

    constexpr int kNumEqBands = 32;
    constexpr float kMinGainDb = -20.0f;
    constexpr float kMaxGainDb = 20.0f;

    constexpr std::array<float, kNumEqBands> KIsoCenterFreqs = {
        20.0f, 25.0f, 31.5f, 40.0f, 50.0f, 63.0f, 80.0f, 100.0f, 125.0f, 160.0f, 200.0f, 250.0f, 315.0f, 400.0f, 500.0f, 630.0f, 800.0f, 1000.0f, 1250.0f, 1600.0f, 2000.0f, 2500.0f, 3150.0f, 4000.0f, 5000.0f, 6300.0f, 8000.0f, 10000.0f, 12500.0f, 16000.0f, 20000.0f
    };

    constexpr std::array<float, kNumEqBands> kEqGainsDb = {
        0.0f, //20Hz - 1
        0.0f, //25Hz - 2
        2.0f, //31.5Hz - 3
        3.0f, //40Hz - 4
        1.0f, //50Hz - 5
        0.0f, //63Hz - 6
        -1.0f, //80Hz - 7
        -2.0f, //100Hz - 8
        0.0f, //125Hz - 9
        1.0f, //160Hz - 10
        2.0f, //200Hz - 11
        1.5f, //250Hz - 12
        0.0f, //315Hz - 13
        -1.0f, //400Hz - 14
        0.0f, //500Hz - 15
        1.0f, //630Hz - 16
        2.0f, //800Hz - 17
        0.0f, //1000Hz - 18
        -1.0f, //1250Hz - 19
        -2.0f, //1600Hz - 20
        -1.0f, //2000Hz - 21
        0.0f, //2500Hz - 22
        1.0f, //3150Hz - 23
        2.0f, //4000Hz - 24
        1.0f, //5000Hz - 25
        0.0f, //6300Hz - 26
        -1.0f, //8000Hz - 27
        -2.0f, //10000Hz - 28
        -1.0f, //12500Hz - 29
        0.0f, //16000Hz - 30
        0.0f, //20000Hz - 31

    };

    //Processing params
    constexpr int kFftSize = 4096;
    constexpr int kOverlapFactor = 4;
    constexpr int kHopSize = kFftSize / kOverlapFactor;

    //Composer params
    constexpr float kThresholdDb = -6.0f;
    constexpr float kRatio = 4.0f;
    constexpr float kAttackMs = 5.0f;
    constexpr float kReleaseMs = 50.0f;

} // namespace eq_x

#endif //EQ_CONFIG_H_