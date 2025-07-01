#ifndef COMPRESSOR_H_
#define COMPRESSOR_H_

#include <vector>

namespace EQProcessor {

class Compressor {

public:
    Compressor(float threshold_db, float ratio, float attack_ms, float release_ms, int sample_rate);
    void ProcessSamples(std::vector<float>* samples, int num_channels);

private:
    float dbToLinear(float db) const;
    float LinearTodb(float linear) const;

    float threshold_linear_;
    float ratio_;
    float attack_coeff_;
    float release_coeff_;
    float envelope_;
    int sample_rate_;
};

} // namespace EQProcessor

#endif // COMPRESSOR_H_