#ifndef WAV_IO_H_
#define WAV_IO_H_

#include <string>
#include <vector>
#include <cstdint>

namespace EQProcessor {

struct WavHeader {
    char chunk_id[4];
    uint32_t chunk_size;
    char format[4];
    char subchunk1_id[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char subchunk2_id[4];
    uint32_t subchunk2_size;
};

struct AudioData {
    WavHeader header;
    std::vector<float> samples;
    int num_channels;
    int sample_rate;
    int num_samples;
};

class WavIO {
 public:
  static bool ReadWavFile(const std::string& filename, AudioData* audio_data);
  static bool WriteWavFile(const std::string& filename, 
                          const AudioData& audio_data);
  
 private:
  static void ConvertToFloat(const std::vector<uint8_t>& raw_data,
                           std::vector<float>* float_data,
                           int bits_per_sample);
  static void ConvertFromFloat(const std::vector<float>& float_data,
                             std::vector<uint8_t>* raw_data,
                             int bits_per_sample);
};

} // namespace EQProcessor

#endif // WAV_IO_H_