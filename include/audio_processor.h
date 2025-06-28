#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <string>
#include <memory>
#include "wav_io.h"
#include "audio_processor.h"
#include "cuEQ_processor.h"
#include "compressor.h"

namespace EQProcessor {

class AudioProcessor {

public:
    AudioProcessor();
    ~AudioProcessor();

    bool ProcessFile(const std::string& input_file, const std::string& output_file);

private:
    std::unique_ptr<cuEQProcessor> cuEQ_processor_;
    std::unique_ptr<Compressor> compressor_;
};

} // namespace EQProcessor


#endif // AUDIO_PROCESSOR_H