#ifndef CU_EQ_PROCESSOR_H_
#define CU_EQ_PROCESSOR_H_

#include <cufft.h>
#include <vector>
#include <memory>

namespace EQProcessor {

class cuEQProcessor {

public:
    cuEQProcessor();
    ~cuEQProcessor();

    bool Initialize(int sample_rate);
    bool ProcessAudio(const std::vector<float>& input_samples, std::vector<float>* output_samples, int num_channels);

private:
    struct GPUMemory {
        float* d_input;
        float* d_ouput;
        cufftComplex* d_fft_buffer;
        float* d_eq_response;
        size_t allocated_size;
    };

    bool AllocateGPUMem(size_t required_size);
    void DeallocateGPUMem();
    bool CreateEQResponse(int sample_rate);
    size_t CalculateOptimalBatchSize();

    cufftHandle fft_plan_forward_;
    cufftHandle fft_plan_inverse_;
    std::unique_ptr<GPUMemory> gpu_memory_;
    int sample_rate_;
    bool initialized_;
};

} // namespace EQProcessor

#endif // CU_EQ_PROCESSOR_H_