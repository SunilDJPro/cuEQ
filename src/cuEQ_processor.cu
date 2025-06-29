#include "cuEQ_processor.h"
#include "eq_config.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <cmath>
#include <algorithm>


namespace EQ_Processor {

__global__ void ApplyEqResponse(cufftComplex* fft_data, const float* eq_response, int fft_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < fft_size / 2 + 1) {
        const float gain = eq_response[idx];
        fft_data[idx].x *= gain;
        fft_data[idx].y *= gain;
    }
}

__global__ void WindowAndOverlapAdd(const float* input, float* output, int fft_size, int hop_size, int num_frames) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int frame_idx = idx / fft_size;
    const int sample_idx = idx % fft_size;

    if (frame_idx < num_frames && sample_idx < fft_size) {
        const float window = 0.5f * (1.0f - cosf(2.0f * M_PI * sample_idx / (fft_size - 1)));
        const float windowed_sample = input[idx] * window;
        const int output_idx = frame_idx * hop_size + sample_idx;
        atomicAdd(&output[output_idx], windowed_sample);
    }
 }

 cuEQProcessor::cuEQProcessor() : fft_plan_forward_(0), fft_plan_inverse_(0), gpu_memory_(nullptr), sample_rate_(0), initialized_(false) {}
 
 cuEQProcessor::~cuEQProcessor() {
    if (fft_plan_forward_) cufftDestroy(fft_plan_forward_);
    if (fft_pan_inverse_) cufftDestroy(fft_plan_inverse_);
    DeallocateGPUMem();
 }

 bool cuEQProcessor::Initialize(int sample_rate) {
    sample_rate_ = sample_rate;

    if (cufftPlan1d(&fft_plan_forward_, kFftSize, CUFFT_R2C, 1) != CUFFT_SUCCESS) {
        std::cerr << "Error: failed to create forward FFT plan!\n";
        return false
    }

    if (cufftPlan1d(&fft_plan_inverse_, kFftSize, CUFFT_C2R, 1) != CUFFT_SUCCESS) {
        std::cerr << "Error: failed to create inverse FFT plan\n";
        return false;
    }

    if (!CreateEQResponse(sample_rate)) {
        std::cerr << "Error: Failed to create EQ response!\n";
        return false;
    }

    initialized_ = true;
    return true;

 }



}