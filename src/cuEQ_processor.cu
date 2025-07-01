#include "cuEQ_processor.h"
#include "eq_config.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace EQProcessor {

// Simple, high-quality CUDA kernels

// Apply EQ response in frequency domain
__global__ void ApplyEqResponse(cufftComplex* fft_data, 
                               const float* eq_response,
                               int num_bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < num_bins) {
    const float gain = eq_response[idx];
    fft_data[idx].x *= gain;
    fft_data[idx].y *= gain;
  }
}

cuEQProcessor::cuEQProcessor() 
    : fft_plan_forward_(0), fft_plan_inverse_(0), 
      gpu_memory_(nullptr), sample_rate_(0), initialized_(false) {}

cuEQProcessor::~cuEQProcessor() {
  if (fft_plan_forward_) cufftDestroy(fft_plan_forward_);
  if (fft_plan_inverse_) cufftDestroy(fft_plan_inverse_);
  DeallocateGPUMem();
}

bool cuEQProcessor::Initialize(int sample_rate) {
  sample_rate_ = sample_rate;
  
  std::cout << "Initializing GPU EQ processor for " << sample_rate << " Hz\n";
  
  // Create cuFFT plans
  if (cufftPlan1d(&fft_plan_forward_, kFftSize, CUFFT_R2C, 1) != CUFFT_SUCCESS) {
    std::cerr << "Error: Failed to create forward FFT plan\n";
    return false;
  }
  
  if (cufftPlan1d(&fft_plan_inverse_, kFftSize, CUFFT_C2R, 1) != CUFFT_SUCCESS) {
    std::cerr << "Error: Failed to create inverse FFT plan\n";
    return false;
  }
  
  std::cout << "cuFFT plans created successfully\n";
  
  // Allocate initial GPU memory for EQ response
  const size_t initial_size = kFftSize;
  if (!AllocateGPUMem(initial_size)) {
    std::cerr << "Error: Failed to allocate initial GPU memory\n";
    return false;
  }
  
  if (!CreateEQResponse(sample_rate)) {
    std::cerr << "Error: Failed to create EQ response\n";
    return false;
  }
  
  initialized_ = true;
  std::cout << "GPU EQ processor initialized successfully\n";
  return true;
}

bool cuEQProcessor::ProcessAudio(const std::vector<float>& input_samples,
                                 std::vector<float>* output_samples,
                                 int num_channels) {
  if (!initialized_) {
    std::cerr << "Error: Processor not initialized\n";
    return false;
  }
  
  const size_t num_samples = input_samples.size();
  const size_t samples_per_channel = num_samples / num_channels;
  
  std::cout << "High-quality processing: " << samples_per_channel << " samples per channel\n";
  
  // Calculate processing parameters
  const size_t num_frames = (samples_per_channel + kHopSize - 1) / kHopSize;
  const size_t output_length = samples_per_channel + kFftSize;  // Add padding for overlap
  
  std::cout << "Number of frames: " << num_frames << " (FFT size: " << kFftSize << ", hop: " << kHopSize << ")\n";
  
  // Ensure GPU memory is allocated
  const size_t required_size = kFftSize * 2;
  if (!AllocateGPUMem(required_size)) {
    return false;
  }
  
  output_samples->resize(num_samples, 0.0f);
  
  // Process each channel with careful frame-by-frame processing
  for (int channel = 0; channel < num_channels; ++channel) {
    std::cout << "Processing channel " << channel + 1 << "/" << num_channels << "\n";
    
    std::vector<float> channel_output(output_length, 0.0f);
    
    // Process each frame with overlap-add
    for (size_t frame = 0; frame < num_frames; ++frame) {
      if (frame % 2000 == 0) {
        std::cout << "  Frame " << frame << "/" << num_frames << "\n";
      }
      
      const size_t frame_start = frame * kHopSize;
      
      // Prepare input frame with proper windowing
      std::vector<float> frame_input(kFftSize, 0.0f);
      
      // Extract frame data
      for (size_t i = 0; i < kFftSize; ++i) {
        const size_t sample_idx = frame_start + i;
        if (sample_idx < samples_per_channel) {
          const size_t interleaved_idx = sample_idx * num_channels + channel;
          if (interleaved_idx < input_samples.size()) {
            frame_input[i] = input_samples[interleaved_idx];
          }
        }
      }
      
      // Apply Hann window for smooth reconstruction
      for (size_t i = 0; i < kFftSize; ++i) {
        const float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (kFftSize - 1)));
        frame_input[i] *= window;
      }
      
      // Copy to GPU
      cudaMemcpy(gpu_memory_->d_input, frame_input.data(),
                kFftSize * sizeof(float), cudaMemcpyHostToDevice);
      
      // Forward FFT
      cufftExecR2C(fft_plan_forward_, gpu_memory_->d_input, gpu_memory_->d_fft_buffer);
      
      // Apply EQ
      const int block_size = 256;
      const int grid_size = (kFftSize / 2 + 1 + block_size - 1) / block_size;
      ApplyEqResponse<<<grid_size, block_size>>>(
          gpu_memory_->d_fft_buffer, gpu_memory_->d_eq_response, kFftSize / 2 + 1);
      
      cudaDeviceSynchronize();
      
      // Inverse FFT
      cufftExecC2R(fft_plan_inverse_, gpu_memory_->d_fft_buffer, gpu_memory_->d_output);
      
      // Copy result back
      std::vector<float> frame_output(kFftSize);
      cudaMemcpy(frame_output.data(), gpu_memory_->d_output,
                kFftSize * sizeof(float), cudaMemcpyDeviceToHost);
      
      // Proper normalization and windowing for overlap-add
      const float fft_normalization = 1.0f / kFftSize;
      const float overlap_compensation = 1.0f;  // No additional scaling needed with proper overlap
      
      for (size_t i = 0; i < kFftSize; ++i) {
        // Apply same window again for smooth overlap-add reconstruction
        const float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (kFftSize - 1)));
        frame_output[i] = frame_output[i] * fft_normalization * window * overlap_compensation;
      }
      
      // Overlap-add to output buffer
      for (size_t i = 0; i < kFftSize; ++i) {
        const size_t output_idx = frame_start + i;
        if (output_idx < channel_output.size()) {
          channel_output[output_idx] += frame_output[i];
        }
      }
    }
    
    // Copy processed channel back to interleaved output
    for (size_t i = 0; i < samples_per_channel; ++i) {
      if (i < channel_output.size()) {
        (*output_samples)[i * num_channels + channel] = channel_output[i];
      }
    }
  }
  
  std::cout << "High-quality processing completed\n";
  return true;
}

bool cuEQProcessor::AllocateGPUMem(size_t required_size) {
  if (gpu_memory_ && gpu_memory_->allocated_size >= required_size) {
    return true;  // Already allocated sufficient memory
  }
  
  DeallocateGPUMem();
  gpu_memory_ = std::make_unique<GPUMemory>();
  gpu_memory_->allocated_size = 0;
  
  const size_t buffer_size = std::max(required_size, static_cast<size_t>(kFftSize));
  const size_t fft_buffer_size = (kFftSize / 2 + 1) * sizeof(cufftComplex);
  const size_t eq_response_size = (kFftSize / 2 + 1) * sizeof(float);
  
  std::cout << "Allocating GPU memory:\n";
  std::cout << "  Buffer size: " << buffer_size << " floats\n";
  std::cout << "  Input/Output buffers: " << (buffer_size * sizeof(float)) / (1024 * 1024) << " MB each\n";
  std::cout << "  FFT buffer: " << fft_buffer_size / (1024 * 1024) << " MB\n";
  std::cout << "  EQ response: " << eq_response_size / (1024 * 1024) << " MB\n";
  
  // Allocate GPU memory with error checking
  cudaError_t cuda_err;
  
  cuda_err = cudaMalloc(&gpu_memory_->d_input, buffer_size * sizeof(float));
  if (cuda_err != cudaSuccess) {
    std::cerr << "Error allocating input buffer: " << cudaGetErrorString(cuda_err) << "\n";
    DeallocateGPUMem();
    return false;
  }
  
  cuda_err = cudaMalloc(&gpu_memory_->d_output, buffer_size * sizeof(float));
  if (cuda_err != cudaSuccess) {
    std::cerr << "Error allocating output buffer: " << cudaGetErrorString(cuda_err) << "\n";
    DeallocateGPUMem();
    return false;
  }
  
  cuda_err = cudaMalloc(&gpu_memory_->d_fft_buffer, fft_buffer_size);
  if (cuda_err != cudaSuccess) {
    std::cerr << "Error allocating FFT buffer: " << cudaGetErrorString(cuda_err) << "\n";
    DeallocateGPUMem();
    return false;
  }
  
  cuda_err = cudaMalloc(&gpu_memory_->d_eq_response, eq_response_size);
  if (cuda_err != cudaSuccess) {
    std::cerr << "Error allocating EQ response buffer: " << cudaGetErrorString(cuda_err) << "\n";
    DeallocateGPUMem();
    return false;
  }
  
  // Initialize buffers to zero
  cudaMemset(gpu_memory_->d_input, 0, buffer_size * sizeof(float));
  cudaMemset(gpu_memory_->d_output, 0, buffer_size * sizeof(float));
  cudaMemset(gpu_memory_->d_fft_buffer, 0, fft_buffer_size);
  
  gpu_memory_->allocated_size = buffer_size;
  
  std::cout << "GPU memory allocated successfully\n";
  
  // CRITICAL: Re-upload EQ response after allocating new memory
  if (sample_rate_ > 0) {
    std::cout << "Re-uploading EQ response after memory reallocation\n";
    if (!CreateEQResponse(sample_rate_)) {
      std::cerr << "Error: Failed to re-create EQ response after memory allocation\n";
      return false;
    }
  }
  
  return true;
}

void cuEQProcessor::DeallocateGPUMem() {
  if (gpu_memory_) {
    if (gpu_memory_->d_input) {
      cudaFree(gpu_memory_->d_input);
      gpu_memory_->d_input = nullptr;
    }
    if (gpu_memory_->d_output) {
      cudaFree(gpu_memory_->d_output);
      gpu_memory_->d_output = nullptr;
    }
    if (gpu_memory_->d_fft_buffer) {
      cudaFree(gpu_memory_->d_fft_buffer);
      gpu_memory_->d_fft_buffer = nullptr;
    }
    if (gpu_memory_->d_eq_response) {
      cudaFree(gpu_memory_->d_eq_response);
      gpu_memory_->d_eq_response = nullptr;
    }
    gpu_memory_->allocated_size = 0;
    gpu_memory_.reset();
  }
}

bool cuEQProcessor::CreateEQResponse(int sample_rate) {
  const int num_bins = kFftSize / 2 + 1;
  std::vector<float> eq_response(num_bins, 1.0f);
  
  std::cout << "Creating clean EQ response with " << num_bins << " frequency bins\n";
  
  // Create frequency response using clean bell filters
  for (int bin = 0; bin < num_bins; ++bin) {
    const float frequency = static_cast<float>(bin) * sample_rate / (2.0f * kFftSize);
    float total_gain_db = 0.0f;
    
    // Apply each EQ band with proper Q factor
    for (int band = 0; band < kNumEqBands; ++band) {
      const float center_freq = kIsoCenterFreqs[band];
      const float gain_db = kEqGainsDb[band];
      
      if (std::abs(gain_db) > 0.01f && frequency > 10.0f) {
        // Standard parametric EQ bell filter
        const float Q = 2.0f;  // Professional Q factor
        const float omega = 2.0f * M_PI * frequency / sample_rate;
        const float omega_c = 2.0f * M_PI * center_freq / sample_rate;
        
        // Bell filter frequency response
        const float delta_omega = omega - omega_c;
        const float bandwidth = omega_c / Q;
        
        if (std::abs(delta_omega) < bandwidth) {
          const float response = 1.0f / (1.0f + std::pow(delta_omega / (bandwidth / 2.0f), 2));
          total_gain_db += gain_db * response;
        }
      }
    }
    
    // Convert to linear gain with reasonable limits
    eq_response[bin] = std::pow(10.0f, std::clamp(total_gain_db, -20.0f, 20.0f) / 20.0f);
  }
  
  // Copy EQ response to GPU
  cudaError_t cuda_err = cudaMemcpy(gpu_memory_->d_eq_response, eq_response.data(),
                                   eq_response.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (cuda_err != cudaSuccess) {
    std::cerr << "Error copying EQ response to GPU: " << cudaGetErrorString(cuda_err) << "\n";
    return false;
  }
  
  std::cout << "Clean EQ response uploaded to GPU\n";
  return true;
}

size_t cuEQProcessor::CalculateOptimalBatchSize() {
  // Query GPU properties to determine optimal batch size
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  
  const size_t available_memory = prop.totalGlobalMem * 0.8;  // Use 80% of GPU memory
  const size_t memory_per_sample = sizeof(float) * 4;  // Input, output, FFT buffer space
  const size_t max_samples = available_memory / memory_per_sample;
  
  // Align to FFT frame boundaries
  const size_t max_frames = max_samples / kFftSize;
  return max_frames * kFftSize;
}

}  // namespace EQProcessor