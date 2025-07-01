#include "cuEQ_processor.h"
#include "eq_config.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace EQProcessor {

// CUDA kernel for applying EQ response in frequency domain
__global__ void ApplyEqResponse(cufftComplex* fft_data, 
                               const float* eq_response,
                               int fft_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_bins = fft_size / 2 + 1;
  
  if (idx < num_bins) {
    const float gain = eq_response[idx];
    
    // Apply gain to complex FFT data
    fft_data[idx].x *= gain;
    fft_data[idx].y *= gain;
  }
}

// CUDA kernel for windowing and overlap-add
__global__ void WindowAndOverlapAdd(const float* input, float* output,
                                   int fft_size, int hop_size, int num_frames) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int frame_idx = idx / fft_size;
  const int sample_idx = idx % fft_size;
  
  if (frame_idx < num_frames && sample_idx < fft_size) {
    // Hann window
    const float window = 0.5f * (1.0f - cosf(2.0f * M_PI * sample_idx / (fft_size - 1)));
    const float windowed_sample = input[idx] * window;
    
    // Overlap-add
    const int output_idx = frame_idx * hop_size + sample_idx;
    atomicAdd(&output[output_idx], windowed_sample);
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
  
  std::cout << "Processing " << samples_per_channel << " samples per channel\n";
  
  // Calculate processing parameters
  const size_t num_frames = (samples_per_channel + kHopSize - 1) / kHopSize;
  const size_t padded_length = samples_per_channel + kFftSize;  // Add padding for overlap
  
  std::cout << "Number of frames: " << num_frames << "\n";
  
  // Allocate GPU memory for single FFT frame processing
  const size_t required_size = kFftSize * 2;  // Double buffer for safety
  if (!AllocateGPUMem(required_size)) {
    return false;
  }
  
  output_samples->resize(num_samples, 0.0f);
  
  // Process each channel
  for (int channel = 0; channel < num_channels; ++channel) {
    std::cout << "Processing channel " << channel + 1 << "/" << num_channels << "\n";
    
    std::vector<float> channel_input(padded_length, 0.0f);
    std::vector<float> channel_output(padded_length, 0.0f);
    
    // Extract channel data with zero padding
    for (size_t i = 0; i < samples_per_channel; ++i) {
      channel_input[i] = input_samples[i * num_channels + channel];
    }
    
    // Debug: Check if we have non-zero input
    float max_input = 0.0f;
    for (size_t i = 0; i < std::min(samples_per_channel, size_t(1000)); ++i) {
      max_input = std::max(max_input, std::abs(channel_input[i]));
    }
    std::cout << "  Channel " << channel << " max input level: " << max_input << "\n";
    
    // Process frames with overlap-add
    for (size_t frame = 0; frame < num_frames; ++frame) {
      if (frame % 1000 == 0) {
        std::cout << "  Processing frame " << frame << "/" << num_frames << "\n";
      }
      
      const size_t frame_start = frame * kHopSize;
      
      // Prepare frame input with zero padding
      std::vector<float> frame_input(kFftSize, 0.0f);
      const size_t copy_size = std::min(static_cast<size_t>(kFftSize), 
                                       samples_per_channel - frame_start);
      
      if (frame_start < samples_per_channel) {
        std::copy(channel_input.begin() + frame_start,
                 channel_input.begin() + frame_start + copy_size,
                 frame_input.begin());
      }
      
      // Apply window function (Hann window) - ONLY ONCE HERE
      for (size_t i = 0; i < kFftSize; ++i) {
        const float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (kFftSize - 1)));
        frame_input[i] *= window;
      }
      
      // Copy frame to GPU
      cudaError_t cuda_err = cudaMemcpy(gpu_memory_->d_input, frame_input.data(),
                                       kFftSize * sizeof(float), cudaMemcpyHostToDevice);
      if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(cuda_err) << "\n";
        return false;
      }
      
      // Debug: Check input data on GPU
      if (frame == 0) {
        std::vector<float> gpu_input_check(kFftSize);
        cudaMemcpy(gpu_input_check.data(), gpu_memory_->d_input, 
                  kFftSize * sizeof(float), cudaMemcpyDeviceToHost);
        float max_gpu_input = 0.0f;
        for (size_t i = 0; i < kFftSize; ++i) {
          max_gpu_input = std::max(max_gpu_input, std::abs(gpu_input_check[i]));
        }
        std::cout << "    GPU input max level: " << max_gpu_input << "\n";
      }
      
      // Forward FFT
      cufftResult fft_result = cufftExecR2C(fft_plan_forward_, gpu_memory_->d_input, 
                                           gpu_memory_->d_fft_buffer);
      if (fft_result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT forward error: " << fft_result << "\n";
        return false;
      }
      
      // Debug: Check FFT output
      if (frame == 0) {
        std::vector<cufftComplex> fft_check(kFftSize / 2 + 1);
        cudaMemcpy(fft_check.data(), gpu_memory_->d_fft_buffer, 
                  (kFftSize / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        float max_fft_mag = 0.0f;
        for (size_t i = 0; i < fft_check.size(); ++i) {
          float magnitude = std::sqrt(fft_check[i].x * fft_check[i].x + fft_check[i].y * fft_check[i].y);
          max_fft_mag = std::max(max_fft_mag, magnitude);
        }
        std::cout << "    FFT output max magnitude: " << max_fft_mag << "\n";
      }
      
      // DEBUG: Re-enable EQ application with proper scaling
      // Debug: Check EQ response on GPU before applying
      if (frame == 0) {
        std::vector<float> eq_check(kFftSize / 2 + 1);
        cudaMemcpy(eq_check.data(), gpu_memory_->d_eq_response, 
                  (kFftSize / 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "    EQ response on GPU: ";
        for (int i = 0; i < 10; ++i) {
          std::cout << eq_check[i] << " ";
        }
        std::cout << "\n";
        
        // Check for invalid values
        int invalid_count = 0;
        for (size_t i = 0; i < eq_check.size(); ++i) {
          if (eq_check[i] != eq_check[i] || eq_check[i] == 0.0f) {  // NaN or zero check
            invalid_count++;
          }
        }
        std::cout << "    Invalid EQ values: " << invalid_count << " out of " << eq_check.size() << "\n";
      }
      
      // Apply EQ
      const int block_size = 256;
      const int grid_size = (kFftSize / 2 + 1 + block_size - 1) / block_size;
      ApplyEqResponse<<<grid_size, block_size>>>(
          gpu_memory_->d_fft_buffer, gpu_memory_->d_eq_response, kFftSize);
      
      // Synchronize to ensure kernel completion
      cuda_err = cudaDeviceSynchronize();
      if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA synchronization error: " << cudaGetErrorString(cuda_err) << "\n";
        return false;
      }
      
      // Debug: Check EQ output
      if (frame == 0) {
        std::vector<cufftComplex> eq_check(kFftSize / 2 + 1);
        cudaMemcpy(eq_check.data(), gpu_memory_->d_fft_buffer, 
                  (kFftSize / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        float max_eq_mag = 0.0f;
        for (size_t i = 0; i < eq_check.size(); ++i) {
          float magnitude = std::sqrt(eq_check[i].x * eq_check[i].x + eq_check[i].y * eq_check[i].y);
          max_eq_mag = std::max(max_eq_mag, magnitude);
        }
        std::cout << "    After EQ max magnitude: " << max_eq_mag << "\n";
      }
      
      // Inverse FFT
      fft_result = cufftExecC2R(fft_plan_inverse_, gpu_memory_->d_fft_buffer, 
                               gpu_memory_->d_output);
      if (fft_result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT inverse error: " << fft_result << "\n";
        return false;
      }
      
      // Debug: Check IFFT output before normalization
      if (frame == 0) {
        std::vector<float> ifft_check(kFftSize);
        cudaMemcpy(ifft_check.data(), gpu_memory_->d_output, 
                  kFftSize * sizeof(float), cudaMemcpyDeviceToHost);
        float max_ifft = 0.0f;
        for (size_t i = 0; i < kFftSize; ++i) {
          max_ifft = std::max(max_ifft, std::abs(ifft_check[i]));
        }
        std::cout << "    IFFT raw output max: " << max_ifft << "\n";
      }
      
      // Copy frame back from GPU
      std::vector<float> frame_output(kFftSize);
      cuda_err = cudaMemcpy(frame_output.data(), gpu_memory_->d_output,
                           kFftSize * sizeof(float), cudaMemcpyDeviceToHost);
      if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA memcpy back error: " << cudaGetErrorString(cuda_err) << "\n";
        return false;
      }
      
      // Normalize with proper scaling (cuFFT doesn't normalize)
      // The key issue: we need to account for the window function energy loss
      const float fft_normalization = 1.0f / kFftSize;
      const float window_energy_compensation = 2.0f;  // Hann window loses ~50% energy
      const float overlap_compensation = 2.0f;  // Overlap-add with 75% overlap needs compensation
      const float total_scale = fft_normalization * window_energy_compensation * overlap_compensation;
      
      for (size_t i = 0; i < kFftSize; ++i) {
        frame_output[i] *= total_scale;
      }
      
      // Debug: Check final frame output level
      if (frame == 0) {
        float max_final = 0.0f;
        for (size_t i = 0; i < kFftSize; ++i) {
          max_final = std::max(max_final, std::abs(frame_output[i]));
        }
        std::cout << "    Final frame output max: " << max_final << "\n";
      }
      
      // Overlap-add
      for (size_t i = 0; i < kFftSize; ++i) {
        const size_t output_idx = frame_start + i;
        if (output_idx < channel_output.size()) {
          channel_output[output_idx] += frame_output[i];
        }
      }
    }
    
    // Debug: Check output levels
    float max_output = 0.0f;
    for (size_t i = 0; i < std::min(samples_per_channel, size_t(1000)); ++i) {
      max_output = std::max(max_output, std::abs(channel_output[i]));
    }
    std::cout << "  Channel " << channel << " max output level: " << max_output << "\n";
    
    // Copy channel output back to interleaved format
    for (size_t i = 0; i < samples_per_channel; ++i) {
      if (i < channel_output.size()) {
        (*output_samples)[i * num_channels + channel] = channel_output[i];
      }
    }
  }
  
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
  
  std::cout << "Creating EQ response with " << num_bins << " frequency bins\n";
  std::cout << "Sample rate: " << sample_rate << " Hz, FFT size: " << kFftSize << "\n";
  
  // Create frequency response based on 31-band ISO EQ
  for (int bin = 0; bin < num_bins; ++bin) {
    const float frequency = static_cast<float>(bin) * sample_rate / (2.0f * kFftSize);
    float total_gain_db = 0.0f;
    
    // Apply all relevant EQ bands for this frequency
    for (int band = 0; band < kNumEqBands; ++band) {
      const float center_freq = kIsoCenterFreqs[band];
      const float gain_db = kEqGainsDb[band];
      
      if (std::abs(gain_db) > 0.01f && frequency > 1.0f) {  // Skip DC and very low frequencies
        // Use octave-based bandwidth (1/3 octave = ~0.231 in log2 space)
        const float log_freq = std::log2(frequency);
        const float log_center = std::log2(center_freq);
        const float octave_distance = std::abs(log_freq - log_center);
        
        // 1/3 octave bandwidth
        if (octave_distance <= 0.5f) {  // Within 1/2 octave for smooth response
          const float weight = std::cos(octave_distance * M_PI);  // Smooth rolloff
          total_gain_db += gain_db * weight * weight;
        }
      }
    }
    
    eq_response[bin] = std::pow(10.0f, total_gain_db / 20.0f);
  }
  
  // Debug: Print some EQ response values
  std::cout << "EQ response samples: ";
  const int debug_samples = std::min(10, num_bins);
  for (int i = 0; i < debug_samples; ++i) {
    const int bin_idx = (i * num_bins) / debug_samples;
    const float freq = static_cast<float>(bin_idx) * sample_rate / (2.0f * kFftSize);
    std::cout << freq << "Hz=" << eq_response[bin_idx] << " ";
  }
  std::cout << "\n";
  
  // Verify we have some non-unity gains
  int non_unity_count = 0;
  for (int i = 0; i < num_bins; ++i) {
    if (std::abs(eq_response[i] - 1.0f) > 0.01f) {
      non_unity_count++;
    }
  }
  std::cout << "Non-unity gains: " << non_unity_count << " out of " << num_bins << " bins\n";
  
  // Copy EQ response to GPU
  cudaError_t cuda_err = cudaMemcpy(gpu_memory_->d_eq_response, eq_response.data(),
                                   eq_response.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (cuda_err != cudaSuccess) {
    std::cerr << "Error copying EQ response to GPU: " << cudaGetErrorString(cuda_err) << "\n";
    return false;
  }
  
  std::cout << "EQ response created and uploaded to GPU\n";
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
