//
//  CPUStft.cpp
//  MNN
//
//  Created by MNN on 2024/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 Ref from onnxruntime
 */

#ifndef M_PI
#define M_PI 3.141592654
#endif
#include <algorithm>
#include <cmath>
#include <complex>
#include "backend/cpu/CPUStft.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "core/Macro.h"
#include "compute/CommonOptFunction.h"

namespace MNN {

#define ___RETURN_IF_ERROR(x) {auto code = (x); if (NO_ERROR != code) {return code;}}
#define ___RETURN_IF(x, y) {if (x) {return NOT_SUPPORT;}}

static bool is_real_valued_signal(const Tensor* shape) {
    return shape->dimensions() == 2 || shape->length(shape->dimensions() -1) == 1;
}

static bool is_complex_valued_signal(const Tensor* shape) {
    return shape->dimensions() > 2 && shape->length(shape->dimensions() -1) == 2;
}

static bool is_power_of_2(size_t size) {
    size_t n_bits = 0;
    while (size != 0) {
        n_bits += size & 1;
        size = size >> 1;
    }
    return n_bits == 1;
}

static const unsigned char BitReverseTable256[] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48,
    0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4,
    0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C,
    0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2,
    0x32, 0xB2, 0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A,
    0xFA, 0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E,
    0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE, 0x01, 0x81, 0x41, 0xC1, 0x21,
    0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9,
    0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9, 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55,
    0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD,
    0x7D, 0xFD, 0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 0x0B,
    0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB, 0x07, 0x87, 0x47, 0xC7,
    0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F,
    0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

template <typename T>
static inline T bit_reverse(T num, unsigned significant_bits) {
  if (significant_bits > 32) {
      MNN_ERROR("Unsupported bit size.");
  }
  uint32_t num_32 = static_cast<uint32_t>(num);
  uint32_t rev = (BitReverseTable256[num_32 & 0xff] << 24) | (BitReverseTable256[(num_32 >> 8) & 0xff] << 16) |
                 (BitReverseTable256[(num_32 >> 16) & 0xff] << 8) | (BitReverseTable256[(num_32 >> 24) & 0xff]);
  return static_cast<T>(((uint64_t)rev) >> (32 - significant_bits));
}

template <typename T>
static T compute_angular_velocity(size_t number_of_samples, bool inverse) {
  // Calculate fundamental angular velocity
  static const T pi = static_cast<T>(M_PI);
  static const T tau = 2 * pi;
  T inverse_switch = inverse ? 1.f : -1.f;
  T angular_velocity = inverse_switch * tau / number_of_samples;
  return angular_velocity;
}

template <typename T>
static std::complex<T> compute_exponential(size_t index, const T angular_velocity) {
  const T angle = static_cast<T>(index) * angular_velocity;
  return std::complex<T>(cos(angle), sin(angle));
}

template <typename T, typename U>
static ErrorCode fft_radix2(Backend* backend, const Tensor* X, Tensor* Y, size_t X_offset, size_t X_stride,
                         size_t Y_offset, size_t Y_stride, int64_t axis, size_t dft_length, const Tensor* window,
                         bool is_onesided, bool inverse, std::vector<std::complex<T>>& V,
                         std::vector<std::complex<T>>& temp_output) {
    // Get shape and significant bits
    const auto X_shape = X->shape();
    size_t number_of_samples = static_cast<size_t>(X_shape[axis]);
    unsigned significant_bits = static_cast<unsigned>(log2(dft_length));

      // Get data
    auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->host<void>())) + X_offset;
    // Get window
    float* window_data = nullptr;
    if (window) {
        window_data = const_cast<float*>(reinterpret_cast<const float*>(window->host<void>()));
    }

    size_t Y_data_stride = 1;
    std::complex<T>* Y_data;
    if (is_onesided) {
        if (temp_output.size() != dft_length) {
            temp_output.resize(dft_length);
        }
        Y_data = temp_output.data();
    } else {
        Y_data = reinterpret_cast<std::complex<T>*>(Y->host<void>()) + Y_offset;
        Y_data_stride = Y_stride;
    }

    auto angular_velocity = compute_angular_velocity<T>(dft_length, inverse);

    // Create vandermonde matrix V ordered with the bit-reversed permutation
    if (V.size() != dft_length) {
        V.resize(dft_length);
        for (size_t i = 0; i < dft_length; i++) {
            size_t bit_reversed_index = bit_reverse(i, significant_bits);
            V[bit_reversed_index] = compute_exponential(i, angular_velocity);
        }
    }

    for (size_t i = 0; i < dft_length; i++) {
        size_t bit_reversed_index = bit_reverse(i, significant_bits);
        auto x = (bit_reversed_index < number_of_samples) ? *(X_data + bit_reversed_index * X_stride) : 0;
        auto window_element = window_data ? *(window_data + bit_reversed_index) : 1;
        *(Y_data + i * Y_data_stride) = std::complex<T>(1, 0) * x * window_element;
    }

    // Run fft_radix2
    unsigned current_significant_bits = 0;
    for (size_t i = 2; i <= dft_length; i <<= 1) {
        size_t midpoint = i >> 1;
        current_significant_bits++;

        for (size_t k = 0; k < midpoint; k++) {
            auto first_idx = bit_reverse(k, current_significant_bits);
            auto second_idx = bit_reverse(midpoint + k, current_significant_bits);
            for (size_t j = 0; j < dft_length; j += i) {
                auto even_index = k + j;
                auto odd_index = k + j + midpoint;
                std::complex<T>* even = (Y_data + even_index * Y_data_stride);
                std::complex<T>* odd = (Y_data + odd_index * Y_data_stride);
                std::complex<T> first = *even + (V[first_idx] * *odd);
                std::complex<T> second = *even + (V[second_idx] * *odd);
                *even = first;
                *odd = second;
            }
        }
    }

    // Scale the output if inverse
    if (inverse) {
        for (size_t i = 0; i < dft_length; i++) {
            std::complex<T>& val = *(Y_data + i * Y_data_stride);
            val /= static_cast<T>(dft_length);
        }
    }

    if (is_onesided) {
        const size_t output_size = (dft_length >> 1) + 1;
        auto destination = reinterpret_cast<std::complex<T>*>(Y->host<void>()) + Y_offset;
        for (size_t i = 0; i < output_size; i++) {
            *(destination + Y_stride * i) = *(Y_data + i * Y_data_stride);
        }
    }

    return NO_ERROR;
}

template <typename T>
T next_power_of_2(T in) {
  in--;
  T out = 1;
  while (out <= in) {
    out <<= 1;
  }
  return out;
}

template <typename T, typename U>
static ErrorCode dft_bluestein_z_chirp(Backend* bn, const Tensor* X, Tensor* Y, std::shared_ptr<Tensor>& b_fft_p, std::shared_ptr<Tensor>& chirp_p, size_t X_offset, size_t X_stride, size_t Y_offset, size_t Y_stride,
    int64_t axis, size_t dft_length, const Tensor* window, bool inverse, std::vector<std::complex<T>>& V,
    std::vector<std::complex<T>>& temp_output) {
    static const T pi = static_cast<T>(M_PI);

    size_t N = static_cast<size_t>(dft_length);
    size_t M = next_power_of_2(2 * N - 1);
    auto dft_input_shape = std::vector<int>({1, (int)M, 2});
    T scale = inverse ? 1.f / N : 1.f;
    T direction = inverse ? 1.f : -1.f;

    bool should_recreate_b_fft = b_fft_p->elementSize() != M * 2;
    bool should_recreate_chirp = chirp_p->elementSize() != M * 2;
    bool should_recreate = should_recreate_b_fft || should_recreate_chirp;
    if (should_recreate) {
        std::shared_ptr<Tensor> b_p(Tensor::create(dft_input_shape, X->getType()));
        auto& b = *b_p;
        b_fft_p.reset(Tensor::create(dft_input_shape, Y->getType()));
        auto& b_fft = *b_fft_p;
        chirp_p.reset(Tensor::create(dft_input_shape, X->getType()));
        auto& chirp = *chirp_p;

        std::complex<T>* b_data = reinterpret_cast<std::complex<T>*>(b.host<void>());
        std::complex<T>* b_fft_data = reinterpret_cast<std::complex<T>*>(b_fft.host<void>());
        std::complex<T>* chirp_data = reinterpret_cast<std::complex<T>*>(chirp.host<void>());
        memset(reinterpret_cast<void*>(b_data), 0, b.usize());
        memset(reinterpret_cast<void*>(b_fft_data), 0, b_fft.usize());
        memset(reinterpret_cast<void*>(chirp_data), 0, chirp.usize());

        for (size_t n = 0; n < N; n++) {
            std::complex<T>& chirp_n = *(chirp_data + n);
            // chirp
            auto exponent = direction * pi * n * n / N;
            chirp_n = std::complex<T>(cos(exponent), sin(exponent));

            // b
            std::complex<T>& b_n = *(b_data + n);
            b_n = std::conj(chirp_n);
        }

        for (size_t n = M - N + 1; n < M; n++) {
            std::complex<T>& b_n = *(b_data + n);
            std::complex<T>& b_m_minus_n = *(b_data + M - n);
            b_n = b_m_minus_n;
        }

        // Forward FFT radix2 for the "b" signal
        // This will be cached and reused!
      auto code = ((fft_radix2<T, std::complex<T>>(bn, &b, &b_fft, 0, 1, 0, 1, 1, M, nullptr,
                                                            false, false, V, temp_output)));
      if (NO_ERROR != code) {
          FUNC_PRINT(1);
          return code;
      }
  }

  // Get data
    auto* X_data = const_cast<U*>(reinterpret_cast<const U*>(X->host<void>())) + X_offset;
    auto* Y_data = reinterpret_cast<std::complex<T>*>(Y->host<void>()) + Y_offset;
    float* window_data = nullptr;
    if (window) {
        window_data = const_cast<float*>(reinterpret_cast<const float*>(window->host<void>()));
    }
    std::shared_ptr<Tensor> a_p(Tensor::create(dft_input_shape, X->getType()));
    auto& a = *a_p;
    std::shared_ptr<Tensor> a_fft_p(Tensor::create(dft_input_shape, Y->getType()));
    auto& a_fft = *a_fft_p;
    std::complex<T>* a_data = reinterpret_cast<std::complex<T>*>(a.host<void>());
    std::complex<T>* a_fft_data = reinterpret_cast<std::complex<T>*>(a_fft.host<void>());
    std::complex<T>* b_fft_data = reinterpret_cast<std::complex<T>*>(b_fft_p->host<void>());
    std::complex<T>* chirp_data = reinterpret_cast<std::complex<T>*>(chirp_p->host<void>());
    memset(reinterpret_cast<void*>(a_data), 0, a.usize());

    const auto& X_shape = X->shape();
    size_t number_of_samples = static_cast<size_t>(X_shape[axis]);

    // Prepare "a" signal
    for (size_t n = 0; n < number_of_samples; n++) {
        std::complex<T>& a_n = *(a_data + n);
        std::complex<T>& chirp_n = *(chirp_data + n);
        auto window_n = window_data ? *(window_data + n) : 1;
        a_n = *(X_data + n * X_stride);  // input
        a_n *= window_n;
        a_n *= chirp_n;
    }

  // Forward FFT radix2 for the "a" signal
    {
        auto code = ((fft_radix2<T, std::complex<T>>(bn, &a, &a_fft, 0, 1, 0, 1, 1, M, nullptr,
                                                            false, false, V, temp_output)));
        if (NO_ERROR != code) {
            return code;
        }
    }

    for (size_t i = 0; i < M; i++) {
        std::complex<T>& a_i = *(a_fft_data + i);
        std::complex<T>& b_i = *(b_fft_data + i);
        a_i *= b_i;
    }

  // Inverse FFT radix2 for the "a" signal
    {
        auto code = ((fft_radix2<T, std::complex<T>>(bn, &a_fft, &a, 0, 1, 0, 1, 1, M, nullptr,
                                                  false, true, V, temp_output)));
        if (NO_ERROR != code) {
            return code;
        }
    }
    const auto& Y_shape = Y->shape();
    size_t dft_output_size = static_cast<size_t>(Y_shape[(axis)]);

    for (size_t i = 0; i < dft_output_size; i++) {
        std::complex<T>& chirp_i = *(chirp_data + i);
        std::complex<T>& out = *(Y_data + i * Y_stride);
        std::complex<T>& c_i = *(a_data + i);
        if (i > 0) {
      // The inverse fft is computed using the same cached vandermonde matrix (V) created by the
      // forward fft. This reversal causes the output to be reversed as well.
      // Therefore we undo the reversal when writing the output back out.
            c_i = *(a_data + M - i);
        }
        out = c_i * chirp_i * scale;
    }
    return NO_ERROR;
}

template <typename T, typename U>
static ErrorCode discrete_fourier_transform(Backend* ctx, const Tensor* X, Tensor* Y, std::shared_ptr<Tensor>& b_fft, std::shared_ptr<Tensor>& chirp,
                                         int64_t axis, int64_t dft_length, const Tensor* window, bool is_onesided, bool inverse,
                                         std::vector<std::complex<T>>& V,
                                         std::vector<std::complex<T>>& temp_output) {
    // Get shape
    const auto& X_shape = X->shape();
    const auto& Y_shape = Y->shape();

    auto batch_and_signal_rank = X->dimensions();
    auto total_dfts = static_cast<size_t>(X->elementSize() / X->length(axis));

    auto is_input_real = X->dimensions() == 2 || X->length(X->dimensions() - 1) == 1;
    auto complex_input_factor = is_input_real ? 1 : 2;
    if (X->dimensions() > 2) {
        total_dfts /= (X->length(X->dimensions() - 1));
        batch_and_signal_rank -= 1;
    }

    // Calculate x/y offsets/strides
    for (size_t i = 0; i < total_dfts; i++) {
        size_t X_offset = 0;
        size_t X_stride = X->stride(axis) / complex_input_factor;
        size_t cumulative_packed_stride = total_dfts;
        size_t temp = i;
        for (size_t r = 0; r < batch_and_signal_rank; r++) {
            if (r == static_cast<size_t>(axis)) {
                continue;
            }
            cumulative_packed_stride /= (X_shape[r]);
            auto index = temp / cumulative_packed_stride;
            temp -= (index * cumulative_packed_stride);
            X_offset += index * X->stride(r) / complex_input_factor;
        }
        
        size_t Y_offset = 0;
        size_t Y_stride = Y->stride(axis) / 2;
        cumulative_packed_stride = total_dfts;
        temp = i;
        for (size_t r = 0; r < batch_and_signal_rank; r++) {
            if (r == static_cast<size_t>(axis)) {
                continue;
            }
            cumulative_packed_stride /= (X_shape[r]);
            auto index = temp / cumulative_packed_stride;
            temp -= (index * cumulative_packed_stride);
            Y_offset += index * (size_t)(Y->stride(r) / 2);
        }
        
        if (is_power_of_2((dft_length))) {
            ___RETURN_IF_ERROR((fft_radix2<T, U>(ctx, X, Y, X_offset, X_stride, Y_offset, Y_stride, axis, (dft_length), window,
                                                  is_onesided, inverse, V, temp_output)));
        } else {
            ___RETURN_IF_ERROR(
                                (dft_bluestein_z_chirp<T, U>(ctx, X, Y, b_fft, chirp, X_offset, X_stride, Y_offset, Y_stride, axis, (dft_length), window, inverse, V, temp_output)));
        }
    }
    return NO_ERROR;
}

static ErrorCode discrete_fourier_transform(Backend* ctx, int64_t axis, bool is_onesided, bool inverse, Tensor* X, Tensor* dft_length, Tensor* Y) {
    // Get input shape
    const auto is_real_valued = is_real_valued_signal(X);
    const auto is_complex_valued = is_complex_valued_signal(X);
    if (axis < 0) {
        axis = axis + X->dimensions();
    }

    int64_t number_of_samples = static_cast<int64_t>(X->length(axis));
    if (dft_length) {
        const auto& dft_length_shape = dft_length->shape();
        number_of_samples = dft_length->host<int>()[0];
    }

    // Get the DFT output size. Onesided will return only the unique values!
    // note: x >> 1 === std::floor(x / 2.f)
    auto dft_output_size = is_onesided ? ((number_of_samples >> 1) + 1) : number_of_samples;

    std::shared_ptr<Tensor> b_fft(new Tensor), chirp(new Tensor);
    std::vector<std::complex<float>> V;
    std::vector<std::complex<float>> temp_output;
    if (is_real_valued) {
      ___RETURN_IF_ERROR((discrete_fourier_transform<float, float>(ctx, X, Y, b_fft, chirp, axis, number_of_samples, nullptr,
                                                                    is_onesided, inverse, V, temp_output)));
    } else if (is_complex_valued) {
      ___RETURN_IF_ERROR((discrete_fourier_transform<float, std::complex<float>>(
          ctx, X, Y, b_fft, chirp, axis, number_of_samples, nullptr, is_onesided, inverse, V, temp_output)));
    }

    return NO_ERROR;
}

template <typename T, typename U>
static ErrorCode short_time_fourier_transform(Backend* ctx, Tensor* signal, Tensor* Y, int frame_step, Tensor* window, bool is_onesided, bool /*inverse*/) {
    // Attr("onesided"): default = 1
    // Input(0, "signal") type = T1
    // Input(1, "frame_length") type = T2
    // Input(2, "window") type = T1, optional
    // Input(3, "frame_step") type = T2
    // Output(0, "output") type = T1

    // Get input signal shape
    const auto& signal_shape = signal->shape();
    const auto batch_size = signal_shape[0];
    const auto signal_size = signal_shape[1];
    const auto signal_components = signal_shape.size() == 2 ? 1 : signal_shape[2];
    // Get the frame length
    int frame_length = window->length(0);
    // Get window length
    // Calculate the window size with preference to the window input.
      const auto window_size = frame_length;
      MNN_ASSERT(window_size <= signal_size);
    // Calculate the number of dfts to run
    const auto n_dfts =
        static_cast<int64_t>(std::floor((signal_size - window_size) / static_cast<float>(frame_step))) + 1;

    // Calculate the output spectra length (onesided will return only the unique values)
    // note: x >> 1 === std::floor(x / 2.f)
    const auto dft_output_size = is_onesided ? (window_size >> 1) + 1 : window_size;

    auto Y_data = reinterpret_cast<float*>(Y->host<void>());

    // Get/create the signal mutable data
    auto* signal_data = const_cast<float*>(reinterpret_cast<const float*>(signal->host<void>()));

    // Define tensor shapes for each dft run
    const int output_components = 2;
    auto dft_input_shape = std::vector<int>{1, window_size, signal_components};
    auto dft_output_shape = std::vector<int>{1, dft_output_size, output_components};

    std::shared_ptr<Tensor> b_fft(new Tensor), chirp(new Tensor);
    std::vector<std::complex<T>> V;
    std::vector<std::complex<T>> temp_output;
    // Tensors do not own the backing memory, so no worries on destruction
    std::shared_ptr<Tensor> input(Tensor::createDevice(dft_input_shape, signal->getType()));
    std::shared_ptr<Tensor> output(Tensor::createDevice(dft_output_shape, Y->getType()));

    // Run each dft of each batch as if it was a real-valued batch size 1 dft operation
    for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int64_t i = 0; i < n_dfts; i++) {
            auto input_frame_begin =
            signal_data + (batch_idx * signal_size * signal_components) + (i * frame_step * signal_components);
            
            auto output_frame_begin = Y_data + (batch_idx * n_dfts * dft_output_size * output_components) + (i * dft_output_size * output_components);
            input->buffer().host = (uint8_t*)input_frame_begin;
            output->buffer().host = (uint8_t*)output_frame_begin;
            // Run individual dft
            ___RETURN_IF_ERROR((discrete_fourier_transform<T, U>(ctx, input.get(), output.get(), b_fft, chirp, 1, window_size, window, is_onesided, false, V, temp_output)));
        }
    }

    return NO_ERROR;
}


CPUStft::CPUStft(Backend* backend, bool abs)
    : Execution(backend), mAbs(abs) {
    // nothing to do
}

ErrorCode CPUStft::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    return NO_ERROR;
}

ErrorCode CPUStft::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto signal = inputs[0];
    const auto is_real_valued = is_real_valued_signal(signal);
    const auto is_complex_valued = is_complex_valued_signal(signal);
    int frameStep = inputs[1]->host<int>()[0];
    if (is_real_valued) {
        ___RETURN_IF_ERROR((short_time_fourier_transform<float, float>(backend(), inputs[0], outputs[0], frameStep, inputs[2], mAbs, false)));
    } else if (is_complex_valued) {
        ___RETURN_IF_ERROR((short_time_fourier_transform<float, std::complex<float>>(backend(), inputs[0], outputs[0], frameStep, inputs[2], mAbs, false)));
    } else {
        MNN_ASSERT(false);
    }
    return NO_ERROR;
}

class CPUStftCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto stft = op->main_as_StftParam();
        return new CPUStft(backend, stft->abs());
    }
};

REGISTER_CPU_OP_CREATOR(CPUStftCreator, OpType_Stft);
} // namespace MNN
