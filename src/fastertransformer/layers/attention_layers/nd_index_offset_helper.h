/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-14 18:56:40
 **************************************************************/

#pragma once

#include <cassert>

namespace ft {

#if defined(__CUDACC__)
#define OF_DEVICE_FUNC __device__ __host__ __forceinline__
#else
#define OF_DEVICE_FUNC inline
#endif

template <typename T, int N>
class NdIndexOffsetHelper {
 public:
  NdIndexOffsetHelper() {}
  template <class... Ts>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(T d0, Ts... dims) {
    constexpr int n = 1 + sizeof...(dims);
    static_assert(n <= N, "");
    T dims_arr[n] = {d0, static_cast<T>(dims)...};
    InitStrides(dims_arr, n);
  }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims) {
    InitStrides(dims, N);
  }

  template <typename U>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const U* dims) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      dims_arr[i] = dims[i];
    }
    InitStrides(dims_arr, N);
  }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims, int n) {
    InitStrides(dims, n);
  }

  template <typename U>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const U* dims, int n) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        dims_arr[i] = dims[i];
      }
    }
    InitStrides(dims_arr, n);
  }

  ~NdIndexOffsetHelper() = default;

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index) const {
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) {
      offset += index[i] * stride_[i];
    }
    offset += index[N - 1];
    return offset;
  }

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index, int n) const {
    assert(n <= N);
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        offset += index[i] * stride_[i];
      }
    }
    return offset;
  }

  template <class... Ts>
  OF_DEVICE_FUNC T NdIndexToOffset(T d0, Ts... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T index[n] = {d0, others...};
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      offset += index[i] * stride_[i];
    }
    if (n == N) {
      offset += index[n - 1];
    } else {
      offset += index[n - 1] * stride_[n - 1];
    }
    return offset;
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index) const {
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) {
      const T idx = remaining / stride_[i];
      index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    index[N - 1] = remaining;
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index, int n) const {
    assert(n <= N);
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        const T idx = remaining / stride_[i];
        index[i] = idx;
        remaining = remaining - idx * stride_[i];
      }
    }
  }

  template <class... Ts>
  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T& d0, Ts&... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T* index[n] = {&d0, &others...};
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      const T idx = remaining / stride_[i];
      *index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    if (n == N) {
      *index[n - 1] = remaining;
    } else {
      *index[n - 1] = remaining / stride_[n - 1];
    }
  }

  OF_DEVICE_FUNC constexpr int Size() const { return N; }

 private:
  OF_DEVICE_FUNC void InitStrides(const T* dims, const int n) {
    for (int i = n - 1; i < N; ++i) {
      stride_[i] = 1;
    }
    for (int i = n - 2; i >= 0; --i) {
      stride_[i] = dims[i + 1] * stride_[i + 1];
    }
  }

  T stride_[N];
};

template <size_t num_dims, typename IndexType>
struct PermuteKernelParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  int permutation[num_dims]{};
  IndexType count{};
  const void* src{};
  void* dst{};
};

__forceinline__ __device__ void InitStrides(const int* dims, const int num_dims, int* stride_) {
  stride_[num_dims - 1] = 1;
  for (int i = num_dims - 2; i >= 0; --i) {
    stride_[i] = dims[i + 1] * stride_[i + 1];
  }
}

__forceinline__ __device__ void OffsetToNdIndex(int offset, int* index, const int num_dims, int* stride_) {
  int remaining = offset;
  for (int i = 0; i < num_dims - 1; ++i) {
    const int idx = remaining / stride_[i];
    index[i] = idx;
    remaining = remaining - idx * stride_[i];
  }
  index[num_dims - 1] = remaining;
}

__forceinline__ __device__ int NdIndexToOffset(const int* index, const int num_dims, int* stride_) {
  int offset = 0;
  for (int i = 0; i < num_dims - 1; ++i) {
    offset += index[i] * stride_[i];
  }
  offset += index[num_dims - 1];
  return offset;
}

///  from c/32 hw32 --> c/32 32hw
__forceinline__ __device__ void Permute(int n, int c, int h, int w, const int i, size_t* src_offset) {
  const int num_dims = 5;
  int src_stride[num_dims], dst_stride[num_dims],src_index[num_dims], dst_index[num_dims];
  int permutation[]{0, 1, 4, 2, 3};
  int src_dims[]{n, c / 32, h, w, 32};
  int dst_dims[]{n, c / 32, 32, h, w};

  InitStrides(src_dims, num_dims, src_stride);
  InitStrides(dst_dims, num_dims, dst_stride);

  OffsetToNdIndex(i, dst_index, num_dims, dst_stride);
  for (size_t dim = 0; dim < num_dims; ++dim) {
    src_index[permutation[dim]] = dst_index[dim];
  }
  *src_offset = NdIndexToOffset(src_index, num_dims, src_stride);

  // dst[i] = src[src_offset];
}

}  // namespace ft
