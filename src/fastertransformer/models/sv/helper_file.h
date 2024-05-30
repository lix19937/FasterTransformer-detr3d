/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-04-16 13:30:56
 **************************************************************/

#ifndef HELPER_FILE_H
#define HELPER_FILE_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include <assert.h>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

#include "helper_cuda.h"
#include "helper_macros.h"

namespace cudaacc {

template<typename T>
inline void Read2Dptr(const std::string& file_name,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      T* d_data /* gpu side */,
                      bool is_need_tr = false)
{
    APP_PRINTF("template function should not enter into ...\n");
    exit(0);
}

template<>
inline void Read2Dptr(const std::string& file_name,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      half* d_data /* gpu side */,
                      bool is_need_tr)
{
    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    ///  with float
    std::vector<float> h_data(len);
    if (!cudaacc::readbyw<float>(file_name, h_data.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
        exit(1);
    }

    std::vector<float> h_data_tr(len);
    if (is_need_tr && c == 1) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                h_data_tr[j * h + i] = h_data[i * w + j];
            }
        }

        h_data.assign(h_data_tr.begin(), h_data_tr.end());
    }

    /// float2half
    std::vector<half> half_data(len);
    for (size_t i = 0; i < h_data.size(); ++i) {
        half_data[i] = __float2half(h_data[i]);
    }
    /// H2D
    cudaacc::CheckCudaErrors(cudaMemcpy(d_data, half_data.data(), len * sizeof(half), cudaMemcpyHostToDevice));
}

template<>
inline void Read2Dptr(const std::string& file_name,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      float* d_data /* gpu side */,
                      bool is_need_tr)
{
    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    std::vector<float> h_data(len);
    if (!cudaacc::readbyw<float>(file_name, h_data.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
        exit(1);
    }

    std::vector<float> h_data_tr(len);
    if (is_need_tr && c == 1) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                h_data_tr[j * h + i] = h_data[i * w + j];
            }
        }

        h_data.assign(h_data_tr.begin(), h_data_tr.end());
    }

    /// H2D
    cudaacc::CheckCudaErrors(cudaMemcpy(d_data, h_data.data(), len * sizeof(float), cudaMemcpyHostToDevice));
}

template<>
inline void Read2Dptr(const std::string& file_name,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      int8_t* d_data /* gpu side */,
                      bool is_need_tr)
{
    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    /// int
    std::vector<int8_t> h_data(len);
    if (!cudaacc::readbyw<int8_t>(file_name, h_data.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
        exit(1);
    }

    /// H2D
    cudaacc::CheckCudaErrors(cudaMemcpy(d_data, h_data.data(), len * sizeof(int8_t), cudaMemcpyHostToDevice));
}

template<>
inline void Read2Dptr(const std::string& file_name,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      uint8_t* d_data /* gpu side */,
                      bool is_need_tr)
{
    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    /// int
    std::vector<uint8_t> h_data(len);
    if (!cudaacc::readbyw<uint8_t>(file_name, h_data.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
        exit(1);
    }

    /// H2D
    cudaacc::CheckCudaErrors(cudaMemcpy(d_data, h_data.data(), len * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

template<>
inline void Read2Dptr(const std::string& file_name,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      int* d_data /* gpu side */,
                      bool is_need_tr)
{
    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    /// int
    std::vector<int> h_data(len);
    if (!cudaacc::readbyw<int>(file_name, h_data.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
        exit(1);
    }

    /// H2D
    cudaacc::CheckCudaErrors(cudaMemcpy(d_data, h_data.data(), len * sizeof(int), cudaMemcpyHostToDevice));
}

// write in CHW order  w is min-unit
// template <typename DT>
inline void
WriteFromDptr(const std::string& file_name, const std::vector<int>& dims, const float* d_data /* gpu side */)
{
    std::ofstream file(file_name, std::ios::out);
    const std::string split_flag(" ");

    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    std::vector<float> h_data(len);
    cudaacc::CheckCudaErrors(cudaMemcpy(h_data.data(), d_data, len * sizeof(float), cudaMemcpyDeviceToHost));

    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                // if (std::is_same<DT, uint8_t>::value) {
                //   int val = ((uint8_t*)h_data)[k * h * w + i * w + j];
                //   file << val << split_flag;
                // } else if (std::is_same<DT, float>::value) {
                float val = h_data[k * h * w + i * w + j];
                file << std::fixed << std::setprecision(6) << val << split_flag;
                // } else if (std::is_same<DT, int32_t>::value) {
                //   int32_t val = ((int32_t*)h_data)[k * h * w + i * w + j];
                //   file << val << split_flag;
                // }
            }
            file << std::endl;
        }
    }
    file.close();
}

inline void WriteFromDptr(const std::string& file_name, const std::vector<int>& dims, const half* d_data /* gpu side */)
{
    std::ofstream file(file_name, std::ios::out);
    const std::string split_flag(" ");

    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    std::vector<half> __h_data(len);
    std::vector<float> h_data(len);
    cudaacc::CheckCudaErrors(cudaMemcpy(__h_data.data(), d_data, len * sizeof(half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < len; ++i) {
        h_data[i] = __h_data[i];
    }

    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                // if (std::is_same<DT, uint8_t>::value) {
                //   int val = ((uint8_t*)h_data)[k * h * w + i * w + j];
                //   file << val << split_flag;
                // } else if (std::is_same<DT, float>::value) {
                float val = h_data[k * h * w + i * w + j];
                file << std::fixed << std::setprecision(6) << val << split_flag;
                // } else if (std::is_same<DT, int32_t>::value) {
                //   int32_t val = ((int32_t*)h_data)[k * h * w + i * w + j];
                //   file << val << split_flag;
                // }
            }
            file << std::endl;
        }
    }
    file.close();
}

inline void
WriteFromDptr(const std::string& file_name, const std::vector<int>& dims, const uint8_t* d_data /* gpu side */)
{
    std::ofstream file(file_name, std::ios::out);
    const std::string split_flag(" ");

    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    std::vector<uint8_t> h_data(len);
    cudaacc::CheckCudaErrors(cudaMemcpy(h_data.data(), d_data, len * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                // if (std::is_same<DT, uint8_t>::value) {
                //   int val = ((uint8_t*)h_data)[k * h * w + i * w + j];
                //   file << val << split_flag;
                // } else if (std::is_same<DT, float>::value) {
                int val = h_data[k * h * w + i * w + j];
                file << val << split_flag;
                // } else if (std::is_same<DT, int32_t>::value) {
                //   int32_t val = ((int32_t*)h_data)[k * h * w + i * w + j];
                //   file << val << split_flag;
                // }
            }
            file << std::endl;
        }
    }
    file.close();
}

inline void WriteFromDptr(const std::string& file_name, const std::vector<int>& dims, const int* d_data /* gpu side */)
{
    std::ofstream file(file_name, std::ios::out);
    const std::string split_flag(" ");

    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    std::vector<int> h_data(len);
    cudaacc::CheckCudaErrors(cudaMemcpy(h_data.data(), d_data, len * sizeof(int), cudaMemcpyDeviceToHost));

    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                // if (std::is_same<DT, uint8_t>::value) {
                //   int val = ((uint8_t*)h_data)[k * h * w + i * w + j];
                //   file << val << split_flag;
                // } else if (std::is_same<DT, float>::value) {
                auto val = h_data[k * h * w + i * w + j];
                file << val << split_flag;
                // } else if (std::is_same<DT, int32_t>::value) {
                //   int32_t val = ((int32_t*)h_data)[k * h * w + i * w + j];
                //   file << val << split_flag;
                // }
            }
            file << std::endl;
        }
    }
    file.close();
}

template<typename T>
inline void CompareGT(const T* d_data /* gpu side */,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      const std::string& file_name,
                      const std::string& flg,
                      float thres = 0.001f)
{
    APP_ASSERT(false);
}

template<>
inline void CompareGT(const half* d_data /* gpu side */,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      const std::string& file_name,
                      const std::string& flg,
                      float thres)
{
    cudaacc::CheckCudaErrors(cudaDeviceSynchronize());

    APP_PRINTF("CompareGT [%s], %.6f\n", flg.c_str(), thres);

    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }
    /// D2H
    std::vector<half> h_data(len);
    cudaacc::CheckCudaErrors(cudaMemcpy(h_data.data(), d_data, len * sizeof(half), cudaMemcpyDeviceToHost));

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    APP_PRINTF("CompareGT %d  %d  %d [%s]\n", c, h, w, flg.c_str());

    /// is float !!!
    std::vector<float> gt_data(len);
    if (!cudaacc::readbyw<float>(file_name, gt_data.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d [%s]\n", file_name.c_str(), c, h, w, flg.c_str());
        exit(1);
    }

    /// max
    float max = 0.f;
    size_t idx_keep = 0;
    for (size_t i = 0; i < gt_data.size(); ++i) {
        auto diff = __half2float(h_data[i]) - gt_data[i];
        auto t = fabs(diff);
        if (t > max) {
            max = t;
            idx_keep = i;
        }
    }
    APP_PRINTF("DIFF Max @%zd %.6f; %.6f|%.6f [%s]\n",
               idx_keep,
               max,
               __half2float(h_data[idx_keep]),
               gt_data[idx_keep],
               flg.c_str());

    /// diff
    for (size_t i = 0; i < gt_data.size(); ++i) {
        auto diff = __half2float(h_data[i]) - gt_data[i];
        if (fabs(diff) > thres) {
            APP_PRINTF("DIFF @%zd %.6f > %.6f; %.6f|%.6f [%s]\n",
                       i,
                       diff,
                       thres,
                       __half2float(h_data[i]),
                       gt_data[i],
                       flg.c_str());
            exit(1);
        }
    }

    APP_PRINTF("Passed [%s]\n", flg.c_str());
}

template<>
inline void CompareGT(const float* d_data /* gpu side */,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      const std::string& file_name,
                      const std::string& flg,
                      float thres)
{
    cudaacc::CheckCudaErrors(cudaDeviceSynchronize());

    APP_PRINTF("CompareGT [%s], %.6f\n", flg.c_str(), thres);

    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }

    /// D2H
    std::vector<float> h_data(len);
    cudaacc::CheckCudaErrors(cudaMemcpy(h_data.data(), d_data, len * sizeof(float), cudaMemcpyDeviceToHost));

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    /// is float !!!
    std::vector<float> gt_data(len);
    if (!cudaacc::readbyw<float>(file_name, gt_data.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
        exit(1);
    }

    /// max
    float max = 0.f;
    size_t idx_keep = 0;
    for (size_t i = 0; i < gt_data.size(); ++i) {
        auto diff = h_data[i] - gt_data[i];
        auto t = fabs(diff);
        if (t > max) {
            max = t;
            idx_keep = i;
        }
    }
    APP_PRINTF(
        "DIFF Max @%zd  %.6f; %.6f|%.6f [%s]\n", idx_keep, max, h_data[idx_keep], gt_data[idx_keep], flg.c_str());

    /// diff
    for (size_t i = 0; i < gt_data.size(); ++i) {
        auto diff = h_data[i] - gt_data[i];
        if (fabs(diff) >= thres) {
            APP_PRINTF("DIFF @%zd %.6f > %.6f; %.6f|%.6f [%s]\n", i, diff, thres, h_data[i], gt_data[i], flg.c_str());
            exit(1);
        }
    }

    APP_PRINTF("Passed [%s]\n", flg.c_str());
}

template<>
inline void CompareGT(const int* d_data /* gpu side */,
                      const std::vector<int>& dims /* dims.size() >=2*/,
                      const std::string& file_name,
                      const std::string& flg,
                      float thres)
{
    cudaacc::CheckCudaErrors(cudaDeviceSynchronize());

    APP_PRINTF("CompareGT [%s], %.6f", flg.c_str(), thres);

    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }
    /// D2H
    std::vector<int> h_data(len);
    cudaacc::CheckCudaErrors(cudaMemcpy(h_data.data(), d_data, len * sizeof(int), cudaMemcpyDeviceToHost));

    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    ///
    std::vector<int> gt_data(len);
    if (!cudaacc::readbyw<int>(file_name, gt_data.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d", file_name.c_str(), c, h, w);
        exit(1);
    }

    /// max
    float max = 0.f;
    size_t idx_keep = 0;
    for (size_t i = 0; i < gt_data.size(); ++i) {
        auto diff = h_data[i] - gt_data[i];
        auto t = fabs(diff);
        if (t > max) {
            max = t;
            idx_keep = i;
        }
    }
    APP_PRINTF("DIFF Max @%zd  %.6f; %d|%d [%s]", idx_keep, max, h_data[idx_keep], gt_data[idx_keep], flg.c_str());

    /// diff
    for (size_t i = 0; i < gt_data.size(); ++i) {
        auto diff = h_data[i] - gt_data[i];
        if (fabs(diff) >= thres) {
            APP_PRINTF("DIFF @%zd %d > %.6f; %d|%d [%s]", i, diff, thres, h_data[i], gt_data[i], flg.c_str());
            exit(1);
        }
    }

    APP_PRINTF("Passed [%s]", flg.c_str());
}

template<typename T>
inline void Diff(const std::vector<T>& a, const std::vector<T>& b, const std::string& flg, float precision)
{
}

template<>
inline void Diff(const std::vector<float>& a, const std::vector<float>& b, const std::string& flg, float precision)
{
    for (size_t i = 0; i < a.size(); ++i) {
        auto diff = a[i] - b[i];
        if (fabs(diff) >= precision) {
            APP_PRINTF("DIFF %zd  %.6f  %.6f  | %.6f  [%s]\n", i, a[i], b[i], diff, flg.c_str());
            exit(1);
        }
    }
}

template<>
inline void Diff(const std::vector<int>& a, const std::vector<int>& b, const std::string& flg, float precision)
{
    for (size_t i = 0; i < a.size(); ++i) {
        auto diff = a[i] - b[i];
        if (fabs(diff) >= precision) {
            APP_PRINTF("DIFF %zd  %d  %d  | %d  [%s]\n", i, a[i], b[i], diff, flg.c_str());
            exit(1);
        }
    }
}

template<typename T>
inline void Compare(const std::string& fa,
                    const std::string& fb,
                    const std::vector<int>& dims,
                    const std::string& flg,
                    float precision = 1.e-6)
{
    APP_PRINTF("Compare [%s]\n", flg.c_str());

    APP_ASSERT(!dims.empty());
    int len = 1;
    for (auto it : dims) {
        len *= it;
    }
    int c = 1, h = 1, w = *dims.rbegin();
    for (int i = 0; i < int(dims.size()) - 2; ++i) {
        c *= dims[i];
    }

    if (dims.size() >= 2) {
        h = dims[dims.size() - 2];
    }

    /// is float !!!
    std::vector<T> a(len);
    if (!cudaacc::readbyw<T>(fa, a.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d\n", fa.c_str(), c, h, w);
        exit(1);
    }

    std::vector<T> b(len);
    if (!cudaacc::readbyw<T>(fb, b.data(), c, h, w)) {
        APP_PRINTF("readrp error %s, chw:%d %d %d\n", fb.c_str(), c, h, w);
        exit(1);
    }

    std::vector<float> diffs;
    for (size_t i = 0; i < a.size(); ++i) {
        auto diff = fabs(a[i] - b[i]);
        if (diff > 1.e-6) {
            diffs.push_back(diff);
        }
    }
    auto max = std::max_element(std::begin(diffs), std::end(diffs));
    APP_PRINTF("max diff:%.6f\n", *max);

    /// diff
    Diff(a, b, flg, precision);
    APP_PRINTF("Passed [%s]\n", flg.c_str());
}

inline char* safe_strncpy(char* dest, const char* src, size_t n)
{
    assert(dest != nullptr && src != nullptr);
    char* ret = dest;
    while (*src != '\0' && --n > 0) {
        *dest++ = *src++;
    }
    *dest = '\0';
    return ret;
}

inline int mkdir_p(const char* dir)
{
    if (access(dir, F_OK) == 0) {
        return 0;
    }

    char tmp[256];
    safe_strncpy(tmp, dir, sizeof(tmp));
    char* p = tmp;
    char delim = '/';

    while (*p) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, S_IRWXU | S_IRWXG | S_IRWXO);
            *p = delim;
        }
        ++p;
    }

    if (0 != mkdir(tmp, S_IRWXU | S_IRWXG | S_IRWXO)) {
        std::cout << "create " << tmp << " has error !  maybe need sudo\n";
        return 1;
    }

    return 0;
}

}  // namespace cudaacc

template<typename T>
void FT_SAVE(const std::string& file_name, const std::vector<int>& dims, const T* d_data /* gpu side */)
{
    cudaacc::WriteFromDptr(file_name, dims, d_data /* gpu side */);
}

// template<typename T>
// void FT_LOAD(const std::string& file_name,
//              const std::vector<int>& dims,
//              const T* d_data /* gpu side */,
//              bool is_need_tr = false)
// {
//     cudaacc::Read2Dptr(file_name, dims,/* dims.size() >=2*/ d_data, /* gpu side */ is_need_tr);
// }

#endif  // HELPER_FILE_H
