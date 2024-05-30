/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-01-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-04-16 13:30:56
 **************************************************************/

#ifndef HELPER_STRING_H  
#define HELPER_STRING_H

#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#ifndef STRCASECMP
#define STRCASECMP _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy_s(sFilePath, nLength, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) fopen_s(&fHandle, filename, mode)
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif
#ifndef SPRINTF
#define SPRINTF sprintf_s
#endif
#else  // Linux Includes

#include <string.h>
#include <strings.h>

#ifndef STRCASECMP
#define STRCASECMP strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#ifndef SPRINTF
#define SPRINTF sprintf
#endif
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif


static const std::string red{"\033[31m"};
static const std::string yellow{"\033[33m"};
static const std::string white{"\033[37m"};

#define LOG_INFO(format, ...)   do {fprintf(stderr, "[I %s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); } while (0)
#define LOG_ERROR(format, ...)  do {fprintf(stderr, "%s[E %s:%d] " format "%s\n", red.c_str(), __FILE__, __LINE__, ##__VA_ARGS__, white.c_str()); } while (0)

namespace cuda_ops {

// CUDA Utility Helper Functions
inline int stringRemoveDelimiter(char delimiter, const char *string) {
  int string_start = 0;
  while (string[string_start] == delimiter) {
    string_start++;
  }

  if (string_start >= static_cast<int>(strlen(string) - 1)) {
    return 0;
  }

  return string_start;
}

inline int getFileExtension(char *filename, char **extension) {
  auto string_length = static_cast<int>(strlen(filename));

  while (filename[string_length--] != '.') {
    if (string_length == 0) break;
  }

  if (string_length > 0) {
    string_length += 2;
  }

  if (string_length == 0)
    *extension = NULL;
  else
    *extension = &filename[string_length];

  return string_length;
}

inline bool checkCmdLineFlag(const int argc, const char **argv,
                             const char *string_ref) {
  bool is_found{false};
  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];

      const char *equal_pos = strchr(string_argv, '=');
      int argv_length = static_cast<int>(
          equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

      int length = static_cast<int>(strlen(string_ref));

      if (length == argv_length &&
          !STRNCASECMP(string_argv, string_ref, length)) {
        is_found = true;
        continue;
      }
    }
  }

  return is_found;
}

// This function wraps the CUDA Driver API into a template function
template <class T>
inline bool getCmdLineArgumentValue(const int argc, const char **argv,
                                    const char *string_ref, T *value) {
  bool is_found = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        if (length + 1 <= static_cast<int>(strlen(string_argv))) {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          *value = (T)atoi(&string_argv[length + auto_inc]);
        }

        is_found = true;
        i = argc;
      }
    }
  }

  return is_found;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv,
                                 const char *string_ref) {
  bool is_found = false;
  int value = -1;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        if (length + 1 <= static_cast<int>(strlen(string_argv))) {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = atoi(&string_argv[length + auto_inc]);
        } else {
          value = 0;
        }

        is_found = true;
        continue;
      }
    }
  }

  if (is_found) {
    return value;
  } else {
    return 0;
  }
}

inline float getCmdLineArgumentFloat(const int argc, const char **argv,
                                     const char *string_ref) {
  bool is_found{false};
  float value{-1};

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        if (length + 1 <= static_cast<int>(strlen(string_argv))) {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = static_cast<float>(atof(&string_argv[length + auto_inc]));
        } else {
          value = 0.f;
        }

        is_found = true;
        continue;
      }
    }
  }

  if (is_found) {
    return value;
  } else {
    return 0;
  }
}

inline bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref,
                                     char **string_retval) {
  bool is_found = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      char *string_argv = const_cast<char *>(&argv[i][string_start]);
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        *string_retval = &string_argv[length + 1];
        is_found = true;
        continue;
      }
    }
  }

  if (!is_found) {
    *string_retval = NULL;
  }

  return is_found;
}

// write hwc, mainly for cv img
inline void writerp_cv(const std::string &file_name, void *src, int h, int w,
                       int c, int type) {
  std::ofstream file(file_name, std::ios::out);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < c; ++k) {
        if (type == 1) {
          int val = ((uint8_t *)src)[i * w + j * c + k];
          file << val << " ";
        } else if (type == 2) {
          float val = ((float *)src)[i * w + j * c + k];
          file << val << " ";
        }
      }
      file << std::endl;
    }
  }
  file.close();
}

inline void writerp(const std::string &file_name, void *src, int w, int h,
                    int type) {
  std::ofstream file(file_name, std::ios::out);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      if (type == 1) {
        int val = ((uint8_t *)src)[i * w + j];
        file << val << " ";
      } else if (type == 2) {
        float val = ((float *)src)[i * w + j];
        file << val << " ";
      }
    }
    file << std::endl;
  }
  file.close();
}

// write in CHW order  w is min-unit
inline void writerp(const std::string &file_name, void *src, int c, int h,
                    int w, int type = 2) {
  std::ofstream file(file_name, std::ios::out);
  const std::string split_flag(" ");

  for (int k = 0; k < c; ++k) {
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        if (type == 1) {
          int val = ((uint8_t *)src)[k * h * w + i * w + j];
          file << val << split_flag;
        } else if (type == 2) {
          float val = ((float *)src)[k * h * w + i * w + j];
          file << std::fixed << std::setprecision(6) << val << split_flag;
        } else if (type == 3) {
          int32_t val = ((int32_t *)src)[k * h * w + i * w + j];
          file << val << split_flag;
        }
      }
      file << std::endl;
    }
  }
  file.close();
}

template <typename T>
inline T exchange(const std::string &src) {
  return atoll(src.c_str());
}

template <>
inline double exchange(const std::string &src) {
  return atof(src.c_str());
}

template <>
inline float exchange(const std::string &src) {
  return atof(src.c_str());
}

template <>
inline int exchange(const std::string &src) {
  return atoi(src.c_str());
}

template <>
inline int8_t exchange(const std::string &src) {
  return atoi(src.c_str());
}

template <>
inline std::string exchange(const std::string &src) {
  return src;
}

template <typename T>
inline void split(std::vector<T> &ret, const std::string &str, char delim = ' ',
                  bool ignore_empty = true) {
  if (str.empty()) {
    return;
  }
  ret.clear();

  size_t n = str.size();
  size_t s = 0;

  while (s <= n) {
    size_t i = str.find_first_of(delim, s);
    size_t len = 0;

    if (i == std::string::npos) {
      len = n - s;
    } else {
      len = i - s;
    }

    if (!ignore_empty || 0 != len) {
      auto tmp = str.substr(s, len);
      ret.push_back(std::move(exchange<T>(tmp)));
    }

    s += len + 1;
  }
}

template <typename T>
inline void split(T *ret, const std::string &str, char delim = ' ',
                  bool ignore_empty = true) {
  if (str.empty()) {
    return;
  }

  size_t n = str.size();
  size_t s = 0, k = 0;

  while (s <= n) {
    size_t i = str.find_first_of(delim, s);
    size_t len = 0;

    if (i == std::string::npos) {
      len = n - s;
    } else {
      len = i - s;
    }

    if (!ignore_empty || 0 != len) {
      auto tmp = str.substr(s, len);
      ret[k++] = std::move(exchange<T>(tmp));
    }

    s += len + 1;
  }
}

// read data in CHW order
template <typename T>
inline bool readrp(const std::string &file_name, T *data, int c, int h, int w,
                   char delim = ' ') {
  std::ifstream file(file_name, std::ios::in);
  if (!file.is_open()) {
    printf("file open error\n");
    return false;
  }

  int line_idx = 0;
  std::string buf;
  while (std::getline(file, buf)) {
    split<T>(data + h * w * line_idx, buf, delim);
    buf.clear();
    ++line_idx;
  }

  if (line_idx != c) {
    printf("line_idx != c error, %d vs %d\n", line_idx, c);
    return false;
  }
  file.close();

  return true;
}

/// min unit is row(dim=w)
template <typename T>
inline bool readbyw(const std::string &file_name, T *data, int c, int h, int w,
                    char delim = ' ') {
  std::ifstream file(file_name, std::ios::in);
  if (!file.is_open()) {
    printf("file open error\n");
    return false;
  }

  const int total_row = c * h;
  int line_idx = 0;
  std::string buf;
  while (std::getline(file, buf)) {
    split<T>(data + w * line_idx, buf, delim);
    buf.clear();
    ++line_idx;
    if (line_idx > total_row) {
      printf(
          "line_idx %d >= expected %d, maybe file was write as an append "
          "`append` method by user, check file total lines !\n",
          line_idx, total_row);
    }
  }

  if (line_idx != total_row) {
    printf("line_idx != c*h error, %d vs %d\n", line_idx, total_row);
    return false;
  }
  file.close();

  return true;
}

template <class T>
void Read2Buff(const std::string &file_name,
               const std::vector<int> &dims /* dims.size() >=2*/,
               std::vector<T> &h_data /* cpu side */) {
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
  h_data.resize(len);
  if (!cuda_ops::readbyw(file_name, h_data.data(), c, h, w)) {
    LOG_ERROR("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
    exit(1);
  }
}

template <class T>
void Read2Buff(const std::string &file_name,
               const std::vector<int> &dims /* dims.size() >=2*/,
               T *h_data /* cpu side */) {
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
  if (!cuda_ops::readbyw(file_name, h_data, c, h, w)) {
    LOG_ERROR("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
    exit(1);
  }
}

template <class T>
void Read2Buff_GPU(const std::string &file_name,
               const std::vector<int> &dims /* dims.size() >=2*/,
               T *d_data /* gpu side */){}

template <>
void Read2Buff_GPU(const std::string &file_name,
               const std::vector<int> &dims /* dims.size() >=2*/,
               float *d_data /* gpu side */) {
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
  if (!cuda_ops::readbyw(file_name, h_data.data(), c, h, w)) {
    LOG_ERROR("readrp error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
    exit(1);
  }

  auto ret = cudaMemcpy(d_data, h_data.data(), sizeof(float) * len, cudaMemcpyHostToDevice);
  if (ret != 0){
    LOG_ERROR("cudaMemcpy error %s, chw:%d %d %d\n", file_name.c_str(), c, h, w);
    exit(1);
  }
}

// yuv420i
inline int readbin_yuv420i(const std::string &bin_file, uint8_t *ptr,
                           const int &h, const int &w) {
  FILE *fp = fopen(bin_file.c_str(), "rb");
  if (fp == NULL) {
    printf("fopen error\n");
    return 1;
  }

  fseek(fp, 0L, SEEK_END);
  auto end = ftell(fp);
  fseek(fp, 0L, SEEK_SET);
  auto start = ftell(fp);

  printf("size:%ld  %ld\n", end, start);
  const int &length = h * w * 1.5;
  if (length != end) {
    printf("length error\n");
    return 1;
  }

  for (int i = 0; i < end; ++i) {
    auto ret = fread(ptr + i, 1, sizeof(uint8_t), fp);
    if (ret != 1) {
      printf("fread error\n");
    }
  }
  fclose(fp);
  fp = NULL;

  return 0;
}

// rgba
inline int readbin_rgba(const std::string &bin_file, uint8_t *ptr,
                        const int &length) {
  FILE *fp = fopen(bin_file.c_str(), "rb");
  if (fp == NULL) {
    printf("fopen error\n");
    return 1;
  }

  fseek(fp, 0L, SEEK_END);
  auto end = ftell(fp);
  fseek(fp, 0L, SEEK_SET);
  auto start = ftell(fp);

  printf("size:%ld  %ld\n", end, start);
  if (length != end) {
    printf("length error\n");
    return 1;
  }

  for (int i = 0; i < end; ++i) {
    auto ret = fread(ptr + i, 1, sizeof(uint8_t), fp);
    if (ret != 1) {
      printf("fread error\n");
    }

    if ((i + 1) % 4 == 0 && i > 0) {  // rgba
      if (255 != ptr[i]) {
        // printf("%d \n" , int(ptr[i]));
        ptr[i] = 255;
      }
    }
  }
  fclose(fp);
  fp = NULL;

  return 0;
}

}  // namespace cuda_ops

#endif  // HELPER_STRING_H
