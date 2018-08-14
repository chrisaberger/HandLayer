#ifndef BFLOAT16_H_
#define BFLOAT16_H_

#include <math.h>
#include <iostream>
#include <string>
#include "cblas.h"

struct bfloat16 {
  union {
    float _fdata;
    uint32_t _idata;
  };

  bfloat16(float _in) : _fdata(_in) {
    const uint32_t and_value = 0xFFFF0000;
    _idata &= and_value;
  }

  friend std::ostream& operator<<(std::ostream& os, const bfloat16& dt);

  bfloat16 operator+(bfloat16 const& obj) const {
    return bfloat16(obj._fdata + _fdata);
  }

  bfloat16 operator*(bfloat16 const& obj) const {
    return bfloat16(obj._fdata * _fdata);
  }

  bfloat16 operator/(bfloat16 const& obj) const {
    return bfloat16(_fdata / obj._fdata);
  }

  bool operator>(bfloat16 const& obj) const {
    return _fdata > obj._fdata;
  }

  void operator-=(bfloat16 const& obj) {
    _fdata -= obj._fdata;
    clamp();
  }

  void operator+=(bfloat16 const& obj) {
    _fdata += obj._fdata;
    clamp();
  }

  bfloat16 operator-(bfloat16 const& obj) const {
    return bfloat16(_fdata - obj._fdata);
  }

  bfloat16 operator-() const { return bfloat16(-_fdata); }

  void round() { std::cout << _idata << std::endl; }

  uint32_t get_mantissa() const {
    const uint32_t and_value = 0x007FFFFF;
    return (_idata & and_value);
  }

  uint32_t get_exponent() const {
    const uint32_t and_value = 0x7F800000;
    return ((_idata & and_value) >> 23) - 127;
  }

  uint32_t get_sign() const {
    const uint32_t and_value = 0x80000000;
    return (_idata & and_value) >> 31;
  }

  void print_fields() const {
    std::cout << "s: " << get_sign() << " e: " << get_exponent()
              << " m: " << std::hex << get_mantissa() << std::dec << std::endl;
  }

  void clamp() {
    const uint32_t and_value = 0xFFFF0000;
    _idata &= and_value;
  }
};

inline std::ostream& operator<<(std::ostream& os, const bfloat16& dt) {
  os << dt._fdata;
  return os;
}

inline bfloat16 log(const bfloat16& in) { return bfloat16(log(in._fdata)); }
inline bfloat16 exp(const bfloat16& in) { return bfloat16(exp(in._fdata)); }
inline bfloat16 tanh(const bfloat16& in) { return bfloat16(tanh(in._fdata)); }
inline bfloat16 sigmoid(const bfloat16& in) {
  return bfloat16(1.0) / (bfloat16(1.0) + exp(-in));
}
inline void gemm(bfloat16* X, bfloat16* W, const int M, const int N,
                 const int K, bfloat16* buffer, const bool bias,
                 const std::string type = "") {
  const float bias_f = bias ? 1.0 : 0.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
              (float*)X, K, (float*)W, N, bias_f, (float*)buffer, N);
  for (size_t i = 0; i < M * N; ++i) {
    buffer[i].clamp();
  }
}
#endif