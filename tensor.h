#ifndef TENSOR_H_
#define TENSOR_H_

#include <vector>

template<class T>
struct Tensor {
  std::vector<size_t> shape;
  size_t numel;
  std::unique_ptr<T> data;

  bool requires_grad;
  std::unique_ptr<T> grad_data;

  Tensor(){};
  Tensor(const std::vector<size_t> shape);

  static Tensor<T> from_vec(const std::vector<T>& vec,
                            const std::vector<size_t>& shape_in);
  static Tensor<T> from_npy(const std::string npy_filename);
  static Tensor<T> copy(const Tensor<T> &src);

  T& operator[](const size_t i);
  const T operator[](const size_t i) const;
  T& operator()(const size_t i, const size_t j);
  const T operator()(const size_t i, const size_t j) const;
  void print();
  void zero();

  T* data_ptr(){return data.get();};
private:
  void recurse_print(const size_t level, const size_t index, const bool last);
};

#endif