#include "tensor.h"
#include "cnpy.h"

template <class T>
Tensor<T> Tensor<T>::from_npy(const std::string filename) {
  cnpy::NpyArray arr = cnpy::npy_load(filename);

  Tensor<T> t;
  t.shape = arr.shape;
  t.numel = arr.num_vals;

  T* rawptr = (T*) malloc(arr.word_size * arr.num_vals * sizeof(T));
  T* loaded_data = arr.data<T>();
  memcpy(rawptr, loaded_data, arr.word_size * arr.num_vals * sizeof(T));
  t.data = std::unique_ptr<T>(rawptr);
  return t;
}

template <class T>
Tensor<T>::Tensor(std::vector<size_t> shape_in) {
  shape = shape_in;

  // Compute the number of elems from the shape.
  numel = 1;
  for(auto const& s: shape){
    numel *= s;
  }

  std::cout << "NUMEL: " << numel << std::endl;
  // Allocate our data.
  T* rawptr = (T*)malloc(sizeof(T) * numel);
  data = std::unique_ptr<T>(rawptr);
}

template<class T>     
T& Tensor<T>::operator[](const size_t i)
{
  assert(i < numel);
  return data.get()[i];
}

template<class T>     
const T Tensor<T>::operator[](const size_t i) const
{
  assert(i < numel);
  return data.get()[i];
}

template<class T>     
T& Tensor<T>::operator()(const size_t i, const size_t j)
{
    assert(shape.size() == 2);
    const size_t index = i * shape[1] + j;
    assert(index < numel);
    return data.get()[index];
}

template<class T>     
const T Tensor<T>::operator()(const size_t i, const size_t j) const
{
    assert(shape.size() == 2);
    const size_t index = i * shape[1] + j;
    assert(index < numel);
    return data.get()[index];
}

template<class T>     
void Tensor<T>::zero(){
  memset(data.get(), 0, numel * sizeof(T));
}

template<class T>     
Tensor<T> Tensor<T>::copy(const Tensor<T>& src){
  Tensor<T> dst;
  dst.numel = src.numel;
  dst.shape = src.shape;

  T* dst_data = (T*)malloc(sizeof(T) * src.numel);
  memcpy(dst_data, src.data.get(), sizeof(T) * src.numel);
  dst.data = std::unique_ptr<T>(dst_data);
  return dst;
}

template <class T>
void Tensor<T>::recurse_print(const size_t level, const size_t indexes,
                              const bool last) {
  if (level == shape.size() - 1) {
    std::cout << " [ ";
    for (size_t i = 0; i < shape[level]; ++i) {
      std::cout << data.get()[indexes + i];
      if(i != shape[level] -1){
        std::cout << ",";
      }
    }
    std::cout << " ]";
    if (!last) {
      std::cout << "," << std::endl;
    }
    return;
  }
  std::cout << "[";
  for (size_t i = 0; i < shape[level]; ++i) {
    recurse_print(level + 1, i * shape[level + 1], i == shape[level] - 1);
  }
  std::cout << " ]";
}

template<class T>
void Tensor<T>::print(){
  std::cout << "Tensor(";
  //auto a = (*this)[0][0];
  recurse_print(0, 0, false);
  std::cout << ")" << std::endl;
}

template class Tensor<float>;