#include "tensor.h"
#include "cnpy.h"

template <class T>
Tensor<T>::Tensor(const std::string filename) {
  cnpy::NpyArray arr = cnpy::npy_load(filename);
  shape = arr.shape;
  numel = arr.num_vals;

  T* mydata = (T*) malloc(arr.word_size * arr.num_vals * sizeof(T));
  T* loaded_data = arr.data<T>();
  memcpy(mydata, loaded_data, arr.word_size * arr.num_vals * sizeof(T));
  data = std::unique_ptr<T>(mydata);
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