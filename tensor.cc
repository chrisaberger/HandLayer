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
  t.data = std::shared_ptr<T>(rawptr);
  return t;
}

template <class T>
Tensor<T> Tensor<T>::from_vec(const std::vector<T>& vec,
                              const std::vector<size_t>& shape_in) {
  Tensor<T> t;
  t.shape = shape_in;
  t.numel = vec.size();

  T* rawptr = (T*)malloc(sizeof(T) * t.numel);
  memcpy(rawptr, vec.data(), t.numel * sizeof(T));
  t.data = std::shared_ptr<T>(rawptr);
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

  // Allocate our data.
  T* rawptr = (T*)malloc(sizeof(T) * numel);
  data = std::shared_ptr<T>(rawptr);
}

template<class T>     
T& Tensor<T>::operator[](const size_t i)
{
  assert(i < numel);
  return data_ptr()[i];
}

template<class T>     
const T Tensor<T>::operator[](const size_t i) const
{
  assert(i < numel);
  return data_ptr()[i];
}

template<class T>     
T& Tensor<T>::operator()(const size_t i, const size_t j)
{
    assert(shape.size() == 2);
    const size_t index = i * shape[1] + j;
    assert(index < numel);
    return data_ptr()[index];
}

template<class T>     
const T Tensor<T>::operator()(const size_t i, const size_t j) const
{
    assert(shape.size() == 2);
    const size_t index = i * shape[1] + j;
    assert(index < numel);
    return data_ptr()[index];
}

template<class T>     
void Tensor<T>::zero(){
  memset(data_ptr(), 0, numel * sizeof(T));
}

template<class T>     
void Tensor<T>::copy_data_from(const Tensor<T>& src){
  assert(src.numel == numel);
  memcpy(data_ptr(), src.data_ptr(), sizeof(T) * src.numel);
}

template<class T>     
Tensor<T> Tensor<T>::copy(const Tensor<T>& src){
  Tensor<T> dst;
  dst.numel = src.numel;
  dst.shape = src.shape;

  T* dst_data = (T*)malloc(sizeof(T) * src.numel);
  memcpy(dst_data, src.data_ptr(), sizeof(T) * src.numel);
  dst.data = std::shared_ptr<T>(dst_data);
  return dst;
}


template<class T>
Tensor<T> Tensor<T>::view(const size_t start, const size_t end) const{
  assert(shape.size() > 0);
  Tensor<T> t;
  const size_t elem_per_dim_1 = numel/shape[0];
  t.numel = (end-start)*elem_per_dim_1;
  t.shape = shape;
  t.shape[0] = end-start;
  t.data = data;
  t._ptr_offset = start * elem_per_dim_1;
  return t;
}


template <class T>
void Tensor<T>::recurse_print(const size_t level, const size_t indexes,
                              const bool last) const {
  if (level == shape.size() - 1) {
    std::cout << " [ ";
    for (size_t i = 0; i < shape[level]; ++i) {
      std::cout << data_ptr()[indexes + i];
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
    size_t new_indexes = 1;
    for (size_t s_idx = level + 1; s_idx < shape.size(); ++s_idx) {
      new_indexes *= shape[s_idx];
    }
    recurse_print(level + 1, i*new_indexes+indexes, i == shape[level] - 1);
  }
  std::cout << " ]";
  if(!last and level != 0){
    std::cout << ", " << std::endl;;
  }
}

template<class T>
void Tensor<T>::print() const {
  std::cout << "Tensor(";
  //auto a = (*this)[0][0];
  recurse_print(0, 0, false);
  std::cout << ")" << std::endl;
}

template class Tensor<size_t>;
template class Tensor<bfloat16>;
template class Tensor<float>;