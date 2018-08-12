#ifndef LOGSOFTMAX_H_
#define LOGSOFTMAX_H_

#include <math.h>
#include <vector>
#include "tensor.h"


/*
def stablesoftmax(x, exp_fn):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    #DEBUG print(str(np.amax(shiftx)) + " " + str(np.amin(shiftx)))
    exps = exp_fn(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)
*/

template <class T>
struct LogSoftmax {

  static Tensor<T> forward(const Tensor<T>& inp){

    Tensor<T> out = Tensor<T>::copy(inp);
    assert(inp.shape.size() == 2);
    for(size_t i = 0; i < inp.shape[0]; ++i){
      T value = (T)0;
  

      T max_value = (T)0;
      std::cout << "HHERE" << std::endl;
      
      // Find max value.
      for(size_t j = 0; j < inp.shape[1]; ++j){
        if(inp(i,j) > max_value){
          max_value = inp(i,j);
        }
      }

      // Stable softmax
      Tensor<T> new_t = out.view(i, i+1);
      T sum = (T)0;
      new_t.print();
      for(size_t j = 0; j < inp.shape[1]; ++j){
        //new_t[j] -= max_value;
        new_t[j] = exp(new_t[j]);
        sum += new_t[j];
      }
      for(size_t j = 0; j < inp.shape[1]; ++j){
        std::cout << new_t[j] << " " <<  sum << std::endl;
        new_t[j] = log(new_t[j] / sum);
      }

      new_t.print();
  
      std::cout << max_value << std::endl;

    }
    return out;
  }

};

#endif