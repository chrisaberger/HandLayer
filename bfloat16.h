#ifndef BFLOAT16_H_
#define BFLOAT16_H_

#include <iostream>
#include <string>

struct bfloat16 {

  float _data;


  friend std::ostream& operator<<(std::ostream& os, const bfloat16& dt);  

};

inline std::ostream& operator<<(std::ostream& os, const bfloat16& dt)  
{  
    os << std::to_string(dt._data);  
    return os;  
}  

#endif