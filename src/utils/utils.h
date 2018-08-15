//  A collection of utility functions.

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <string.h>
#include <vector>
#include "timer.h"

#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef DEBUG
#define LOG(x)                                                         \
  do {                                                                 \
    std::cerr << "LOG(line: " << __LINE__ << " file: " << __FILENAME__ \
              << "): " << x << std::endl;                              \
  } while (0)

#else
#define LOG(x)
#endif

#define ASSERT(condition, message)                                       \
  {                                                                      \
    if (!(condition)) {                                                  \
      std::cerr << "ASSERTION ERROR\nmessage: " << message               \
                << " condition: " << #condition << " @ " << __FILENAME__ \
                << " (" << __LINE__ << ")" << std::endl;                 \
    }                                                                    \
  }

#endif