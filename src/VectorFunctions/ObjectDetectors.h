#pragma once

namespace ASSET {

#define CREATE_MEMBER_DETECTOR(X)                             \
  template<typename T>                                        \
  class Detect_##X {                                          \
    struct Fallback {                                         \
      int X;                                                  \
    };                                                        \
    struct Derived : T, Fallback {};                          \
                                                              \
    template<typename U, U>                                   \
    struct Check;                                             \
                                                              \
    typedef char ArrayOfOne[1];                               \
    typedef char ArrayOfTwo[2];                               \
                                                              \
    template<typename U>                                      \
    static ArrayOfOne &func(Check<int Fallback::*, &U::X> *); \
    template<typename U>                                      \
    static ArrayOfTwo &func(...);                             \
                                                              \
   public:                                                    \
    typedef Detect_##X type;                                  \
    enum {                                                    \
      value = sizeof(func<Derived>(0)) == 2                   \
    };                                                        \
  };

}  // namespace ASSET