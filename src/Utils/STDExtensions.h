#pragma once

#include <type_traits>

namespace std {

  template<class T>
  struct remove_const_reference {
    using type = typename std::remove_const<typename std::remove_reference<T>::type>::type;
  };

}  // namespace std
