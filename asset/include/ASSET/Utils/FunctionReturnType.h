#pragma once

namespace ASSET {

  template<typename T>
  struct return_type;
  template<typename R, typename... Args>
  struct return_type<R (*)(Args...)> {
    using type = R;
  };
  template<typename R, typename C, typename... Args>
  struct return_type<R (C::*)(Args...)> {
    using type = R;
  };
  template<typename R, typename C, typename... Args>
  struct return_type<R (C::*)(Args...) const> {
    using type = R;
  };
  template<typename R, typename C, typename... Args>
  struct return_type<R (C::*)(Args...) volatile> {
    using type = R;
  };
  template<typename R, typename C, typename... Args>
  struct return_type<R (C::*)(Args...) const volatile> {
    using type = R;
  };
  template<typename T>
  using return_type_t = typename return_type<T>::type;

}  // namespace ASSET
