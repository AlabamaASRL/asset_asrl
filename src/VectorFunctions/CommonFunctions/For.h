#pragma once

#include "NestedFunction.h"

namespace ASSET {

  //! Declaration of For_Impl
  template<class Derived, int N, class StartFunc, class BodyFunc>
  struct For_Impl;

  //! User-facing For class
  /*!
    \tparam N Max iteration counter. Iterates from 0 to N, inclusive.
    \tparam StartFunc The function that is affected by the loop.
    \tparam BodyFunc The function that is iteratively called on StartFunc.

    Format as follows:
    for(int i=0; i<=N; i++){
      start = body(i, start);
    }
  */
  template<int N, class StartFunc, class BodyFunc>
  struct For : For_Impl<For<N, StartFunc, BodyFunc>, N, StartFunc, BodyFunc> {
    using Base = For_Impl<For<N, StartFunc, BodyFunc>, N, StartFunc, BodyFunc>;
    For(StartFunc f) : Base::For_Impl(f) {};
  };

  //! Implementation of For loop
  /*!
    Uses template recursion to generate a function with the same effect as a for
    loop. The i-th level in the recursion inherits from the NestedFunction of
    BodyFunc::Definition<i> with For_Impl<i-1, StartFunc, BodyFunc>.
  */
  template<class Derived, int N, class StartFunc, class BodyFunc>
  struct For_Impl : NestedFunction_Impl<Derived,
                                        decltype(BodyFunc::template Definition<N>()),
                                        For_Impl<Derived, N - 1, StartFunc, BodyFunc>> {
    using OFuncType = decltype(BodyFunc::template Definition<N>());
    using IFuncType = For_Impl<Derived, N - 1, StartFunc, BodyFunc>;
    using Base = NestedFunction_Impl<Derived, OFuncType, IFuncType>;

    For_Impl() {};
    For_Impl(StartFunc f) : Base::NestedFunction_Impl(OFuncType(), IFuncType(f)) {};
  };

  //! Base Specialization of For_Impl
  /*!

  */
  template<class Derived, class StartFunc, class BodyFunc>
  struct For_Impl<Derived, 0, StartFunc, BodyFunc>
      : NestedFunction_Impl<Derived, decltype(BodyFunc::template Definition<0>()), StartFunc> {
    using OFuncType = decltype(BodyFunc::template Definition<0>());
    using Base = NestedFunction_Impl<Derived, OFuncType, StartFunc>;

    For_Impl() {};
    For_Impl(StartFunc inner) : Base::NestedFunction_Impl(OFuncType(), inner) {};
  };

}  // namespace ASSET
