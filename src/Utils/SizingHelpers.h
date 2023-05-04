#pragma once

namespace ASSET {

  template<int Arg>
  struct SZ_NEG {
    enum {
      value = ((Arg < 0) ? -1 : Arg)
    };
  };

  template<int Arg1, int Arg2>
  struct SZ_SUMOP {
    enum {
      value = ((Arg1 < 0 || Arg2 < 0) ? -1 : Arg1 + Arg2),
      Identity = 0
    };
  };

  template<int Arg1, int Arg2>
  struct SZ_DIFF {
    enum {
      value = ((Arg1 < 0 || Arg2 < 0) ? -1 : Arg1 - Arg2),
      Identity = 0
    };
  };

  template<int Arg1, int Arg2>
  struct SZ_MAXOP {
    enum {
      value = ((Arg1 < 0 || Arg2 < 0) ? -1 : ((Arg1 > Arg2) ? Arg1 : Arg2)),
      Identity = 0
    };
  };

  template<int Arg1, int Arg2>
  struct SZ_MINOP {
    enum {
      value = ((Arg1 < 0 || Arg2 < 0) ? -1 : ((Arg1 > Arg2) ? Arg2 : Arg1)),
      Identity = 10000
    };
  };

  template<int Arg1, int Arg2>
  struct SZ_MAXREALOP {
    enum {
      value = ((Arg1 > Arg2) ? Arg1 : Arg2),
      Identity = 0
    };
  };
  template<int Arg1, int Arg2>
  struct SZ_MINREALOP {
    enum {
      value = ((Arg1 < 0 || Arg2 < 0) ? -1 : ((Arg1 > Arg2) ? Arg1 : Arg2)),
      Identity = 0
    };
  };

  template<int Arg1, int Arg2>
  struct SZ_LSUMOP {
    enum {
      value = ((Arg1 < 0) ? -1 : Arg1 + Arg2),
      Identity = 0
    };
  };

  template<int Arg1, int Arg2>
  struct SZ_PRODOP {
    enum {
      value = ((Arg1 < 0 || Arg2 < 0) ? -1 : Arg1 * Arg2),
      Identity = 1
    };
  };

  template<int Arg1, int Arg2>
  struct SZ_DIVOP {
    enum {
      value = ((Arg1 < 0 || Arg2 < 0) ? -1 : Arg1 / Arg2),
      Identity = 1
    };
  };

  template<template<int, int> class SZ_OP, int... Args>
  struct SZ_BINOP {};
  template<template<int, int> class SZ_OP, int Arg1, int Arg2, int... Args>
  struct SZ_BINOP<SZ_OP, Arg1, Arg2, Args...> {
    enum {
      value = SZ_OP < SZ_OP < Arg1,
      Arg2 > ::value,
      SZ_BINOP < SZ_OP,
      Args... > ::value > ::value
    };
  };

  template<template<int, int> class SZ_OP, int Arg1>
  struct SZ_BINOP<SZ_OP, Arg1> {
    enum {
      value = SZ_OP < Arg1,
      SZ_OP < 1,
      1 > ::Identity > ::value
    };
  };

  template<template<int, int> class SZ_OP>
  struct SZ_BINOP<SZ_OP> {
    enum {
      value = SZ_OP < 0,
      0 > ::Identity
    };
  };

  template<int... Args>
  using SZ_SUM = SZ_BINOP<SZ_SUMOP, Args...>;

  template<int... Args>
  using SZ_LSUM = SZ_BINOP<SZ_LSUMOP, Args...>;

  template<int... Args>
  using SZ_PROD = SZ_BINOP<SZ_PRODOP, Args...>;

  template<int... Args>
  using SZ_DIV = SZ_BINOP<SZ_DIVOP, Args...>;

  template<int... Args>
  using SZ_MAX = SZ_BINOP<SZ_MAXOP, Args...>;

  template<int... Args>
  using SZ_MIN = SZ_BINOP<SZ_MINOP, Args...>;

}  // namespace ASSET
