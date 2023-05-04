#pragma once

namespace ASSET {

  template<int IR, int OR>
  struct InputOutputSize {
    static const int InputRows = IR;
    static const int OutputRows = OR;
  };

  template<>
  struct InputOutputSize<-1, -1> {
    int InputRows = 0;
    int OutputRows = 0;
  };

  template<int OR>
  struct InputOutputSize<-1, OR> {
    int InputRows = 0;
    static const int OutputRows = OR;
  };

  template<int IR>
  struct InputOutputSize<IR, -1> {
    static const int InputRows = IR;
    int OutputRows = 0;
  };

}  // namespace ASSET
