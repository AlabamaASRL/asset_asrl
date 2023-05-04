#pragma once

#include "bench/BenchTimer.h"
#include "pch.h"

namespace ASSET {

  template<class Functor, class... Args>
  double BenchLoopFunctor(std::string name, int n, const Functor& f, Args... args) {
    Eigen::BenchTimer t;

    std::cout << "Bench Start: " << name << std::endl;
    t.start();
    for (int i = 0; i < n; i++) {
      f(n, args...);
    }
    t.stop();
    std::cout << "Bench Stop : " << t.total() * 1000.0 << " ms" << std::endl << std::endl;
    return t.total();
  }

  template<class Functor, class... Args>
  double BenchFunctor(std::string name, const Functor& f, Args... args) {
    Eigen::BenchTimer t;
    std::cout << "Bench Start: " << name << std::endl;
    t.start();
    f(args...);
    t.stop();
    std::cout << "Bench Stop : " << t.total() * 1000.0 << " ms" << std::endl << std::endl;
    return t.total();
  }

}  // namespace ASSET
