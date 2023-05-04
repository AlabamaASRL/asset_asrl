/*
File Name: MemoryMangement.h

File Description: Defines ASSET's memory manager for allocation of temporary matrices in
VectorFunction expressions

////////////////////////////////////////////////////////////////////////////////

Original File Developer : James B. Pezent - jbpezent - jbpezent@crimson.ua.edu

Current File Maintainers:
    1. James B. Pezent - jbpezent         - jbpezent@crimson.ua.edu
    2. Full Name       - GitHub User Name - Current Email
    3. ....


Usage of this source code is governed by the license found
in the LICENSE file in ASSET's top level directory.

*/


#pragma once
#include <algorithm>
#include <tuple>
#include <utility>

#include "pch.h"

namespace ASSET {

  namespace detail {
    /// <summary>
    /// Bump Stack Allocator for data type Scalar.
    /// Attempts to fill allocations out of a single contigous Eigen vector.
    /// Whenever allocation requests exceed size of this vector, blocks are allocated from a
    /// new vector appended to a link list. Once all blocks are freed, resizes the main vector to the maximum
    /// size seen since last time. That way if the allocation pattern repeats no data will be pushed to the
    /// linked list.
    /// </summary>
    /// <typeparam name="Scalar"></typeparam>
    template<class Scalar>
    struct TempStack {

      TempStack(int InitSize) {
        resize(InitSize);
      }
      TempStack() : TempStack(64) {
      }
      void resize(int size) {
        this->Data.resize(size);
        DataSize = size;
        NextDataSize = size;
        NextStart = 0;
        OverFlowSize = 0;
      }

      inline Scalar* getBlock(int blocksize) {
        Scalar* dat;
        if (OverFlowSize == 0 && ((NextStart + blocksize) <= DataSize)) {
          // If blocksize can be fit into Data, give ptr to next free location

          this->Data.segment(NextStart, blocksize).setZero();
          dat = this->Data.data() + NextStart;
          NextStart += blocksize;
        } else {
          // If blocksize too large to fit in Data, allocate new vector in OverflowData

          this->OverFlowData.emplace_back(VectorX<Scalar>());
          this->OverFlowData.back().resize(blocksize);
          OverFlowSize += blocksize;
          NextDataSize = DataSize + OverFlowSize;
          dat = this->OverFlowData.back().data();
        }
        return dat;
      }

      inline void freeBlock(int blocksize) {
        if (OverFlowSize == 0)
          NextStart -= blocksize;
        else
          OverFlowSize -= blocksize;
        if (NextStart == 0 && OverFlowSize == 0 && (NextDataSize != DataSize)) {
          DataSize = NextDataSize;
          this->Data.resize(DataSize);
          if (OverFlowData.size() > 1)
            OverFlowData.clear();
        }
      }
      inline int size() {
        return Data.size();
      }

     private:
      VectorX<Scalar> Data;                     // Persistent Data Stack
      std::list<VectorX<Scalar>> OverFlowData;  // Temporary Overlow Data stacks
      int NextStart;                            // Next free index in Data where block can be allocated
      int DataSize;                             // Size of Data
      int OverFlowSize = 0;                     // Size of all blocks that have been spilled to OverFlowData
      int NextDataSize = 0;  // What size to resize Data too whenever we return top of stack
    };


    template<int... JS>
    struct NJumpTable {
      using sequence = std::tuple<std::integral_constant<int, JS>...>;
      template<class Ftype>
      static void run(Ftype&& f, int crit_size) {
        auto seq = sequence();
        bool done = false;
        ASSET::tuple_for_each(seq, [&](auto i) {
          if (!done && crit_size <= i.value) {
            f(i);
            done = true;
          }
        });
        if (!done)
          f(std::integral_constant<int, -1>());
      }
    };


    using OldDefaultJumpTable = NJumpTable<4, 8, 16>;
    using NewDefaultJumpTable = NJumpTable<8, 16, 64, 256, 384, 512, 1024, 2048>;


    template<int R, int C>
    struct RCBase {
      static const int rows = R;
      static const int cols = C;
      RCBase(int, int) {
      }
    };
    template<int R>
    struct RCBase<R, -1> {
      static const int rows = R;
      int cols;
      RCBase(int r, int c) : cols(c) {
      }
    };
    template<int C>
    struct RCBase<-1, C> {
      int rows;
      static const int cols = C;
      RCBase(int r, int c) : rows(r) {
      }
    };
    template<>
    struct RCBase<-1, -1> {
      int rows;
      int cols;
      RCBase(int r, int c) : rows(r), cols(c) {
      }
    };


    /// <summary>
    /// This holder type was neccessary to stop a slow constructor in std::tuple
    /// I dont know why it was slow  or why this class stops it since they are doing the same thing
    /// </summary>
    /// <typeparam name="...TempSpecs"></typeparam>
    template<int size, class... TempSpecs>
    struct MaxTempPack {
      std::tuple<typename std::remove_const_reference<TempSpecs>::type::template MaxMatType2<size>...> data;
      MaxTempPack(TempSpecs... tspecs) {
        auto tmpsp = std::make_tuple(tspecs...);

        constexpr int sds = sizeof...(tspecs);
        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              int rows = std::get<i.value>(tmpsp).rows;
              int cols = std::get<i.value>(tmpsp).cols;
              std::get<i.value>(data).resize(rows, cols);
            });
      }
    };


    template<class... TempSpecs>
    struct ExactTempPack {
      std::tuple<typename std::remove_const_reference<TempSpecs>::type::ExactTempType...> data;
      ExactTempPack() {
      }
    };


  }  // namespace detail

  /// <summary>
  ///  Template type for specifying the type and size of temporary matrix that must be created by the
  ///  allocator.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  template<class T>
  struct TempSpec : detail::RCBase<T::RowsAtCompileTime, T::ColsAtCompileTime> {
    using Base = detail::RCBase<T::RowsAtCompileTime, T::ColsAtCompileTime>;
    using ExactTempType = T;
    using MatType = T;
    using Scalar = typename T::Scalar;
    static constexpr int RowsAtCompileTime = T::RowsAtCompileTime;
    static constexpr int ColsAtCompileTime = T::ColsAtCompileTime;
    static constexpr bool IsConstantSize = (RowsAtCompileTime >= 0) && (ColsAtCompileTime >= 0);
    static constexpr bool IsArray = false;
    static constexpr bool IsVector = false;
    static constexpr bool IsTuple = false;

    template<int MR, int MC>
    using MaxMatType = Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, 0, MR, MC>;
    template<int value>
    using MaxMatType2 = Eigen::Matrix<Scalar,
                                      RowsAtCompileTime,
                                      ColsAtCompileTime,
                                      0,
                                      (RowsAtCompileTime == -1) ? value : RowsAtCompileTime,
                                      (ColsAtCompileTime == -1) ? value : ColsAtCompileTime>;


    TempSpec(int rows, int cols) : Base(rows, cols) {
    }
  };

  /// <summary>
  ///  Template type for specifying a constant size array of TempSpecs that must be allocated.
  /// </summary>
  /// <typeparam name="T"></typeparam>

  template<class T, int Size>
  struct ArrayOfTempSpecs {
    using Scalar = typename T::Scalar;
    using ExactTempType = std::array<T, Size>;
    using MatType = T;
    static constexpr int size = Size;
    static constexpr bool IsConstantSize = TempSpec<T>::IsConstantSize;
    static constexpr bool IsArray = true;
    static constexpr bool IsVector = false;
    static constexpr bool IsTuple = false;
    TempSpec<T> tspec;
    ArrayOfTempSpecs(int rows, int cols) : tspec(rows, cols) {
    }
  };

  /// <summary>
  ///  Template type for specifying a heterogenous tuple of TempSpecs that must be allocated.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  template<class... T>
  struct TupleOfTempSpecs {

    using Scalar =
        typename std::remove_const_reference<decltype(std::get<0>(std::tuple<T...>()))>::type::Scalar;

    using ExactTempType = std::tuple<T...>;

    static constexpr bool IsConstantSize = (... && TempSpec<T>::IsConstantSize);
    static constexpr bool IsArray = false;
    static constexpr bool IsVector = false;
    static constexpr bool IsTuple = true;
    static constexpr int size = sizeof...(T);

    std::tuple<TempSpec<T>...> tspecs;

    // TupleOfTempSpecs(TempSpec<T>... ts) :tspecs(ts...) {}

    TupleOfTempSpecs(std::tuple<TempSpec<T>...> tsp) : tspecs(tsp) {
    }
  };


  struct MemoryManager {
    using ScalarStackType = detail::TempStack<double>;
    using SuperScalarStackType = detail::TempStack<DefaultSuperScalar>;

    template<class JTable, class Func, class... TempSpecs>
    static void allocate_run_impl(const JTable& jt, int critical_size, Func&& f, const TempSpecs&... tspecs) {
      if constexpr ((... && TempSpecs::IsConstantSize)) {
        auto Temps = detail::ExactTempPack<typename std::remove_const_reference<TempSpecs>::type...>();
        std::apply(f, Temps.data);
      } else {
        if constexpr (OnlyOldTables)
          MemoryManager::run_old_table_impl(jt, critical_size, tspecs...);
        else if constexpr (OnlyNewTables)
          MemoryManager::run_new_table_impl(jt, f, tspecs...);
        else if constexpr (OnlyArena)
          MemoryManager::run_arena_impl(f, tspecs...);
        else {
          if (MemoryManager::UseArena)
            MemoryManager::run_arena_impl(f, tspecs...);
          else
            MemoryManager::run_new_table_impl(jt, f, tspecs...);
        }
      }
    }

    template<class Func, class... TempSpecs>
    static void allocate_run(int critical_size, Func&& f, const TempSpecs&... tspecs) {
      MemoryManager::allocate_run_impl(DefaultJumpTable(), critical_size, f, tspecs...);
    }

    /*These are safe to be called anywhere except inside an allocating function*/
    static void resize(int sizeScalar, int sizeSuper) {
      MemoryManager::ScalarStack.resize(sizeScalar);
      MemoryManager::SuperScalarStack.resize(sizeSuper);
    }
    static void resize(int size) {
      MemoryManager::resize(size, size);
    }
    static void enable_arena_memory(int sizeScalar, int sizeSuper) {
      MemoryManager::UseArena = true;
      MemoryManager::resize(sizeScalar, sizeSuper);
    }
    static void enable_arena_memory(int size) {
      MemoryManager::UseArena = true;
      MemoryManager::resize(size);
    }
    static void enable_arena_memory() {
      MemoryManager::UseArena = true;
      MemoryManager::resize(64);
    }
    static void disable_arena_memory() {
      MemoryManager::UseArena = false;
      MemoryManager::resize(64);
    }
    static bool arena_memory_enabled() {
      return MemoryManager::UseArena;
    }
    static int size_scalar() {
      return MemoryManager::ScalarStack.size();
    }
    static int size_super_scalar() {
      return MemoryManager::SuperScalarStack.size();
    }


    static void Build(py::module& m);

   private:
    static thread_local ScalarStackType ScalarStack;
    static thread_local SuperScalarStackType SuperScalarStack;
    static bool UseArena;


    static const bool OnlyOldTables = false;
    static const bool OnlyNewTables = false;
    static const bool OnlyArena = true;

    using DefaultJumpTable = typename std::
        conditional<OnlyOldTables, detail::OldDefaultJumpTable, detail::NewDefaultJumpTable>::type;

    template<class Scalar>
    static Scalar* getBlock(int blocksize) {
      if constexpr (std::is_same<Scalar, double>::value) {
        return MemoryManager::ScalarStack.getBlock(blocksize);
      } else {
        return MemoryManager::SuperScalarStack.getBlock(blocksize);
      }
    }
    template<class Scalar>
    static void freeBlock(int blocksize) {
      if constexpr (std::is_same<Scalar, double>::value) {
        return MemoryManager::ScalarStack.freeBlock(blocksize);
      } else {
        return MemoryManager::SuperScalarStack.freeBlock(blocksize);
      }
    }


    template<class... TempSpecs>
    inline static int count_blocksize(const TempSpecs&... tspecs) {
      int blksize = 0;
      auto CalcSpecSize = [](const auto& tspec) { return (tspec.rows * tspec.cols); };
      auto CountSpace = [&](const auto& tspec) {
        using type = typename std::remove_const_reference<decltype(tspec)>::type;

        if constexpr (type::IsConstantSize) {
          // Do nothing, allocate as constant size Eigen matrix on Stack
        } else if constexpr (type::IsArray) {
          blksize += CalcSpecSize(tspec.tspec) * tspec.size;
        } else if constexpr (type::IsTuple) {
          ASSET::tuple_for_each(tspec.tspecs, [&](const auto& tspeci) { blksize += CalcSpecSize(tspeci); });
        } else {
          blksize += CalcSpecSize(tspec);
        }
      };
      (CountSpace(tspecs), ...);
      return blksize;
    }


    template<class Scalar, class... TempSpecs>
    inline static auto make_temps(Scalar* data, const TempSpecs&... tspecs) {

      int start = 0;

      auto make_map = [&](const auto& tspec) {
        using type = typename std::remove_const_reference<decltype(tspec)>::type;
        using MAP = typename Eigen::Map<typename type::MatType>;
        int start_t = start;
        start += (tspec.rows * tspec.cols);
        return MAP(data + start_t, tspec.rows, tspec.cols);
      };

      auto make_temp = [&](const auto& tspec) {
        using type = typename std::remove_const_reference<decltype(tspec)>::type;
        if constexpr (type::IsConstantSize) {
          // Do nothing, allocate as constant size Eigen matrix on Stack

          return typename type::ExactTempType();
        } else if constexpr (type::IsArray) {
          auto array_temp = [&](auto i) { return make_map(tspec.tspec); };
          return MemoryManager::make_map_array(array_temp, std::integral_constant<int, type::size>());
        } else if constexpr (type::IsTuple) {
          auto tuple_temp = [&](const auto&... tspeci) { return std::tuple {make_map(tspeci)...}; };
          return std::apply(tuple_temp, tspec.tspecs);
        } else {
          return make_map(tspec);
        }
      };

      return std::tuple {make_temp(tspecs)...};
    }


    template<class Func, class... TempSpecs>
    inline static void run_arena_impl(Func&& f, const TempSpecs&... tspecs) {
      using Scalar =
          typename std::remove_const_reference<decltype(std::get<0>(std::tuple {tspecs...}))>::type::Scalar;

      int blksize = MemoryManager::count_blocksize(tspecs...);

      /////////////////////////////////////////////////////////
      Scalar* data = MemoryManager::getBlock<Scalar>(blksize);
      //////////////////////////////////////////////////////////

      auto Temps = MemoryManager::make_temps(data, tspecs...);
      std::apply(f, Temps);

      //////////////////////////////////////////////////////////
      data = nullptr;
      MemoryManager::freeBlock<Scalar>(blksize);
      //////////////////////////////////////////////////////////
    }

    template<class JTable, class Func, class... TempSpecs>
    inline static void run_old_table_impl(const JTable& jt,
                                          int critical_size,
                                          Func&& f,
                                          const TempSpecs&... tspecs) {
      auto MaxImpl = [&](auto maxsize) {
        auto Temps =
            detail::MaxTempPack<maxsize.value, typename std::remove_const_reference<TempSpecs>::type...>(
                tspecs...);
        std::apply(f, Temps.data);
      };
      JTable::run(MaxImpl, critical_size);
    }

    template<class JTable, class Func, class... TempSpecs>
    inline static void run_new_table_impl(const JTable& jt, Func&& f, const TempSpecs&... tspecs) {
      using Scalar =
          typename std::remove_const_reference<decltype(std::get<0>(std::tuple {tspecs...}))>::type::Scalar;


      int blksize = MemoryManager::count_blocksize(tspecs...);

      auto MaxImpl = [&](auto maxsize) {
        Eigen::Matrix<Scalar, maxsize.value, 1> Data;
        Scalar* data = Data.data();
        int start = 0;
        auto MakeTemps = [&](auto tspec) {
          using type = typename std::remove_const_reference<decltype(tspec)>::type;
          if constexpr (tspec.IsConstantSize)
            return typename type::MatType();
          else {
            int start_t = start;
            start += (tspec.rows * tspec.cols);
            using MAP = typename Eigen::Map<typename type::MatType>;
            return MAP(data + start_t, tspec.rows, tspec.cols);
          }
        };
        auto Temps = std::tuple {MakeTemps(tspecs)...};
        std::apply(f, Temps);
      };

      JTable::run(MaxImpl, blksize);
    }

    template<class Function, std::size_t... Indices>
    static auto make_map_array_helper(Function f, std::index_sequence<Indices...>)
        -> std::array<typename std::invoke_result<Function, std::size_t>::type, sizeof...(Indices)> {
      return {{f(Indices)...}};
    }

    template<int N, class Function>
    static auto make_map_array(Function f, std::integral_constant<int, N> n)
        -> std::array<typename std::invoke_result<Function, std::size_t>::type, N> {
      return make_map_array_helper(f, std::make_index_sequence<N> {});
    }
  };


}  // namespace ASSET
