#pragma once

#include "Conditional.h"

namespace ASSET {

  // Enum
  enum ComparativeFlags {
    MinFlag,
    MaxFlag,
  };

  ////////////////////////////////////////////////////////////////////////////////
  // Class Definition
  template<class First, class... Rest>
  struct ComparativeFunction<First, Rest...>
      : IfElseFunction<ConditionalStatement<First, ComparativeFunction<Rest...>>,
                       First,
                       ComparativeFunction<Rest...>> {
    using Second = ComparativeFunction<Rest...>;
    using BaseCond = ConditionalStatement<First, Second>;
    using Base = IfElseFunction<BaseCond, First, Second>;

    // Static Parameters
    static const bool IsComparative = true;

    // ---------------------------------------------------------------------------
    // Constructors
    ComparativeFunction() {
    }
    ComparativeFunction(ComparativeFlags type, First first, Rest... rest)
        : Base(BaseCond(first,
                        type == ComparativeFlags::MinFlag ? ConditionalFlags::LessThanFlag
                                                          : ConditionalFlags::GreaterThanFlag,
                        Second(type, rest...)),
               first,
               Second(type, rest...)) {
    }
  };

  // =============================================================================

  template<class First, class Second>
  struct ComparativeFunction<First, Second>
      : IfElseFunction<ConditionalStatement<First, Second>, First, Second> {
    using BaseCond = ConditionalStatement<First, Second>;
    using Base = IfElseFunction<BaseCond, First, Second>;

    // Static Parameters
    static const bool IsComparative = true;

    // ---------------------------------------------------------------------------
    // Constructors
    ComparativeFunction() {
    }
    ComparativeFunction(ComparativeFlags type, First first, Second second)
        : Base(BaseCond(first,
                        type == ComparativeFlags::MinFlag ? ConditionalFlags::LessThanFlag
                                                          : ConditionalFlags::GreaterThanFlag,
                        second),
               first,
               second) {
    }
  };


}  // namespace ASSET
