/*
File Name: DeepCopySpecs.h

File Description: Defines the rubber_types compatible type erasure spec SingleDeepCopySpec<T>
defining an objects ability to deep copy itself into another object of type T. This method allows for
underlying type erased objects to be copied into another compatible type-erasure class without introducing
another layer of indirection. Also defines another spec DeepCopySpecOverloaded<Specs> which allows multiple
SingleDeepCopySpecs to be merged together. This had to be done explicitly rather than through
rubber_types::MergeSpecs to bypass reubber_types difficulty in handling overloaded functions. inally at the
end of the file, I define an alias template that combines the two classes into the pseudo-spec that shoule be
used by other files.


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

#include "pch.h"

namespace ASSET {

  template<class T>
  struct SingleDeepCopySpec {
    struct Concept {  // abstract base class for model.
      virtual ~Concept() = default;
      // Your (internal) interface goes here.
      virtual void deep_copy_into(T& obj) const = 0;
    };
    template<class Holder>
    struct Model : public Holder, public virtual Concept {
      using Holder::Holder;
      // Pass through to encapsulated value.
      virtual void deep_copy_into(T& obj) const override {
        return rubber_types::model_get(*this).deep_copy_into(obj);
      }
    };
    template<class Container>
    struct ExternalInterface : public Container {
      using Container_ = Container;
      using Container_::Container_;

      // Define the external interface. Should match encapsulated type.
      void deep_copy_into(T& obj) const {
        return rubber_types::interface_get(*this).deep_copy_into(obj);
      }
    };
  };

  template<class... Specs>
  struct DeepCopySpecOverloaded {
    struct Concept : public virtual rubber_types::detail::ConceptOf<Specs>... {
      using Specs::Concept::deep_copy_into...;
    };

    template<class Holder>
    struct Model : public rubber_types::detail::Rfold<rubber_types::detail::ModelOf, Specs..., Holder>,
                   public virtual Concept {
      using Base = rubber_types::detail::Rfold<rubber_types::detail::ModelOf, Specs..., Holder>;
      using Base::Base;
    };
    template<class Container>
    struct ExternalInterface
        : public rubber_types::detail::Rfold<rubber_types::detail::ExternalInterfaceOf, Specs..., Container> {
      using Base =
          rubber_types::detail::Rfold<rubber_types::detail::ExternalInterfaceOf, Specs..., Container>;
      using Base::Base;
      template<class T>
      void deep_copy_into(T& obj) const {
        return rubber_types::interface_get(*this).deep_copy_into(obj);
      }
    };
  };

  template<class... Types>
  using DeepCopySpecs = DeepCopySpecOverloaded<SingleDeepCopySpec<Types>...>;

}  // namespace ASSET
