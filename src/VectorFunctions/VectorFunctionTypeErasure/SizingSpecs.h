/*
File Name: SizingSpecs.h

File Description: Defines the rubber_types compatible type erasure spec SizableSpec
defining the ability to query the Input/Output rows of a type-erased vectorfunction as well as its
name and thread safety. Poorly named, will probably change later.


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

  struct SizableSpec {
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Concept {  // abstract base class for model.
      virtual ~Concept() = default;
      // Your (internal) interface goes here.
      virtual std::string name() const = 0;

      virtual int IRows() const = 0;
      virtual int ORows() const = 0;
      virtual bool thread_safe() const = 0;
    };
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class Holder>
    struct Model : public Holder, public virtual Concept {
      using Holder::Holder;
      // Pass through to encapsulated value.
      virtual std::string name() const override {
        return rubber_types::model_get(this).name();
      }

      virtual int IRows() const override {
        return rubber_types::model_get(*this).IRows();
      }
      virtual int ORows() const override {
        return rubber_types::model_get(*this).ORows();
      }
      virtual bool thread_safe() const override {
        return rubber_types::model_get(*this).thread_safe();
      }
    };
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class Container>
    struct ExternalInterface : public Container {
      using Container_ = Container;
      using Container_::Container_;

      // Define the external interface. Should match encapsulated type.
      std::string name() const {
        return rubber_types::interface_get(this).name();
      }
      int IRows() const {
        return rubber_types::interface_get(*this).IRows();
      }
      int ORows() const {
        return rubber_types::interface_get(*this).ORows();
      }
      bool thread_safe() const {
        return rubber_types::interface_get(*this).thread_safe();
      }
    };
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };

}  // namespace ASSET
