#pragma once
#include "Utils/SizingHelpers.h"
#include "pch.h"

namespace ASSET {

  template<int _XV, int _UV, int _PV>
  struct ODEConstSizes {
    static const int XV = _XV;
    static const int UV = _UV;
    static const int PV = _PV;
    static const int XtV = SZ_SUM<_XV, 1>::value;
    static const int XtUV = SZ_SUM<_XV, 1, _UV>::value;
    static const int XtUPV = SZ_SUM<_XV, 1, _UV, _PV>::value;
  };

  template<int _XV, int _UV, int _PV>
  struct ODEXVSizes : ODEConstSizes<_XV, _UV, _PV> {
    inline int TVar() const {
      return this->XV;
    }
    inline int XVars() const {
      return this->XV;
    }
    inline int XtVars() const {
      return this->XtV;
    }
    void setXVars(int xv) {
    }
  };

  template<int _UV, int _PV>
  struct ODEXVSizes<-1, _UV, _PV> : ODEConstSizes<-1, _UV, _PV> {
    inline int TVar() const {
      return this->XVdynamic;
    }
    inline int XVars() const {
      return this->XVdynamic;
    }
    inline int XtVars() const {
      return this->XtVdynamic;
    }
    void setXVars(int xv) {
      this->XVdynamic = xv;
      this->XtVdynamic = xv + 1;
    }

   protected:
    int XVdynamic = 0;
    int XtVdynamic = 0;
  };

  template<int _XV, int _UV, int _PV>
  struct ODEXUVSizes : ODEXVSizes<_XV, _UV, _PV> {
    inline int UVars() const {
      return this->UV;
    }
    inline int XtUVars() const {
      return this->UV + this->XtVars();
    }
    void setUVars(int uv) {
    }
  };
  template<int _XV, int _PV>
  struct ODEXUVSizes<_XV, -1, _PV> : ODEXVSizes<_XV, -1, _PV> {
    inline int UVars() const {
      return this->UVdynamic;
    }
    inline int XtUVars() const {
      return this->UVdynamic + this->XtVars();
    }
    void setUVars(int uv) {
      this->UVdynamic = uv;
    }

   protected:
    int UVdynamic = 0;
  };

  template<int _XV, int _UV, int _PV>
  struct ODEXUPVSizes : ODEXUVSizes<_XV, _UV, _PV> {
    inline int PVars() const {
      return this->PV;
    }
    inline int XtUPVars() const {
      return this->PV + this->XtUVars();
    }
    void setPVars(int pv) {
    }
    void setXtUPVars(int xv, int uv, int pv) {
      this->setXVars(xv);
      this->setUVars(uv);
      this->setPVars(pv);
    }
  };


  template<int _XV, int _UV>
  struct ODEXUPVSizes<_XV, _UV, -1> : ODEXUVSizes<_XV, _UV, -1> {
    inline int PVars() const {
      return this->PVdynamic;
    }
    inline int XtUPVars() const {
      return this->PVdynamic + this->XtUVars();
    }
    void setPVars(int pv) {
      this->PVdynamic = pv;
    }
    void setXtUPVars(int xv, int uv, int pv) {
      this->setXVars(xv);
      this->setUVars(uv);
      this->setPVars(pv);
    }


   protected:
    int PVdynamic = 0;
  };


  template<int _XV, int _UV, int _PV>
  struct ODESize : ODEXUPVSizes<_XV, _UV, _PV> {


    std::vector<int> Xidxs() const {
      std::vector<int> idxs(this->XVars());
      std::iota(idxs.begin(), idxs.end(), 0);
      return idxs;
    }
    std::vector<int> Xtidxs() const {
      std::vector<int> idxs(this->XtVars());
      std::iota(idxs.begin(), idxs.end(), 0);
      return idxs;
    }
    std::vector<int> XtUidxs() const {
      std::vector<int> idxs(this->XtUVars());
      std::iota(idxs.begin(), idxs.end(), 0);
      return idxs;
    }
    std::vector<int> Uidxs() const {
      std::vector<int> idxs(this->UVars());
      std::iota(idxs.begin(), idxs.end(), this->XtVars());
      return idxs;
    }


    std::vector<int> idxs_impl(const std::vector<int>& zidxs, const std::vector<int>& idxs) const {

      auto minelem = *std::min_element(zidxs.begin(), zidxs.end());
      auto maxelem = *std::max_element(zidxs.begin(), zidxs.end());

      if (minelem < 0 || maxelem >= idxs.size()) {
        throw std::invalid_argument("Indexing error in ODESizes idxs");
      }

      std::vector<int> nidxs(zidxs.size());

      for (int i = 0; i < zidxs.size(); i++) {
        nidxs[i] = idxs[zidxs[i]];
      }


      return nidxs;
    }

    std::vector<int> Xidxs(const std::vector<int>& zidxs) const {
      return idxs_impl(zidxs, this->Xidxs());
    }
    std::vector<int> Xtidxs(const std::vector<int>& zidxs) const {
      return idxs_impl(zidxs, this->Xtidxs());
    }
    std::vector<int> XtUidxs(const std::vector<int>& zidxs) const {
      return idxs_impl(zidxs, this->XtUidxs());
    }
    std::vector<int> Uidxs(const std::vector<int>& zidxs) const {
      return idxs_impl(zidxs, this->Uidxs());
    }


    template<class Obj, class Derived>
    static void BuildODESizeMembers(Obj& obj) {
      obj.def("XVars", &Derived::XVars);
      obj.def("UVars", &Derived::UVars);
      obj.def("PVars", &Derived::PVars);
      obj.def("TVar", &Derived::TVar);
      obj.def("tVar", &Derived::TVar);  // Capital is inconsistent in hindsight

      obj.def("XtVars", &Derived::XtVars);
      obj.def("XtUVars", &Derived::XtUVars);
      obj.def("XtUPVars", &Derived::XtUPVars);


      obj.def("Xidxs", py::overload_cast<>(&Derived::Xidxs, py::const_));
      obj.def("Xidxs", py::overload_cast<const std::vector<int>&>(&Derived::Xidxs, py::const_));

      obj.def("Xtidxs", py::overload_cast<>(&Derived::Xtidxs, py::const_));
      obj.def("Xtidxs", py::overload_cast<const std::vector<int>&>(&Derived::Xtidxs, py::const_));


      obj.def("XtUidxs", py::overload_cast<>(&Derived::XtUidxs, py::const_));
      obj.def("XtUidxs", py::overload_cast<const std::vector<int>&>(&Derived::XtUidxs, py::const_));

      obj.def("Uidxs", py::overload_cast<>(&Derived::Uidxs, py::const_));
      obj.def("Uidxs", py::overload_cast<const std::vector<int>&>(&Derived::Uidxs, py::const_));
    }
  };


}  // namespace ASSET
