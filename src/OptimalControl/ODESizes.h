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


    std::map<std::string, Eigen::VectorXi> _idxs;

    void add_default_idxs() {
       for (int i = 0; i < this->XVars(); i++) {
           this->add_idx(std::to_string(i), this->Xidxs(i));
       }
       this->add_idx("t", this->XVars());
       for (int i = 0; i < this->UVars(); i++) {
           this->add_idx(std::to_string(i), this->Uidxs(i));
       }
       for (int i = 0; i < this->PVars(); i++) {
           this->add_idx(std::to_string(i), this->Pidxs(i));
       }
    }

    void add_idx(const std::string& name, const Eigen::VectorXi & idx) {
        if (_idxs.count(name)) {
        throw std::invalid_argument(fmt::format("Variable index group with name: {0:} already exists.", name));
        }
        if (idx.size() == 0) {
        throw std::invalid_argument(fmt::format("Variable index group with name: {0:} has no elements.", name));
        }
        _idxs[name]=idx;
    }

    void add_idx(const std::string& name, int indx) {
        Eigen::VectorXi idxv(1);
        idxv[0] = indx;
        this->add_idx(idxv);
    }

    Eigen::VectorXi idx(const std::string& name) const {
        if (_idxs.count(name)==0) {
        throw std::invalid_argument(
            fmt::format("No variable index group with name: {0:} exists.", name));
        }
        return _idxs.at(name);
    }

    void set_idxs(const std::map<std::string, Eigen::VectorXi>& idxs) {
        this->_idxs = idxs;
    }
    std::map<std::string, Eigen::VectorXi> get_idxs() const {
        return this->_idxs;
    }

    Eigen::VectorXi Xidxs() const {
      Eigen::VectorXi idxs(this->XVars());
      std::iota(idxs.begin(), idxs.end(), 0);
      return idxs;
    }
    Eigen::VectorXi Xtidxs() const {
      Eigen::VectorXi idxs(this->XtVars());
      std::iota(idxs.begin(), idxs.end(), 0);
      return idxs;
    }
    Eigen::VectorXi XtUidxs() const {
      Eigen::VectorXi idxs(this->XtUVars());
      std::iota(idxs.begin(), idxs.end(), 0);
      return idxs;
    }
    Eigen::VectorXi Uidxs() const {
      Eigen::VectorXi idxs(this->UVars());
      std::iota(idxs.begin(), idxs.end(), this->XtVars());
      return idxs;
    }
    Eigen::VectorXi Pidxs() const {
      Eigen::VectorXi idxs(this->UVars());
      std::iota(idxs.begin(), idxs.end(), this->XtUVars());
      return idxs;
    }


    Eigen::VectorXi idxs_impl(const Eigen::VectorXi& zidxs, const Eigen::VectorXi& idxs) const {

      auto minelem = *std::min_element(zidxs.begin(), zidxs.end());
      auto maxelem = *std::max_element(zidxs.begin(), zidxs.end());

      if (minelem < 0 || maxelem >= idxs.size()) {
        throw std::invalid_argument("Indexing error in ODESizes idxs");
      }

      Eigen::VectorXi nidxs(zidxs.size());

      for (int i = 0; i < zidxs.size(); i++) {
        nidxs[i] = idxs[zidxs[i]];
      }

      return nidxs;
    }

    Eigen::VectorXi idxs_impl(int zidx, const Eigen::VectorXi& idxs) const {
      Eigen::VectorXi zidxs(1);
      zidxs[0] = zidx;
      return this->idxs_impl(zidxs, idxs);
    }



    Eigen::VectorXi Xidxs(const Eigen::VectorXi& zidxs) const {
      return idxs_impl(zidxs, this->Xidxs());
    }
    Eigen::VectorXi Xtidxs(const Eigen::VectorXi& zidxs) const {
      return idxs_impl(zidxs, this->Xtidxs());
    }
    Eigen::VectorXi XtUidxs(const Eigen::VectorXi& zidxs) const {
      return idxs_impl(zidxs, this->XtUidxs());
    }
    Eigen::VectorXi Uidxs(const Eigen::VectorXi& zidxs) const {
      return idxs_impl(zidxs, this->Uidxs());
    }
    Eigen::VectorXi Pidxs(const Eigen::VectorXi& zidxs) const {
      return idxs_impl(zidxs, this->Pidxs());
    }


    Eigen::VectorXi Xidxs(int zidxs) const {
      return idxs_impl(zidxs, this->Xidxs());
    }
    Eigen::VectorXi Xtidxs(int zidxs) const {
      return idxs_impl(zidxs, this->Xtidxs());
    }
    Eigen::VectorXi XtUidxs(int zidxs) const {
      return idxs_impl(zidxs, this->XtUidxs());
    }
    Eigen::VectorXi Uidxs(int zidxs) const {
      return idxs_impl(zidxs, this->Uidxs());
    }
    Eigen::VectorXi Pidxs(int zidxs) const {
      return idxs_impl(zidxs, this->Pidxs());
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
      obj.def("Xidxs", py::overload_cast<const Eigen::VectorXi&>(&Derived::Xidxs, py::const_));

      obj.def("Xtidxs", py::overload_cast<>(&Derived::Xtidxs, py::const_));
      obj.def("Xtidxs", py::overload_cast<const Eigen::VectorXi&>(&Derived::Xtidxs, py::const_));


      obj.def("XtUidxs", py::overload_cast<>(&Derived::XtUidxs, py::const_));
      obj.def("XtUidxs", py::overload_cast<const Eigen::VectorXi&>(&Derived::XtUidxs, py::const_));

      obj.def("Uidxs", py::overload_cast<>(&Derived::Uidxs, py::const_));
      obj.def("Uidxs", py::overload_cast<const Eigen::VectorXi&>(&Derived::Uidxs, py::const_));

      obj.def("add_idx",
              py::overload_cast<const std::string & , const Eigen::VectorXi&>(&Derived::add_idx));
      obj.def("get_idxs", &Derived::get_idxs);
      obj.def("set_idxs", &Derived::set_idxs);
      obj.def("idx", &Derived::idx);



    }
  };


}  // namespace ASSET
