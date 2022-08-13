#pragma once

#include "Utils/SizingHelpers.h"

namespace ASSET {

template <int _XV, int _UV, int _PV>
struct ODEConstSizes {
  static const int XV = _XV;
  static const int UV = _UV;
  static const int PV = _PV;
  static const int XtV = SZ_SUM<_XV, 1>::value;
  static const int XtUV = SZ_SUM<_XV, 1, _UV>::value;
  static const int XtUPV = SZ_SUM<_XV, 1, _UV, _PV>::value;
};

template <int _XV, int _UV, int _PV>
struct ODEXVSizes : ODEConstSizes<_XV, _UV, _PV> {
  inline int TVar() const { return this->XV; }
  inline int XVars() const { return this->XV; }
  inline int XtVars() const { return this->XtV; }
  void setXVars(int xv) {}
};

template <int _UV, int _PV>
struct ODEXVSizes<-1, _UV, _PV> : ODEConstSizes<-1, _UV, _PV> {
  inline int TVar() const { return this->XVdynamic; }
  inline int XVars() const { return this->XVdynamic; }
  inline int XtVars() const { return this->XtVdynamic; }
  void setXVars(int xv) {
    this->XVdynamic = xv;
    this->XtVdynamic = xv + 1;
  }

 protected:
  int XVdynamic = 0;
  int XtVdynamic = 0;
};

template <int _XV, int _UV, int _PV>
struct ODEXUVSizes : ODEXVSizes<_XV, _UV, _PV> {
  inline int UVars() const { return this->UV; }
  inline int XtUVars() const { return this->UV + this->XtVars(); }
  void setUVars(int uv) {}
};
template <int _XV, int _PV>
struct ODEXUVSizes<_XV, -1, _PV> : ODEXVSizes<_XV, -1, _PV> {
  inline int UVars() const { return this->UVdynamic; }
  inline int XtUVars() const { return this->UVdynamic + this->XtVars(); }
  void setUVars(int uv) { this->UVdynamic = uv; }

 protected:
  int UVdynamic = 0;
};

template <int _XV, int _UV, int _PV>
struct ODESize : ODEXUVSizes<_XV, _UV, _PV> {
  inline int PVars() const { return this->PV; }
  inline int XtUPVars() const { return this->PV + this->XtUVars(); }
  void setPVars(int pv) {}
  void setXtUPVars(int xv, int uv, int pv) {
    this->setXVars(xv);
    this->setUVars(uv);
    this->setPVars(pv);
  }
};
template <int _XV, int _UV>
struct ODESize<_XV, _UV, -1> : ODEXUVSizes<_XV, _UV, -1> {
  inline int PVars() const { return this->PVdynamic; }
  inline int XtUPVars() const { return this->PVdynamic + this->XtUVars(); }
  void setPVars(int pv) { this->PVdynamic = pv; }
  void setXtUPVars(int xv, int uv, int pv) {
    this->setXVars(xv);
    this->setUVars(uv);
    this->setPVars(pv);
  }

 protected:
  int PVdynamic = 0;
};

}  // namespace ASSET
