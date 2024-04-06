#include <bind/Utils/BindUtils.h>

void ASSET::BindUtils(py::module& m) {
  auto um = m.def_submodule("Utils", "Contains miscilanaeous utilities");
  um.def("get_core_count", &ASSET::get_core_count);
}
