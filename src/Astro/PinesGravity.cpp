#include "ASSET_Extensions.h"
#include "PinesGravityModel.h"

namespace ASSET {
// The full recurrence is scalar-templated, so ASSET differentiates the same
// degree/order model used for propagation without a reduced gravity surrogate.
struct PinesGravity
    : VectorFunction<PinesGravity, 3, 3, AutodiffFwd, AutodiffFwd> {
  using Base = VectorFunction<PinesGravity, 3, 3, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;  // ASSET handles batch lanes.

  std::shared_ptr<const PinesGravityModel> model;
  PinesGravity(const std::string& file, int degree, double mu, double radius,
               const std::string& normalization)
      : model(std::make_shared<const PinesGravityModel>(file, degree, mu, radius, normalization)) {}

  template<class InType, class OutType>
  void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
    using Scalar = typename InType::Scalar;
    const auto result = model->acceleration_scalar<Scalar>({x[0], x[1], x[2]});
    auto& fx = fx_.const_cast_derived();
    for (int i = 0; i < 3; ++i) fx[i] = result[i];
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<PinesGravity>(m, name,
        "Differentiable fully normalized Pines gravity; km, seconds; includes central gravity.");
    obj.def(py::init<const std::string&, int, double, double, const std::string&>(),
            py::arg("coefficient_file"), py::arg("degree"), py::arg("mu"),
            py::arg("reference_radius"), py::arg("normalization") = "fully_normalized");
    obj.def_property_readonly("degree", [](const PinesGravity& f) { return f.model->degree(); });
    obj.def_property_readonly("order", [](const PinesGravity& f) { return f.model->degree(); });
    obj.def_property_readonly("mu", [](const PinesGravity& f) { return f.model->mu(); });
    obj.def_property_readonly("reference_radius", [](const PinesGravity& f) { return f.model->reference_radius(); });
    obj.def_property_readonly("coefficient_file", [](const PinesGravity& f) { return f.model->coefficient_file(); });
    obj.def_property_readonly("normalization", [](const PinesGravity&) { return "fully_normalized"; });
    obj.def_property_readonly("supports_derivatives", [](const PinesGravity&) { return true; });
    obj.def("acceleration", [](const PinesGravity& f, const Eigen::Vector3d& r) {
      const auto a = f.model->acceleration({r[0], r[1], r[2]});
      return Eigen::Vector3d(a[0], a[1], a[2]);
    }, py::arg("position_ecef"), py::call_guard<py::gil_scoped_release>());
    Base::DenseBaseBuild(obj);
    // DenseBaseBuild exposes numerical evaluation; expression composition
    // needs its own overload for this extension type.
    obj.def("__call__", [](const PinesGravity& f, const GenericFunction<-1, -1>& position) {
      return GenericFunction<-1, -1>(f.eval(position));
    }, py::arg("position"));
  }
};

void BuildPinesGravity(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<PinesGravity>(m, "PinesGravity");
}
}  // namespace ASSET
