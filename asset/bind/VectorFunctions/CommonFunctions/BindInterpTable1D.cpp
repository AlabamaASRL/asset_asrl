#include <bind/VectorFunctions/CommonFunctions/BindInterpTable1D.h>

void ASSET::BindInterpTable1D(py::module& m) {
  using MatType = InterpTable1D::MatType;
  auto obj = py::class_<InterpTable1D, std::shared_ptr<InterpTable1D>>(m, "InterpTable1D");

  obj.def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, int, std::string>(),
          py::arg("ts"),
          py::arg("Vs"),
          py::arg("axis") = 0,
          py::arg("kind") = std::string("cubic"));

  obj.def(py::init<const Eigen::VectorXd&, const MatType&, int, std::string>(),
          py::arg("ts"),
          py::arg("Vs"),
          py::arg("axis") = 0,
          py::arg("kind") = std::string("cubic"));

  obj.def(py::init<const std::vector<Eigen::VectorXd>&, int, std::string>(),
          py::arg("Vts"),
          py::arg("tvar") = -1,
          py::arg("kind") = std::string("cubic"));

  obj.def("interp", py::overload_cast<double>(&InterpTable1D::interp, py::const_));
  obj.def("interp", py::overload_cast<const Eigen::VectorXd&>(&InterpTable1D::interp, py::const_));

  obj.def("__call__", py::overload_cast<double>(&InterpTable1D::interp, py::const_), py::is_operator());
  obj.def("__call__",
          py::overload_cast<const Eigen::VectorXd&>(&InterpTable1D::interp, py::const_),
          py::is_operator());

  obj.def("__call__", [](std::shared_ptr<InterpTable1D>& self, const GenericFunction<-1, 1>& t) {
    py::object pyfun;
    if (self->vlen == 1) {
      auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(self).eval(t));
      pyfun = py::cast(f);
    } else {
      auto f = GenericFunction<-1, -1>(InterpFunction1D<-1>(self).eval(t));
      pyfun = py::cast(f);
    }
    return pyfun;
  });

  obj.def("__call__", [](std::shared_ptr<InterpTable1D>& self, const Segment<-1, 1, -1>& t) {
    py::object pyfun;

    if (self->vlen == 1) {
      auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(self).eval(t));
      pyfun = py::cast(f);
    } else {
      auto f = GenericFunction<-1, -1>(InterpFunction1D<-1>(self).eval(t));
      pyfun = py::cast(f);
    }
    return pyfun;
  });

  obj.def("interp_deriv1", &InterpTable1D::interp_deriv1);
  obj.def("interp_deriv2", &InterpTable1D::interp_deriv2);

  obj.def_readwrite("WarnOutOfBounds", &InterpTable1D::WarnOutOfBounds);
  obj.def_readwrite("ThrowOutOfBounds", &InterpTable1D::ThrowOutOfBounds);

  obj.def("sf", [](std::shared_ptr<InterpTable1D>& self) {
    if (self->vlen != 1) {
      throw std::invalid_argument(
          "InterpTable1D storing Vector data cannot be converted to Scalar Function.");
    }
    return GenericFunction<-1, 1>(InterpFunction1D<1>(self));
  });
  obj.def("vf", [](std::shared_ptr<InterpTable1D>& self) {
    return GenericFunction<-1, -1>(InterpFunction1D<-1>(self));
  });
}
