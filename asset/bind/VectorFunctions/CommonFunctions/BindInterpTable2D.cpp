#include <bind/VectorFunctions/CommonFunctions/BindInterpTable2D.h>

void ASSET::BindInterpTable2D(py::module& m) {
  using MatType = InterpTable2D::MatType;
  auto obj = py::class_<InterpTable2D, std::shared_ptr<InterpTable2D>>(m, "InterpTable2D");

  obj.def(py::init<const Eigen::VectorXd&,
                   const Eigen::VectorXd&,
                   const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>&,
                   std::string>(),
          py::arg("xs"),
          py::arg("ys"),
          py::arg("Z"),
          py::arg("kind") = std::string("cubic"));

  obj.def("interp", py::overload_cast<double, double>(&InterpTable2D::interp, py::const_));
  obj.def("interp", py::overload_cast<const MatType&, const MatType&>(&InterpTable2D::interp, py::const_));

  obj.def_readwrite("WarnOutOfBounds", &InterpTable2D::WarnOutOfBounds);
  obj.def_readwrite("ThrowOutOfBounds", &InterpTable2D::ThrowOutOfBounds);

  obj.def("interp_deriv1", &InterpTable2D::interp_deriv1);
  obj.def("interp_deriv2", &InterpTable2D::interp_deriv2);

  obj.def("find_elem", &InterpTable2D::find_elem);

  obj.def(
      "__call__", py::overload_cast<double, double>(&InterpTable2D::interp, py::const_), py::is_operator());
  obj.def("__call__",
          py::overload_cast<const MatType&, const MatType&>(&InterpTable2D::interp, py::const_),
          py::is_operator());

  obj.def("__call__",
          [](std::shared_ptr<InterpTable2D>& self,
             const GenericFunction<-1, 1>& x,
             const GenericFunction<-1, 1>& y) {
            return GenericFunction<-1, 1>(InterpFunction2D(self).eval(stack(x, y)));
          });

  obj.def("__call__",
          [](std::shared_ptr<InterpTable2D>& self, const Segment<-1, 1, -1>& x, const Segment<-1, 1, -1>& y) {
            return GenericFunction<-1, 1>(InterpFunction2D(self).eval(stack(x, y)));
          });

  obj.def("__call__", [](std::shared_ptr<InterpTable2D>& self, const Segment<-1, 2, -1>& xy) {
    return GenericFunction<-1, 1>(InterpFunction2D(self).eval(xy));
  });

  obj.def("__call__", [](std::shared_ptr<InterpTable2D>& self, const GenericFunction<-1, -1>& xy) {
    return GenericFunction<-1, 1>(InterpFunction2D(self).eval(xy));
  });

  obj.def("sf", [](std::shared_ptr<InterpTable2D>& self) {
    return GenericFunction<-1, 1>(InterpFunction2D(self));
  });
  obj.def("vf", [](std::shared_ptr<InterpTable2D>& self) {
    return GenericFunction<-1, -1>(InterpFunction2D(self));
  });

  m.def("InterpTable2DSpeedTest",
        [](const GenericFunction<-1, 1>& tabf,
           double xl,
           double xu,
           double yl,
           double yu,
           int nsamps,
           bool lin) {
          Eigen::ArrayXd xsamps;
          xsamps.setRandom(nsamps);
          xsamps += 1;
          xsamps /= 2;
          xsamps *= (xu - xl);
          xsamps += xl;

          Eigen::ArrayXd ysamps;
          ysamps.setRandom(nsamps);
          ysamps += 1;
          ysamps /= 2;
          ysamps *= (yu - yl);
          ysamps += yl;

          if (lin) {
            xsamps.setLinSpaced(xl, xu);
            ysamps.setLinSpaced(yl, yu);
          }

          Eigen::VectorXd xy(2);
          Vector1<double> f;
          f.setZero();

          Utils::Timer Runtimer;
          Runtimer.start();

          double tmp = 0;
          for (int i = 0; i < nsamps; i++) {

            xy[0] = xsamps[i];
            xy[1] = ysamps[i];

            tabf.compute(xy, f);
            tmp += f[0] / double(i + 3);

            // fmt::print("{0:} \n",f[0]);

            f.setZero();
          }
          Runtimer.stop();
          double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
          fmt::print("Total Time: {0:} ms \n", tseconds * 1000);

          return tmp;
        });
}
