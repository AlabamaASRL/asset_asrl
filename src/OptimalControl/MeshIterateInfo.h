#pragma once
#include "pch.h"

namespace ASSET {

  struct MeshIterateInfo {

    bool converged = false;
    int numsegs;
    int up_numsegs;
    double tol;


    double avg_error;
    double gmean_error;
    double max_error;
    double global_error = -1;

    Eigen::VectorXd times;
    Eigen::VectorXd error;
    Eigen::VectorXd distribution;
    Eigen::VectorXd distintegral;


    MeshIterateInfo() {
    }

    MeshIterateInfo(int numsegs,
                    double tol,
                    const Eigen::VectorXd& times,
                    const Eigen::VectorXd& error,
                    const Eigen::VectorXd& distribution)
        : numsegs(numsegs),
          up_numsegs(numsegs),
          tol(tol),
          times(times),
          error(error),
          distribution(distribution) {

      int n = times.size();
      ////////////////////////////////////////////////////

      Eigen::VectorXd hs = this->times.tail(n - 1) - this->times.head(n - 1);
      this->max_error = this->error.maxCoeff();
      this->avg_error = (this->error.head(n - 1).cwiseProduct(hs)).sum();

      this->gmean_error = std::exp((std::log(this->max_error) + std::log(this->avg_error)) / 2.0);

      // this->gmean_error = std::exp((this->error.head(n - 1).array().log()*hs.array()).sum());


      ////////////////////////////////////////////////////
      this->distintegral.resize(this->times.size());
      this->distintegral[0] = 0;

      for (int i = 0; i < n - 1; i++) {
        this->distintegral[i + 1] =
            this->distintegral[i] + (this->distribution[i]) * (this->times[i + 1] - this->times[i]);
      }

      this->distintegral = this->distintegral / this->distintegral[n - 1];
      /////////////////////////////////////////////////////
    }

    Eigen::VectorXd calc_bins(int nbins) {

      Eigen::VectorXd bins;
      bins.setLinSpaced(nbins + 1, 0.0, 1.0);
      int elem = 0;
      for (int i = 1; i < nbins; i++) {
        double di = double(i) / double(nbins);
        auto it = std::upper_bound(this->distintegral.cbegin() + elem, this->distintegral.cend(), di);
        elem = int(it - this->distintegral.cbegin()) - 1;

        double t0 = this->times[elem];
        double t1 = this->times[elem + 1];
        double d0 = this->distintegral[elem];
        double d1 = this->distintegral[elem + 1];
        double slope = (d1 - d0) / (t1 - t0);
        bins[i] = (di - d0) / slope + t0;
      }
      return bins;
    }

    static void print_header(int iter) {

      fmt::print("{0:=^{1}}\n", "", 52);
      fmt::print("Mesh Iteration: {0:}\n", iter);
      fmt::print("{0:=^{1}}\n", "", 52);
      fmt::print("|Phase|#Segs| Max Err | Avg Err | EtE Err |Up #Segs|\n");
    }

    void print(int phasenum) {

      auto level1 = std::log(tol);
      auto level3 = std::log(tol * 1000);
      auto level5 = std::log(tol * 100000.0);
      auto level2 = (level1 + level3) / 2.0;
      auto level4 = (level3 + level5) / 2.0;


      auto calccolor = [&](double val) {
        auto logval = std::log(val);
        fmt::color c;

        if (logval < level1)
          c = fmt::color::lime_green;
        else if (logval < level2)
          c = fmt::color::yellow;
        else if (logval < level3)
          c = fmt::color::orange;
        else if (logval < level4)
          c = fmt::color::red;
        else
          c = fmt::color::dark_red;
        return fmt::fg(c);
      };


      fmt::print("|{:<5}|", phasenum);
      fmt::print("{:<5}|", this->numsegs);
      fmt::print(calccolor(this->max_error), "{:>9.3e}", this->max_error);
      fmt::print("|");
      fmt::print(calccolor(this->avg_error), "{:>9.3e}", this->avg_error);
      fmt::print("|");
      if (this->global_error > 0) {
        fmt::print(calccolor(this->global_error), "{:>9.3e}", this->global_error);
        fmt::print("|");

      } else {
        fmt::print("   N/A   |");
      }
      fmt::print("{:<8}|\n", this->up_numsegs);
    }


    static void Build(py::module& m) {
      auto obj = py::class_<MeshIterateInfo>(m, "MeshIterateInfo");

      obj.def_readonly("times", &MeshIterateInfo::times);
      obj.def_readonly("error", &MeshIterateInfo::error);
      obj.def_readonly("distribution", &MeshIterateInfo::distribution);
      obj.def_readonly("distintegral", &MeshIterateInfo::distintegral);
      obj.def_readonly("avg_error", &MeshIterateInfo::avg_error);
      obj.def_readonly("max_error", &MeshIterateInfo::max_error);
      obj.def_readonly("numsegs", &MeshIterateInfo::numsegs);
      obj.def_readonly("converged", &MeshIterateInfo::converged);
    }
  };


}  // namespace ASSET