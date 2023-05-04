#pragma once
#include "OptimizationProblemBase.h"
#include "mkl.h"

namespace ASSET {


  struct Jet {


    static void print_beginning() {
      std::string jestr("          __  ______  ______\n"
                        "         / / / ____/ /_  __/\n"
                        "    __  / / / __/     / /   \n"
                        "   / /_/ / / /___    / /    \n"
                        "   \\____/ /_____/   /_/     \n\n");


      fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
      fmt::print(fmt::fg(fmt::color::crimson), jestr);
      fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
      fmt::print(": ");
      fmt::print(fmt::fg(fmt::color::royal_blue), "Jet");
      fmt::print("\n");
    }

    static void print_progress(
        int i, double tsec, int NumJobs, int NumConv, int NumAcc, int NumNoConv, int NumDiv) {
      double prog = 100 * double(i + 1) / double(NumJobs);
      int len = 76 * double(i + 1) / double(NumJobs);
      int wspace = 76 - len;
      double remtime = (tsec / double(i + 1)) * (NumJobs - i - 1);
      auto cyan = fmt::fg(fmt::color::cyan);
      auto green = fmt::fg(fmt::color::green);


      auto sminhrs = [cyan](double ts) {
        if (ts < 60.0) {
          fmt::print(cyan, "{0:>10.2f} s     \n", ts);
        } else if (ts < 3600.0) {
          fmt::print(cyan, "{0:>10.2f} min     \n", ts / 60.0);
        } else {
          fmt::print(cyan, "{0:>10.2f} hr     \n", ts / 3600.0);
        }
      };

      fmt::print("\n");


      fmt::print(" Remaining Time : ");
      sminhrs(remtime);
      fmt::print(" Elapsed Time   : ");
      sminhrs(tsec);
      fmt::print(" Progress       : ");
      fmt::print(cyan, "{0:>10.2f} %  \n", prog);
      fmt::print(" [");
      fmt::print(green, "{0:#^{1}}{0:.^{2}}", "", len, wspace);
      fmt::print("]");
      fmt::print("\n\n");
      fmt::print("  Completed        : ");
      fmt::print(cyan, "{0:>10}/{1:<10}   \n", (i + 1), NumJobs);

      fmt::print("    Optimal        : ");
      fmt::print(cyan, "{0:>10}/{1:<10}   \n", NumConv, NumJobs);
      fmt::print("    Acceptable     : ");
      fmt::print(cyan, "{0:>10}/{1:<10}   \n", NumAcc, NumJobs);
      fmt::print("    Not Converged  : ");
      fmt::print(cyan, "{0:>10}/{1:<10}   \n", NumNoConv, NumJobs);
      fmt::print("    Diverged       : ");
      fmt::print(cyan, "{0:>10}/{1:<10}   \n", NumDiv, NumJobs);

      if (i < (NumJobs - 1))
        fmt::print("\033[11F");
      else
        fmt::print("\n");
    };

    static void print_finished() {


      fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
      fmt::print(": ");
      fmt::print(fmt::fg(fmt::color::royal_blue), "Jet");
      fmt::print("\n");
      fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
    }


    ////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////// Map///////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    template<class T, class Args1, class Args2>
    static std::vector<std::shared_ptr<T>> map(
        const std::vector<std::function<std::shared_ptr<T>(Args1)>>& genfuncs,
        const std::vector<Args2>& args,
        const Eigen::VectorXi& genfidxes,
        int nt,
        bool verbose) {


      int NumJobs = args.size();
      int NumConv = 0;
      int NumAcc = 0;
      int NumNoConv = 0;
      int NumDiv = 0;

      std::vector<std::future<PSIOPT::ConvergenceFlags>> results(NumJobs);
      std::vector<std::shared_ptr<T>> optprobs(NumJobs);
      ctpl::ThreadPool pool(nt);
      Utils::Timer t;

      auto Job = [&](int threadid, int i) {
        mkl_set_num_threads_local(1);

        int gfidx = genfidxes[i];
        if constexpr (std::is_same_v<Args2, py::args>) {
          optprobs[i] = genfuncs[gfidx](*args[i]);
        } else {
          optprobs[i] = genfuncs[gfidx](args[i]);
        }

        return optprobs[i]->jet_run();
      };

      if (verbose)
        print_beginning();
      t.start();

      for (int i = 0; i < NumJobs; i++) {
        results[i] = pool.push(Job, i);
      }

      for (int i = 0; i < NumJobs; i++) {
        auto flag = results[i].get();
        if (verbose) {
          if (flag == PSIOPT::ConvergenceFlags::CONVERGED)
            NumConv++;
          if (flag == PSIOPT::ConvergenceFlags::ACCEPTABLE)
            NumAcc++;
          if (flag == PSIOPT::ConvergenceFlags::NOTCONVERGED)
            NumNoConv++;
          if (flag == PSIOPT::ConvergenceFlags::DIVERGING)
            NumDiv++;
          double tsec = double(t.count<std::chrono::microseconds>()) / 1000000.0;
          print_progress(i, tsec, NumJobs, NumConv, NumAcc, NumNoConv, NumDiv);
        }
      }
      if (verbose)
        print_finished();
      return optprobs;
    }


    template<class T, class Args1, class Args2>
    static std::vector<std::shared_ptr<T>> map(std::function<std::shared_ptr<T>(Args1)> genfunc,
                                               const std::vector<Args2>& args,
                                               int nt,
                                               bool verbose) {

      std::vector<std::function<std::shared_ptr<T>(Args1)>> genfuncs;
      genfuncs.push_back(genfunc);
      Eigen::VectorXi genfidxes(args.size());

      genfidxes.setConstant(0);

      return Jet::map(genfuncs, args, genfidxes, nt, verbose);
    }

    template<class T>
    static std::vector<std::shared_ptr<T>> map(const std::vector<std::shared_ptr<T>>& optprobs,
                                               int nt,
                                               bool verbose) {


      std::function<std::shared_ptr<T>(std::shared_ptr<T>)> genfunc = [](std::shared_ptr<T> optprob) {
        return optprob;
      };


      return Jet::map(genfunc, optprobs, nt, verbose);
    }
    ////////////////////////////////////////////////////////////////////////////////////

    static void Build(py::module& m) {

      auto obj = py::class_<Jet>(m, "Jet");

      obj.def_static(
          "map",
          [](const std::vector<std::shared_ptr<OptimizationProblemBase>>& optprobs, int nt) {
            return Jet::map(optprobs, nt, true);
          },
          py::call_guard<py::gil_scoped_release>());

      obj.def_static(
          "map",
          [](std::function<std::shared_ptr<OptimizationProblemBase>(py::detail::args_proxy)> genfun,
             const std::vector<py::args>& args,
             int nt) { return Jet::map(genfun, args, nt, true); },
          py::call_guard<py::gil_scoped_release>());


      obj.def_static(
          "map",
          [](const std::vector<std::shared_ptr<OptimizationProblemBase>>& optprobs, int nt, bool v) {
            return Jet::map(optprobs, nt, v);
          },
          py::call_guard<py::gil_scoped_release>());

      obj.def_static(
          "map",
          [](std::function<std::shared_ptr<OptimizationProblemBase>(py::detail::args_proxy)> genfun,
             const std::vector<py::args>& args,
             int nt,
             bool v) { return Jet::map(genfun, args, nt, v); },
          py::call_guard<py::gil_scoped_release>());
    }
  };


}  // namespace ASSET