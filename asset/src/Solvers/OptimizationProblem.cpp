#include <ASSET/Solvers/OptimizationProblem.h>

void ASSET::OptimizationProblem::transcribe() {
  this->nlp = std::make_shared<NonLinearProgram>(this->Threads);

  int numVars = this->ActiveVariables.size();

  if (numVars == 0) {
    fmt::print(fmt::fg(fmt::color::red),
               "Transcription Error!!!\n"
               "No variables provided to OptimizationProblem");
    throw std::invalid_argument("");
  }

  int numEqCons = 0;
  int numIqCons = 0;

  for (auto& func: this->userEqualities) {
    int irows = func.func.IRows();
    int orows = func.func.ORows();
    int numappl = func.indices.size();

    MatrixXi vindex(irows, numappl);
    MatrixXi cindex(orows, numappl);

    for (int i = 0; i < numappl; i++) {
      if (func.indices[i].maxCoeff() > numVars) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Variable indices out of bounds in equality constraint");
        throw std::invalid_argument("");
      }
      vindex.col(i) = func.indices[i];
      for (int j = 0; j < orows; j++) {
        cindex(j, i) = numEqCons;
        numEqCons++;
      }
    }

    this->nlp->EqualityConstraints.emplace_back(ConstraintFunction(func.func, vindex, cindex));

    ThreadingFlags ThreadMode =
        func.func.thread_safe() ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                                : ThreadingFlags::MainThread;

    this->nlp->EqualityConstraints.back().ThreadMode = ThreadMode;
  }

  for (auto& func: this->userInequalities) {
    int irows = func.func.IRows();
    int orows = func.func.ORows();
    int numappl = func.indices.size();

    MatrixXi vindex(irows, numappl);
    MatrixXi cindex(orows, numappl);

    for (int i = 0; i < numappl; i++) {
      if (func.indices[i].maxCoeff() > numVars) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Variable indices out of bounds in inequality constraint");
        throw std::invalid_argument("");
      }
      vindex.col(i) = func.indices[i];
      for (int j = 0; j < orows; j++) {
        cindex(j, i) = numIqCons;
        numIqCons++;
      }
    }

    this->nlp->InequalityConstraints.emplace_back(ConstraintFunction(func.func, vindex, cindex));

    ThreadingFlags ThreadMode =
        func.func.thread_safe() ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                                : ThreadingFlags::MainThread;

    this->nlp->InequalityConstraints.back().ThreadMode = ThreadMode;
  }

  for (auto& func: this->userObjectives) {
    int irows = func.func.IRows();
    int numappl = func.indices.size();

    MatrixXi vindex(irows, numappl);

    for (int i = 0; i < numappl; i++) {
      if (func.indices[i].maxCoeff() > numVars) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Variable indices out of bounds in inequality constraint");
        throw std::invalid_argument("");
      }
      vindex.col(i) = func.indices[i];
    }

    this->nlp->Objectives.emplace_back(ObjectiveFunction(func.func, vindex));

    ThreadingFlags ThreadMode =
        func.func.thread_safe() ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                                : ThreadingFlags::MainThread;

    this->nlp->Objectives.back().ThreadMode = ThreadMode;
  }

  this->nlp->make_NLP(numVars, numEqCons, numIqCons);
  this->optimizer->setNLP(this->nlp);

  //////DO NOT GET RID OF THIS!!!!!!//
  this->doTranscription = false;
  ////////////////////////////////////
}
