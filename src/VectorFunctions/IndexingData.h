/*
File Name: IndexingData.h

File Description: This file implements the struct SolverIndexingData which holds all meta data neccessary
for an asset vector function to be used as a constraint or objective inside of psiopt. It is coupled with a
function by the interface classes ConstraintFunction and ObjectiveFunction.


////////////////////////////////////////////////////////////////////////////////

Original File Developer : James B. Pezent - jbpezent - jbpezent@crimson.ua.edu

Current File Maintainers:
    1. James B. Pezent - jbpezent         - jbpezent@crimson.ua.edu
    2. Full Name       - GitHub User Name - Current Email
    3. ....


Usage of this source code is governed by the license found
in the LICENSE file in ASSET's top level directory.

*/

#pragma once
#include "FunctionalFlags.h"
#include "pch.h"

namespace ASSET {

  struct SolverIndexingData {
    using MatrixXi = Eigen::MatrixXi;
    using VectorXi = Eigen::VectorXi;

    int input_size = 0;
    int output_size = 0;
    int num_funcappl = 0;

    bool vindex_init = false;
    bool cindex_init = false;
    bool unique_constraints = true;

    /// <summary>
    /// Matrix whose columns contains the ordered indices of the variables
    /// that will be forwarded to a constraint or objective function
    /// </summary>
    MatrixXi Vindex;

    /// <summary>
    /// Matrix whose columns containt the ordered constraint output indices for the correspoding
    /// column in Vindex. This matrix is empy for objective functions.
    /// </summary>
    MatrixXi Cindex;

    /// <summary>
    /// Each element indicates whether the correspoding indices in Vindex are sorted
    /// and contigous (ie: 10,11,12...)
    /// </summary>
    Eigen::Matrix<ParsedIOFlags, -1, 1> VindexContinuity;

    /// <summary>
    /// Each element indicates whether the correspoding indices in Cindex are sorted
    /// and contigous (ie: 10,11,12...)
    /// </summary>
    Eigen::Matrix<ParsedIOFlags, -1, 1> CindexContinuity;

    /// <summary>
    /// Holds the index of the start of the region of memory allocated by Psiopt to sum the constraint
    /// output of the ith call of a constraint function.
    /// </summary>
    VectorXi InnerConstraintStarts;

    /// <summary>
    /// Holds the index of the start of the region of memory allocated by Psiopt to sum the gradient
    /// output of the ith call of a constraint or objective function.
    /// </summary>
    VectorXi InnerGradientStarts;

    /// <summary>
    /// Holds the index of the start of the region of memory allocated by Psiopt to store the locations
    /// where the derivatives of the ith call to a function should be summed into the global KKT matrix.
    /// </summary>
    VectorXi InnerKKTStarts;


    SolverIndexingData() {
    }
    SolverIndexingData(int irr, int orr, const MatrixXi& vindex, const MatrixXi& cindex)
        : input_size(irr), output_size(orr) {
      this->setVindexCindex(vindex, cindex);
    }

    SolverIndexingData(int irr, const MatrixXi& vindex) : input_size(irr), output_size(1) {
      this->setVindex(vindex);
    }

    void getGradientSpace(EigenRef<VectorXi> GXrows, int& freeloc) {
      this->InnerGradientStarts.resize(this->NumAppl());
      for (int V = 0; V < this->NumAppl(); V++) {
        this->InnerGradientStarts[V] = freeloc;
        for (int i = 0; i < this->input_size; i++) {
          GXrows[freeloc] = this->VLoc(i, V);
          freeloc++;
        }
      }
    }
    void getConstraintSpace(EigenRef<VectorXi> FXrows, int& freeloc) {
      this->InnerConstraintStarts.resize(this->NumAppl());
      for (int V = 0; V < this->NumAppl(); V++) {
        this->InnerConstraintStarts[V] = freeloc;
        for (int j = 0; j < this->output_size; j++) {
          FXrows[freeloc] = this->CLoc(j, V);
          freeloc++;
        }
      }
    }

    std::vector<SolverIndexingData> thread_split(int Threads) const {
      std::vector<SolverIndexingData> split;
      split.reserve(Threads);

      int cols = this->num_funcappl;
      int colpThr = cols / Threads;
      int rempThr = cols % Threads;

      VectorXi perThr = VectorXi::Constant(Threads, colpThr);
      perThr.head(rempThr) += VectorXi::Constant(rempThr, 1);
      int start = 0;
      int range;
      if (colpThr > 0)
        range = Threads;
      else
        range = rempThr;
      for (int i = 0; i < range; i++) {
        if (this->cindex_init) {
          split.emplace_back(SolverIndexingData(this->input_size,
                                                this->output_size,
                                                this->Vindex.middleCols(start, perThr[i]),
                                                this->Cindex.middleCols(start, perThr[i])));
        } else {
          split.emplace_back(SolverIndexingData(this->input_size, this->Vindex.middleCols(start, perThr[i])));
        }
        split.back().unique_constraints = this->unique_constraints;
        start += perThr[i];
      }
      return split;
    }

    void setVindex(const MatrixXi& vt) {
      this->Vindex = vt;
      this->vindex_init = true;
      this->num_funcappl = this->Vindex.cols();
      this->VindexContinuity.resize(this->Vindex.cols());
      for (int i = 0; i < this->Vindex.cols(); i++) {
        this->VindexContinuity[i] = this->checkContinuity(this->Vindex.col(i));
      }
    }
    void setCindex(const MatrixXi& ct) {
      this->Cindex = ct;
      this->CindexContinuity.resize(this->Cindex.cols());
      this->cindex_init = true;

      for (int i = 0; i < this->Cindex.cols(); i++) {
        this->CindexContinuity[i] = this->checkContinuity(this->Cindex.col(i));
      }
    }
    const MatrixXi& getVindex() const {
      return this->Vindex;
    }
    const MatrixXi& getCindex() const {
      return this->Cindex;
    }
    void setVindexCindex(const MatrixXi& vt, const MatrixXi& ct) {
      this->setVindex(vt);
      this->setCindex(ct);
    }
    void pushVindex(int push) {
      for (int i = 0; i < this->Vindex.size(); i++) {
        this->Vindex(i) += push;
      }
    }
    void pushCindex(int push) {
      for (int i = 0; i < Cindex.size(); i++) {
        this->Cindex(i) += push;
      }
    }
    void pushVindexCindex(int vpush, int cpush) {
      this->pushVindex(vpush);
      this->pushCindex(cpush);
    }

    inline int NumAppl() const {
      return this->num_funcappl;
    }
    inline int CLoc(int loc, int col) const {
      return this->Cindex(loc, col);
    }
    inline int VLoc(int loc, int col) const {
      return this->Vindex(loc, col);
    }

    static ParsedIOFlags checkContinuity(const Eigen::VectorXi& ix) {
      int s = 0;
      for (int i = 0; i < (ix.size() - 1); i++) {
        s = ix[i + 1] - ix[i] - 1;
        if (s != 0)
          return ParsedIOFlags::NotContiguous;
      }
      return ParsedIOFlags::Contiguous;
    }
  };

}  // namespace ASSET
