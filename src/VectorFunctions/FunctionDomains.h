#pragma once

#include "pch.h"

namespace ASSET {

  template<int IR, int Start, int Size>
  struct SingleDomain {
    static const int DomainSize = IR;
    static constexpr std::array<std::array<int, 2>, 1> SubDomains = {std::array<int, 2> {Start, Size}};
    static const int start = Start;
    static const int size = Size;
  };

  template<int IR, class T, class... Ts>
  struct CompositeDomain {
    static const int DomainSize = IR;

    static constexpr bool contains_elem(int n) {
      std::tuple<T, Ts...> ts;
      bool t = false;
      const_tuple_for_each(ts, [&](const auto& func_i) {
        if (!t) {
          for (int i = 0; i < func_i.SubDomains.size(); i++) {
            int start = func_i.SubDomains[i][0];
            int size = func_i.SubDomains[i][1];
            if (n >= start && n < (start + size))
              t = true;
          }
        }
      });
      return t;
    }
    static constexpr int max_range(int start) {
      int maxx = 0;
      for (int i = start; i < IR; i++) {
        if (contains_elem(i))
          maxx++;
        else
          break;
      }
      return maxx;
    }
    static constexpr std::array<int, IR> dmn = {make_array<IR>(max_range)};
    static constexpr int sub_domains(std::array<int, IR> v) {
      int sr = 0;
      int i = 0;
      while (i < IR) {
        if (v[i] > 0) {
          sr += 1;
          i += v[i];
        } else {
          i += 1;
        }
      }
      return sr;
    }
    static const int NumSubDomains = sub_domains(dmn);
    static constexpr std::array<int, 2> calcSubDomains(int sd) {
      std::array<int, 2> v = {0, 0};
      int sr = 0;
      int i = 0;
      while (i < IR) {
        if (dmn[i] > 0) {
          if (sr == sd) {
            v[0] = i;
            v[1] = dmn[i];
            return v;
          }
          sr += 1;
          i += dmn[i];
        } else {
          i += 1;
        }
      }
      return v;
    }
    static constexpr std::array<std::array<int, 2>, NumSubDomains> SubDomains = {
        make_array<NumSubDomains>(calcSubDomains)};

    // CompositeDomain(int ir, T b, Ts... a) {}
    // CompositeDomain() = default;
  };


  template<class T, class... Ts>
  struct CompositeDomain<-1, T, Ts...> {

    static const int DomainSize = -1;
    static constexpr std::array<std::array<int, 2>, 1> SubDomains = {std::array<int, 2> {-1, -1}};
    static const int start = -1;
    static const int size = -1;

    // CompositeDomain(int ir, T b, Ts... a) {}
    // CompositeDomain() = default;
  };


  template<int IR>
  struct DomainHolder {
    DomainMatrix input_domain() const {
      DomainMatrix dmn(2, 1);
      dmn(0, 0) = 0;
      dmn(1, 0) = IR;
      return dmn;
    }
    void set_input_domain(int irr, const std::vector<DomainMatrix>& sub_domains) {};
  };

  template<>
  struct DomainHolder<-1> {
    DomainMatrix SubDomains;

    DomainMatrix input_domain() const {
      return SubDomains;
    }
    void set_input_domain(int irr, const std::vector<DomainMatrix>& sub_domains) {
      if (sub_domains.size() == 1) {
        this->SubDomains = sub_domains[0];
        return;
      }
      Eigen::VectorXi full(irr);
      full.setZero();

      for (auto& dmn: sub_domains) {
        for (int i = 0; i < dmn.cols(); i++) {
          full.segment(dmn(0, i), dmn(1, i)).setOnes();
        }
        if (full.sum() == irr) {
          this->SubDomains.resize(2, 1);
          this->SubDomains(0, 0) = 0;
          this->SubDomains(1, 0) = irr;

          return;
        }
      }
      std::vector<std::array<int, 2>> sds;

      bool find = true;
      for (int i = 0; i < irr; i++) {
        if (full[i] == 1) {
          if (find) {
            sds.emplace_back(std::array<int, 2> {i, 1});
            find = false;
          } else
            sds.back()[1]++;
        } else
          find = true;
      }

      this->SubDomains.resize(2, sds.size());
      for (int i = 0; i < sds.size(); i++) {
        this->SubDomains(0, i) = sds[i][0];
        this->SubDomains(1, i) = sds[i][1];
      }
    }
  };

}  // namespace ASSET
