#include "FunctionDomains.h"

// void ASSET::DomainHolder<-1>::set_input_domain(int irr, const std::vector<DomainMatrix>& sub_domains) {
//     if (sub_domains.size() == 1) {
//         this->SubDomains = sub_domains[0];
//         return;
//     }
//     Eigen::VectorXi full(irr);
//     full.setZero();
//
//     for (auto& dmn : sub_domains) {
//         for (int i = 0; i < dmn.cols(); i++) {
//             full.segment(dmn(0, i), dmn(1, i)).setOnes();
//         }
//         if (full.sum() == irr) {
//             this->SubDomains.resize(2, 1);
//             this->SubDomains(0, 0) = 0;
//             this->SubDomains(1, 0) = irr;
//
//             return;
//         }
//     }
//     std::vector<std::array<int, 2>> sds;
//
//     bool find = true;
//     for (int i = 0; i < irr; i++) {
//         if (full[i] == 1) {
//             if (find) {
//                 sds.emplace_back(std::array<int, 2>{i, 1});
//                 find = false;
//             }
//             else
//                 sds.back()[1]++;
//         }
//         else
//             find = true;
//     }
//
//     this->SubDomains.resize(2, sds.size());
//     for (int i = 0; i < sds.size(); i++) {
//         this->SubDomains(0, i) = sds[i][0];
//         this->SubDomains(1, i) = sds[i][1];
//     }
// }