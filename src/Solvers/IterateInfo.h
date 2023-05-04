#pragma once


namespace ASSET {

  struct IterateInfo {


    int iter = 0;

    double Mu = 0;
    double PrimObj = 0;
    double BarrObj = 0;
    double KKTInf = 0;
    double BarrInf = 0;
    double EConInf = 0;
    double IConInf = 0;

    double PenPar1 = 0.0;
    double PenPar2 = 0.0;

    int LSiters = 0;
    double alphaP = 1.0;
    double alphaD = 1.0;
    double alphaT = 1.0;

    double Hpert = 0;
    int Hfacs = 0;

    double KKTNormErr = 0;
    double BarrNormErr = 0;
    double EConNormErr = 0;
    double IConNormErr = 0;
    double AllConNormErr = 0;

    int PPivots = 0;
    double MaxEMult = 0;
    double MaxIMult = 0;
    double MeritVal = 0.0;
  };

}  // namespace ASSET
