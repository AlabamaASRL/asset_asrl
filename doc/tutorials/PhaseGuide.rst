Optimal Control Phase Tutorial
==============================


.. list-table:: List of Scalar Math Functions
   :widths: 15 25 20 20
   :header-rows: 1

   * - Transcription
     - Description
     - Integral Method
     - Control Representation
   * - :code:`'LGL3'`
     - Third Order Legendre Guass Lobatto Collocation
     - Trapezoidal Rule
     - Piecewise Linear, Piecewise Constant (BlockConstant)



.. list-table:: List of Scalar Math Functions
   :widths: 20 50 30
   :header-rows: 1

   * - Phase Region
     - Description
     - Implied Input Size
   * - Front
     - First full-state in the phase
     - 2*len(XtUVars) + len(ODEPvars) + len(StatPvars)