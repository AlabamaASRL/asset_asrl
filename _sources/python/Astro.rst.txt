ASSET Astro Library
===================

Included with the ASSET install are a subset of dynamical models that are available for use through the Python interface. 
These models include standard two-body, circular restricted three body problem (CR3BP), ephemeris, and low-thrust models.
The Python implementation of these files can be found in asset/src/asset_asrl/Astro.
Below we will demonstrate some of the sample models that can be found.

.. warning::

	The layout, inputs, outputs, and very existence of these models and the Astro library itself **will** change during the continued development of ASSET.
	**While these are available to use, use them at your own risk.** 



Example Models
--------------
These simply demonstrate how to set setup a given ODE. For more information on how to use these ODEs with ASSET see :ref:`Tutorials`.
All of the constants used are assumed to be meters and seconds. This convention **needs** to be maintained for compatibility with the internals of the models.
With the given inputs to the frames (:math:`\mu` and :math:`l^*`, which is the gravity parameter and characteristic length of the system respectively), other characteristic values
(such as :math:`v^*`, the characteristic non-dimensional velocity) are stored in each model. Failure to be conistent with unit scaling for accelerations (such as a low thrust acceleration)
**will** result in incorrect models. It is highly recommended to use the constants defined in Astro.Constants as a result.

.. note::
	While all of these models accept dimensional values pertaining to the masses and distances of the system, **they are all non-dimensionalized inside of the models.**
	Thus it is necessary that all inputs to the models (ODE variables, controls, etc.) are appropriately non-dimensionalized **by the user** before proceeding.

.. warning::
	
	When using any model that contains Spice data, be sure that the start and stop Julian days for the Spice data range includes the time range you wish to model for.
	Attempting to integrate or access data outside of these bounds will produce erroneous results.

.. note::
	 
	When using any of the provided low-thrust models, these models are all assumed to be massless models. That is, none of these low-thrust models
	consider effect of  the expenditure of propellant during their course of use. The thrusters are assumed to use little enough fuel as for this to not be a consideration.

.. warning::

	These models are provided as examples to use in constructing your own models. While you may use these, use them at your own risk. We highly reccomend writing your own models
	as demonstrated in :ref:`ODE Tutorial`.

Two-Body Ballistic
^^^^^^^^^^^^^^^^^^

.. code-block:: python
	
	import asset_asrl as ast
	from asset_asrl.Astro.AstroModels import TwoBody
	import asset_asrl.Astro.Constants as c

	#BE SURE YOUR UNITS ARE SCALED CORRECTLY

	mu = c.MuEarth #gravity parameter of Earth in m^3/s^2
	lstar = c.RadiusEarth #radius of Earth, in meters

	ode = TwoBody(mu, lstar)

Two-Body Low-Thrust
^^^^^^^^^^^^^^^^^^

.. code-block:: python
	
	import asset_asrl as ast
	from asset_asrl.Astro.AstroModels import TwoBody_LT
	import asset_asrl.Astro.Constants as c
	from asset_asrl.Astro.Extensions.ThrusterModels import LowThrustAcc

	#BE SURE YOUR UNITS ARE SCALED CORRECTLY

	mu = c.MuEarth #gravity parameter of Earth in m^3/s^2
	lstar = c.RadiusEarth #radius of Earth, in meters
	ltacc = .01 #low thrust acceleration in m/s^2

	#set NonDim_LTacc to false since we are using a dimensional acceleration
	ltmodel = LowThrustAcc(NonDim_LTacc = False, LTacc = ltacc)
	ode = TwoBody_LT(mu, lstar, thruster = ltmodel)

CR3BP Ballistic
^^^^^^^^^^^^^^^

.. code-block:: python
	
	import asset_asrl as ast
	from asset_asrl.Astro.AstroModels import CR3BP
	import asset_asrl.Astro.Constants as c

	#BE SURE YOUR UNITS ARE SCALED CORRECTLY

	mu1 = c.MuEarth #gravity parameter of Earth in m^3/s^2
	mu2 = c.MuMoon #gravity parameter of Moon in m^3/s^2
	lstar = 385000*1000.0 #Characteristic distance between Earth and Moon in meters

	ode = CR3BP(mu1, mu2, lstar)


CR3BP Low-Thrust
^^^^^^^^^^^^^^^^

.. code-block:: python
	
	import asset_asrl as ast
	from asset_asrl.Astro.AstroModels import CR3BP_LT
	from asset_asrl.Astro.Extensions.ThrusterModels import LowThrustAcc
	import asset_asrl.Astro.Constants as c

	#BE SURE YOUR UNITS ARE SCALED CORRECTLY

	mu1 = c.MuEarth #gravity parameter of Earth in m^3/s^2
	mu2 = c.MuMoon #gravity parameter of Moon in m^3/s^2
	lstar = 385000*1000.0 #Characteristic distance between Earth and Moon in meters

	#Create the low-thrust model
	ltacc = .01 #low thrust acceleration in m/s^2
	#set NonDim_LTacc to false since we are using a dimensional acceleration
	ltmodel = LowThrustAcc(NonDim_LTacc = False, LTacc = ltacc)

	ode = CR3BP_LT(mu1, mu2, lstar, thruster = ltmodel)

Ephemeris Models
----------------
Below are examples of ephemeris models implemented in the Astro library. You will need to have a working install of spiceypy, as well as Spice ephemeris kernels, along with all necessary items to use spiceypy.
Your Spice Kernel will need to have information regarding all bodies you wish to add.

.. warning:::

	Currently these models are entirely dependent on constants that are obtained from Astro.Constants.
	While it is possible to use different :math:`\mu` values for the bodies in the system, or names other than those provided in Astro.Constants,
	failure to provide the correct names and :math:`\mu` values will lead to erroneous results, or failure to build the model.
	**As such we recommend only using those bodies that are included in Astro.Constants. Use other bodies and :math:`\mu` values at your own risk.**

EPPR Ballistic
^^^^^^^^^^^^^^

.. code-block:: python
	
	import asset_asrl as ast
	from asset_asrl.Astro.AstroModels import EPPR, EPPRFrame
	import asset_asrl.Astro.Constants as c
	import spiceypy as sppy

	#For example, load de432s.
	sppy.spiceypy.furnsh("directory_to_your_kernels/de432s.bsp")

	#BE SURE YOUR UNITS ARE SCALED CORRECTLY

	#Julian start day of Spice data
	JD0 = 2451545.0
	#Julian end day of Spice data
	JDF = JD0 + 5.0*365.0   
	#Number of points to use for Spice data
	N = 4000

	SpiceFrame = 'J2000'

	#Primaries of the system
	P1 = "SUN"
	P2 = "EARTH"
	EFrame = EPPRFrame(P1, c.MuSun, P2, c.MuEarth, c.AU, JD0, JDF, N = N, SpiceFrame = SpiceFrame)

	#These are additional bodies besides P1 and P2 to include
	Bodies = ["MOON", "JUPITER BARYCENTER", "VENUS"]
	#Add the bodies to the ODE
	EFrame.AddSpiceBodies(Bodies, N = N)

	ode = EPPR(EFrame)

EPPR Low-Thrust
^^^^^^^^^^^^^^^

.. code-block:: python
	
	import asset_asrl as ast
	from asset_asrl.Astro.AstroModels import EPPR_LT, EPPRFrame
	from asset_asrl.Astro.Extensions.ThrusterModels import LowThrustAcc
	import asset_asrl.Astro.Constants as c
	import spiceypy as sppy

	#For example, load de432s.
	sppy.spiceypy.furnsh("directory_to_your_kernels/de432s.bsp")

	#BE SURE YOUR UNITS ARE SCALED CORRECTLY

	#Julian start day of Spice data
	JD0 = 2451545.0
	#Julian end day of Spice data
	JDF = JD0 + 5.0*365.0   
	#Number of points to use for Spice data
	N = 4000

	SpiceFrame = 'J2000'

	#Primaries of the system
	P1 = "SUN"
	P2 = "EARTH"
	EFrame = EPPRFrame(P1, c.MuSun, P2, c.MuEarth, c.AU, JD0, JDF, N = N, SpiceFrame = SpiceFrame)

	#These are additional bodies besides P1 and P2 to include
	Bodies = ["MOON", "JUPITER BARYCENTER", "VENUS"]
	#Add the bodies to the ODE
	EFrame.AddSpiceBodies(Bodies, N = N)

	#Create the low-thrust model
	ltacc = .01 #low thrust acceleration in m/s^2
	#set NonDim_LTacc to false since we are using a dimensional acceleration
	ltmodel = LowThrustAcc(NonDim_LTacc = False, LTacc = ltacc)

	ode = EPPR_LT(EFrame, thruster = ltmodel)


NBody Ballistic
^^^^^^^^^^^^^^^

.. code-block:: python
	
	import asset_asrl as ast
	from asset_asrl.Astro.AstroModels import NBody, NBodyFrame
	import asset_asrl.Astro.Constants as c
	import spiceypy as sppy

	#For example, load de432s.
	sppy.spiceypy.furnsh("directory_to_your_kernels/de432s.bsp")

	#BE SURE YOUR UNITS ARE SCALED CORRECTLY

	#Julian start day of Spice data
	JD0 = 2451545.0
	#Julian end day of Spice data
	JDF = JD0 + 5.0*365.0   
	#Number of points to use for Spice data
	N = 4000

	SpiceFrame = 'J2000'

	#Primaries of the system
	P1 = "SUN"
	NBFrame = NBodyFrame(P1, c.MuSun, c.AU, JD0, JDF, N = N, SpiceFrame = SpiceFrame)

	#These are additional bodies besides P1 to include
	Bodies = ["EARTH", "MOON", "JUPITER BARYCENTER", "VENUS"]
	#Add the bodies to the ODE
	NBFrame.AddSpiceBodies(Bodies, N = N)

	ode = NBody(NBFrame)


NBody Low-Thrust
^^^^^^^^^^^^^^^^

.. code-block:: python
	
	import asset_asrl as ast
	from asset_asrl.Astro.AstroModels import NBody_LT, NBodyFrame
	from asset_asrl.Astro.Extensions.ThrusterModels import LowThrustAcc
	import asset_asrl.Astro.Constants as c
	import spiceypy as sppy

	#For example, load de432s.
	sppy.spiceypy.furnsh("directory_to_your_kernels/de432s.bsp")

	#BE SURE YOUR UNITS ARE SCALED CORRECTLY

	#Julian start day of Spice data
	JD0 = 2451545.0
	#Julian end day of Spice data
	JDF = JD0 + 5.0*365.0   
	#Number of points to use for Spice data
	N = 4000

	SpiceFrame = 'J2000'

	#Primaries of the system
	P1 = "SUN"
	NBFrame = NBodyFrame(P1, c.MuSun, c.AU, JD0, JDF, N = N, SpiceFrame = SpiceFrame)

	#These are additional bodies besides P1 to include
	Bodies = ["EARTH", "MOON", "JUPITER BARYCENTER", "VENUS"]
	#Add the bodies to the ODE
	NBFrame.AddSpiceBodies(Bodies, N = N)

	#Create the low-thrust model
	ltacc = .01 #low thrust acceleration in m/s^2
	#set NonDim_LTacc to false since we are using a dimensional acceleration
	ltmodel = LowThrustAcc(NonDim_LTacc = False, LTacc = ltacc)

	ode = NBody_LT(NBFrame, thruster = ltmodel)