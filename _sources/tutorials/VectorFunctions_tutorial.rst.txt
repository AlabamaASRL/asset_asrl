Python: Vector Functions
========================

One of the goals of ASSET is to provide the ability to users to construct functions within Python that are able to be used by ASSET.
By doing this we can simplify a user's workflow, where the benefits of high speed C++ code can be combined with the ease of use Python provides.
However, the construction of these functions in Python requires that the user stick to the conventions that have been established for the ASSET Python interface.
This document will outline those conventions, as well as providing examples on what and what not to do when writing your Vector Functions in Python.

Fundamentals
############

For ASSET, a Vector Function is anything that takes a fixed sized input and returns a fixed sized output.

We will start with a simple example of a Vector Function.
First things first, ASSET needs to be imported into the environment.

.. code-block:: python

    import asset as ast

    vf = ast.VectorFunctions
    Args = vf.Arguments


The variable :code:`vf` exposes the Vector Functions of ASSET for use, while :code:`Args` will be neccessary for when we want to define the inputs to our Vector Function.
Now that we have ASSET and Vector Functions available we can start writing our Vector Function.

.. code-block:: python

    def VectorFunction_Example():
        args = Args(6)
        x = args.head(3)
        y = args.tail(3)

        output = x * y[0] + x

        return output


It is necessary for all functions in ASSET to know the size of the input to the Vector Function.
In this example we have decided that our function will take a **fixed** sized input of 6 arguments when the function is to be evaluated.
The first and last 3 arguments :code:`args` are seperated out into seperate vectors :code:`x` and :code:`y`.
A real example of a similar type operation would be on a vector containing 3D position and velocity information, where the first 3 arguments may be position and the last 3 are velocity.

We then define that the output of our function in Python is to return the :code:`x` vector multiplied by the first element of the :code:`y` vector and added with :code:`x`.

Something to note here is that one should take care to make their :code:`Vector Function` expressions as simple as possible.
What do we mean by this?
In the above example we write :code:`output = x*y[0] + x`, using the variable :code:`x` twice in the expression.
There is a cost to doing this, as each time a :code:`Vector Function` is used in an expression it is evaluated by ASSET in the backend.
This is the cost of being able to quickly and easily construct functions in Python that can be used by ASSET.
What can be done, then, to help mitigate the speed decrease by repeated expression use? One example would be to write :code:`output` in the above code as:

.. code-block:: python

    def VectorFunction_Example():
        args = Args(6)
        x = args.head(3)
        y = args.tail(3)

        output = x * (y[0] + 1)

        return output

See what we did there? We factored out :code:`x` leaving our :code:`Vector Function` expression only having to evaluate :code:`x` once, instead of twice.
While this may seem trivial, writing your expression in this manner will greatly increase the speed of ASSET evaluations.
Of course, it is not the end of the world if there is no way to simplify the expression further.
ASSET will still be able to use your :code:`Vector Function` fine, albeit slightly slower.

The way ASSET handles these operations is to treat both :code:`x` and :code:`y` as :code:`Vector Function` objects.
This means that both :code:`x` and :code:`y` will have all the methods of the :code:`Vector Function` interface available to them.
Additionally, :code:`output` is a Vector Function as it is the result of operations of a :code:`Vector Function` object.
:code:`Vector Functions` may have specific indices accessed through the bracket operator, as shown above.
The consequence of this is that ASSET has to perform an evaluation of each :code:`Vector Function` object in the expression, meaning :code:`x, y, and output` are all evaluated each time our :code:`VectorFunction_Example` is used.

Something else to mention is that *scalar* multiplication, addition, subtraction, and division with ASSET Vector Functions works as one would expect.
Vector operations on the other hand, such as dot products and cross products require one to use the Vector Function interface.
As an example, what would the above code have looked like if instead we wanted to return the dot product of the two vectors :code:`x` and :code:`y` squared?

.. code-block:: python

    def VectorFunction_Example():
        args = Args(6)
        x = args.head(3)
        y = args.tail(3)

        output_dot_product = vf.dot(x, y).squared()

        return output_dot_product

Notice that we do not take :code:`x` and :code:`y` and use the :code:`*` operator to perform a dot product.
Instead we use the :code:`vf.dot()` method from :code:`Vector Functions`.
Similarly, if we wanted to instead take the cross product of :code:`x` and :code:`y` we would use :code:`vf.cross(x, y)`.
Back to the above code sample, we then use the :code:`Vector Function .squared()` method to square the output of the dot product.
All of the available methods can be found in :ref:`Vector Functions`.

Turning a Vector Function into an ODE
#####################################

So instead of a toy example like we just showed, what would it require to construct a real world example, like the dynamics of a two-body solar sail problem?

.. math::

    \begin{equation}
    \hat{a} = -\frac{\mu \hat{r}}{|r^3|} + \hat{a_s} \\
    \hat{a_s} = \mu\beta(\hat{r}\cdot\hat{n})^{2}\frac{1}{|r^4|}\cdot\frac{\hat{n}}{|n^3|}
    \end{equation}

Above we have the familiar two-body solar sailing equations of motion, where :math:`\mu` is the gravitational parameter of the central body, :math:`r` is the position vector of the spacecraft, and :math:`\hat{a_s}` is the acceleration from the solar sail.
For :math:`\hat{a_s}`, :math:`\hat{n}` is the normal direction of the sail and :math:`\beta` is the sail optical parameter.

Now, that we have our equations, we need to write them in a form for use with ASSET.
First, we will create a function to compute the acceleration from the solar sail:

.. code-block:: python

    import asset as ast

    vf = ast.VectorFunctions
    oc = ast.OptimalControl

    Args = vf.Arguments
    Tmodes = oc.TranscriptionModes


    def SolarSail_Acc(r, n, scale):
        ndr2 = vf.dot(r, n).squared()
        acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        return acc

Where the :code:`scale` is going to be :math:`\beta\cdot\mu`.
Just like that we have defined our solar sail acceleration, with the use of a few commonly used :code:`Vector Function` methods, such as :code:`n.normalized_power3()`, which returns the :math:`\hat{n}` normalized by its third power.
Just as an example, perhaps for a different problem, one could use :code:`n.normalized_power(a)`, where :math:`a` is some arbitrary power to take the norm of :math:`\hat{n}` with.
Once again, all these methods can be in :ref:`Vector Functions`.
Notice that we added a few more things to include from ASSET.
We will use those at the end of constructing our :code:`Vector Function` expression.

The last step to defining our :code:`Vector Function` expression in Python is to write a function to compute the gravitational acceleration and add it up with the solar sail acceleration.
This is where we will define our total number of arguments into the problem, like we did in the first few code examples.
This will also be the function we use later on to construct an :code:`ODE` expression for ASSET.

.. code-block:: python

    def Full_TwoBody_SolarSail_Model(mu, beta):
        args = Args(10)
        r = args.head_3()
        v = args.segment_3(3)
        n = args.tail_3()
        acc = -mu * r.normalized_power3() + SolarSail_Acc(r, n, beta * mu)
        return vf.Stack([v, acc])

We define our total number of inputs (args) to the function as 10.
Now wait a minute, shouldn't that only be 9 since there are 3 components for each :math:`\hat{r}` (position vector), :math:`\hat{v}` (velocity vector), and :math:`\hat{n}` (normal direction of the sail relative to the Sun)? That gives us 9, but for ASSET to be able to
use this function, we also have to give it the time of the state, so there is an implied 10th variable of time that must be in the state vector.
There are a few other conventions for constructing models specifically for ASSET, so take a look at :ref:`ODE` to get a better idea for that.
The focus of our discussion here is simply how to write any :code:`Vector Function` for ASSET, not just ODEs.
A :code:`Vector Function` could be an ODE model, but it could also be constraint equations, objective equations, or any number of other functions.
Remember, a :code:`Vector Function` to ASSET is just a function that takes a fixed sized input, and returns a fixed sized output.

Back to the problem at hand, we divide those :code:`args` into their respective vectors :code:`r`, :code:`v`, and :code:`n`.

Here :code:`r` takes the first 3 args, :code:`v` takes the next 3 as a segment of :code:`args` from the index 3 of :code:`args` and grabs the next three inputs.
See :code:`asset.VectorFunctions.segment()` for a better understanding.
Lastly, :code:`n` takes the last 3 elements of :code:`args` with the :code:`.tail()` method of :code:`VectorFunctions`.

We can then create our full acceleration from both the solar sail and gravity as normal, where :code:`acc` will be a :math:`3\times 1` :code:`VectorFunction` that requires a total of 6 input arguments for the :code:`SolarSail_Acc` function.
The last thing we need to do to fully construct our equations of motion is to stack the time derivative of position (velocity), with our accelerations.
To combine our desired outputs (velocity and acceleraton), we use the :code:`vf.Stack([vec1, vec2])`, which will take :code:`vec1` and place it on top of :code:`vec2`, in this case creating an output :code:`VectorFunction` of length 6, 3 from the velocity and 3 from the acceleration.



Now we have the full :code:`VectorFunction` for our two-body solar sail problem! The last thing we have to do, before you should head over to the :ref:`Python: Phase Tutorial`, is to map this :code:`VectorFunction` to an ASSET ODE type and assign it an integrator.

.. code-block:: python

    Two_Body_SolarSail_ODE = oc.ode_x_u.ode(Full_TwoBody_SolarSail_Model(1, 0.01), 6, 3)

    phase = oc.ode_x_u.phase(Two_Body_SolarSail_ODE, Tmodes.LGL3)

    integrator = Two_Body_SolarSail_ODE.integrator(0.01)

Using the :code:`OptimalControl` section of ASSET, we can use our function to construct an ODE, allowing ASSET to use this function as a dynamical model for solving and optimization problems.
Here we use :code:`oc.ode_x_u.ode(Function(), NumberStateVariables, NumberControlVariables)` which takes as inputs, the function we want to convert to an ODE (along with an inputs to that function), as well as the size of the state variables and control variables necessary for that function to evaluate.
Once again, we see here that we only added 9 variables, so where is that 10th variable, time, that ASSET needs?
ASSET will **always** assume that your time variable is implicitly a part of your full state vector.
The time variable **must always** come after your state variabeles, and before your control and parameter variables.
For a better rundown of this convention, please see the sections for ODEs, and the ASSET Phase Interface.
Back to the problem at hand, we have created our ode :code:`Two_Body_SolarSail_ODE`, which is then handed to the ASSET optimal control method :code:`oc.ode_x_u.phase(ode, Tmodes.TranscriptionModes)` which takes the ODE function, as well as an enumerator from :code:`TranscriptionModes` to assign a transcription type for this phase.
In the above problem we are using Legendre-Gauss-Lobatto 3rd order collocation.
All available transcription modes can be found in :code:`OptimalControl.TranscriptionModes`.
Lastly, we give our :code:`Two_Body_SolarSail_ODE` an integrator (which for right now is always Runge-Kutta 4th order), where :code:`Two_Body_SolarSail_ODE.integrator(.01)` assigns the integrator a step size of :math:`.01`.

The full code for this is:

.. code-block:: python

    import asset as ast

    vf = ast.VectorFunctions
    oc = ast.OptimalControl

    Args = vf.Arguments
    Tmodes = oc.TranscriptionModes


    def SolarSail_Acc(r, n, scale):
        ndr2 = vf.dot(r, n).squared()
        acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        return acc


    def Full_TwoBody_SolarSail_Model(mu, beta):
        args = Args(10)
        r = args.head_3()
        v = args.segment_3(3)
        n = args.tail_3()
        acc = -mu * r.normalized_power3() + SolarSail_Acc(r, n, beta * mu)
        return vf.Stack([v, acc])


    Two_Body_SolarSail_ODE = oc.ode_x_u.ode(Full_TwoBody_SolarSail_Model(1, 0.01), 6, 3)

    phase = Two_Body_SolarSail_ODE.phase(Two_Body_SolarSail_ODE, Tmodes.LGL3)

    integrator = Two_Body_SolarSail_ODE.integrator(0.01)

Thats all the code required to construct a Vector Function to evaluate the two-body solar sailing equations of motion!

While the above code is a perfectly adequate way to construct an ode (ie: writing a vector function and passing to an ode object), we can
also implement the same behavior by simply extending the ASSET ode object directly in python as shown below.

.. code-block:: python

    import asset as ast

    vf = ast.VectorFunctions
    oc = ast.OptimalControl

    Args = vf.Arguments
    Tmodes = oc.TranscriptionModes


    def SolarSail_Acc(r, n, scale):
        ndr2 = vf.dot(r, n).squared()
        acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        return acc

    class Full_TwoBody_SolarSail_Model(oc.ode_x_u.ode):
        def __init__(self,mu,beta):
            Xvars = 6
            Uvars = 3
            Ivars = Xvars + 1 + Uvars
            #############################
            args = Args(Ivars)
            r = args.head_3()
            v = args.segment_3(3)
            n = args.tail_3()
            acc = -mu * r.normalized_power3() + SolarSail_Acc(r, n, beta * mu)
            odeeq =  vf.Stack([v, acc])
            super().__init__(odeeq,Xvars,Uvars)


    Two_Body_SolarSail_ODE = Full_TwoBody_SolarSail_Model(1, 0.01)

    phase = Two_Body_SolarSail_ODE.phase(Tmodes.LGL3)

    integrator = Two_Body_SolarSail_ODE.integrator(0.01)

We simply inherit from the ASSET dynamic ode object, :code:`oc.ode_x_u.ode`, write our vector function in the constructor and then forward it to the ode
type along with size information at the end of the call. The new model can then be constructed directly.


Now, its time to head over to tutorials provided for Phase and ODE to learn more about how to put these functions into action for optimization problems.
