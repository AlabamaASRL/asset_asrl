Example 6: 6-DOF Time Optimal Docking
=====================================

.. math::

    \frac{d r }{dt} & = \vec{v} = [\dot{x},\dot{y},\dot{z}]

    \ddot{x} &= 2 n \dot{y} + 3 n^2 x + \frac{T_x^G}{m}

    \ddot{y} &= -2 n \dot{x} +  \frac{T_y^G}{m}

    \ddot{z} &= - n^2 z + \frac{T_z^G}{m}


    \frac{d \mathbf{q} }{dt}  &= \frac{1}{2} \mathbf{q} \mathbf{q}_{\vec{\omega}}

    \frac{d \vec{\omega} }{dt} &= I_1^{-1}\left(\vec{\tau} + (I_1\vec{\omega})\times\vec{\omega} \right)

    \frac{d \mathbf{p} }{dt}  &= \frac{1}{2} \mathbf{p} \mathbf{p}_{\vec{\phi}}

    \frac{d \vec{\phi} }{dt} &= I_2^{-1}\left((I_2\vec{\phi})\times\vec{\phi} \right)


.. math::
    
    \vec{T}^G = \mathbf{q} \vec{T} \mathbf{q}^-1


.. math::
    
    \vec{r}_q + \vec{r} -\vec{r}_p  = \vec{0}

    \dot{\vec{r}}_q + \dot{\vec{r}}  -  \dot{\vec{r}}_p = \vec{0} 

    \vec{r}_q       = \mathbf{q} \vec{D} \mathbf{q}^-1
    \dot{\vec{r}}_q = \vec{r}_q \times (\mathbf{q} \vec{\omega} \mathbf{q}^-1)

    \vec{r}_p       = \mathbf{p} \vec{D} \mathbf{p}^-1
    \dot{\vec{r}}_p = \vec{r}_p \times (\mathbf{p} \vec{\phi} \mathbf{p}^-1)