Example 5: Cart Pole Swing Up
=============================


.. math::

    \dot{h}      &= v \sin(\gamma)
    
    \dot{\theta} &= \frac{v}{r} \cos(\gamma) \cos(\psi)
    
    \dot{v}      &= -\frac{D}{m} - g \sin(\gamma) 
    
    \dot{\gamma} &=  \frac{L}{mv}\cos(\beta) + \cos(\gamma)\left( \frac{v}{r} - \frac{g}{v} \right)
    
    \dot{\psi}   &=  \frac{L}{mv \cos(\gamma)}\sin(\beta) +\frac{v}{r \cos(\theta)}\cos(\gamma)\sin(\psi)\sin(\theta)
    

References
----------