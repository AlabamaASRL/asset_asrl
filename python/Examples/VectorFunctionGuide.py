import numpy as np
import asset as ast




'''
One of the goals of ASSET is to provide the ability to users to construct functions
dynamically within Python that are able to be used by ASSET. By doing this we can simplify a user’s workflow,
where the benefits of high speed C++ code can be combined with the ease of use Python provides. 
In this section we will give an depth overview of ASSET's vector function type. At high-level,
this is simple functional (in the programming sense) domain specific langauge for defining 
mathematical Vector Functions that take a fixed number of inputs
and produce a fixed number outputs, with both inputs and outputs asumed to be column vectors.

'''

### Arguments, Segments and Elements
print("########## Arguments, Segments and Elements #################")

'''
To start let is import asset and the vectorfunctions module which contains all types
and functions defing the lanaguage. From this module we will then import the Arguments type
and give it a shorthand name. The Arguments type is the base expression in the vector function system
and represents a function that simply takes some number of input arguments and returns them as outputs.
It always serves as the starting point for defining a more complicated functional expressions involving
some or all of its outputs.
'''

vf        = ast.VectorFunctions
Args      = vf.Arguments

'''
We can construct the object by simply specifing the number of Arguments, in this
case 6. This instance X is now a first class function can takes any vector of size 6
and return that vector. Since it is a vectorfunction we can compute is output value using the
()operator,first derivative using the .jacobian method and second derivative using the .adjointhessian method.
To do do this we provide either a numpy vector or python list of real valued inputs, and additionally for the second derivative
a vector of list of lagrange multipliers with same dimensions as the output of the function. One importnat note, ASSET does not compute
 the full 3D tensor second derivative of vector valued functions, instead it computes the second derivative
dotted with a vector lagrange multiplers, resulting in 2D symetric matrix with rows and columns equal to the number of inputs.
We refer to this as the adjointhessian, and in the case of a function with a single output is equalvalent to the normal hessian.
Since X here is a simple linear function, the first derivative is simply the identity matrix and the adjointhessian is zero. This is
a rather trivial example, but the same methods can applyed to anty asset vector function that we can construct. We should also note that while
these methods are available for all vector functions, for most applications and examples you wont ever actually need to explicitly
call the function or its derivatives at real arguments as that will be handled for you by some other interface such as an integrator or optimal
control problem.
'''

xvals = np.array([0,1,2,3,4,5])
lvals = np.ones((6))

X = Args(6)


print( X(xvals) ) #prints [0,1,2,3,4,5]

print( X.jacobian(xvals) ) #prints Identiy matrix of size 6

print( X.adjointhessian(xvals,lvals) ) #prints zero matrix of size 6

'''
As you can see, Arguments itself does not do any thing very interesting, but what it does do is
serve as a starting point for defining functions of elements and subvectors. For example, we may
make a new object referencing its elements by using the bracket operator. This will return an object of
another fundamental type, Element, which is function that takes all input arguments and returns the specified 
element as a scalar output. Attempting to address an element out of bounds of the Arguments will immediatly throw an
error at the offending operation.
'''

xvals = np.array([0,1,2,3,4,5])

X = Args(6)

x0 = X[0]
x5 = X[5]

#x42 =X[42]  #throws an error

print(x0(xvals))  # prints [0.0]
print(x5(xvals))  # prints [5.0]

'''
Often times you will write an expression where the Arguments represent many seperate
distict scalar elements that will be combined to construct a more complucated expression.
In this case, we can by pass the many lines neccessary to index them individually, by using
the .tolist() method of Arguments. This method will return all the individual elements concatented
in order inside of a single python list, which we can then unpack into individual named elments in a single line.
You may use whatever method you wish, but we personally prefer the tolist method in this case.
'''

xvals = np.array([0,1,2,3,4,5])

X = Args(6)

x0 = X[0]
x1 = X[1]
x2 = X[2]
x3 = X[3]
x4 = X[4]
x5 = X[5]

## Equivalent to 

x0,x1,x2,x3,x4,x5 = X.tolist()

'''
In addition to scalar Elements, one may also adress contiguous subvectors in a set of arguments
using the .head(),.tail(), and .segment() methods of arguments, or standard python (contiguous) list indexing.
For exmample, if we want to treat the first three arguments of the Arguments below as a singlevector R, we can
use the .head(n) method. The .head(n) method returns a subvector of size n starting at the first element. This syntax mirrors the
Eigen C++ library, which we find to be quite nice, but you may also use standard python list indexing to acomplish the same
goal. Similarly, if we want to adress the last three arguments as a single vector V , we can use the .tail(n) method which returns
the last n elements of some arguments. Finally we can address vectors of length n starting at index i with some arguments
using the .segment(i,n) method. The return type of all of these methods is the fundmental Segment type, which is a function that returns
as its output the specified subvector of the arguments.

'''

xvals = np.array([1,2,3,4,5,6])

X = Args(6)

R = X.head(3)
R = X[0:3]    # Same as above

print(R(xvals)) #prints [1,2,3]


V = X.tail(3)
V = X[3:6]

print(V(xvals))  #prints [4,5,6]


R = X.segment(0,3) # same as R above
V = X.segment(3,3) # same as V above


N = X.segment(1,4) # first argument is starting index, second is size
N = X[1:5]         #same as above but python style

print(N(xvals))    # prints [2,3,4,5]

'''
Parrelling what we did before with elements we can also partion an input argument list
list into Segemts and Elements using the .tolist([(start,size), ..]) method. In this case we should
pass a python list of tuples, where the first elemnt of the tuple is the starting idex of the subvector
and the second is the size, subvectors of size one are returned as elements. Note that this method does not
require you to specify partinion all of the argument set, though this example does. Furthermore, it is not
required that the subvectors specified be in any particualar order, though we highly recommend you sort them 
according to starting index.

'''

xvals = np.array([1,2,3,4,5,6,7,8])

X = Args(8)

R = X.head(3)
V = X.segment(3,3)
t = X[6]
u = X[7]

## Equivalent to the Above
R,V,t,u = X.tolist([ (0,3), (3,3), (6,1),(7,1) ])



print(R(xvals)) #prints [1,2,3]

print(V(xvals))  #prints [4,5,6]

print(t(xvals))  #prints [7]

print(u(xvals))  #prints [8]


'''
Finally, all of the above indexing methods behave exaclty the same when applied
to Segments rather Arguments, and we can address their individual components 
as Elements, and split them in smaller subvectors. For example, we may split R into
its scalar components using tolist, adress a single component using brackets, or a subsegment
using head,tail,segment etc..
'''

xvals = np.array([1,2,3,4,5,6])

X = Args(6)

R = X.head(3)
V = X.tail(3)

r0,r1,r2 = R.tolist()

print(r0(xvals))  #prints([1])

v0 = V[0]

print(v0(xvals))  #prints([4])

V12 = V.tail(2)

print(V12(xvals))  #prints([5,6])



############ Standard Math Opertations #######################
print("########## A Standard Math Opertations #################")
      
      
'''
Having covered most everythgin related to constructing arguments, and their elements
and subvectors, we can move on the to combining them together into meaningful mathemeatical functions.
 We should note that the result of any mathematical non idexing operation will have
the generic type VectorFunction or ScalarFunction, which themselves may be operated on and combined with the three
fundamental types using the same rules. In general, types will be converted automatically, and
users should not concern themselves with the types of resulting expressions and should only make sure that their expressions are mathematically consistent. 
We may add subtract multiply, and divide functions by other functions and numerical constants using 
the standard rules of vector math. For example, 
we may add or subtract two functions of the same output size size to together, add or subtratc vectors
of constants or constant scalars, mutlpy functions by constant scalars, mutply functions by Scalar functions, etc.

'''      

xvals = np.array([1,2,3,4,5,6])

X = Args(6)

R = X.head(3)
V = X.tail(3)

S = R[0]*V[0]*V[1]

RpV = R + V

RmC = R - np.array([1,1,1])

Rtv0 = R*V[0]

RtC   = R*2

RdC   = R/2

Vdr0 = V/R[0]

N = Rtv0 + RdC

v1pv0 = (V[1]+V[0] + 9.0)*2.0

inv0 = 1.0/v0

'''
As this is a vector math language, certain operations involving vectors are not 
allowed via standard multply and divide operator overloads. For example one may
not multiply two VectorFunctions together using the * operator as is possible in numpy. 
This is an explicit choise because in our opnion, for the types of expressions written using asset, 
allowing elementwise vector mutlplication creates more problems in terms incorrect problem formulation than it solves.
However, these operations can be acomplished using methods we describe later.Note,
this does not apply to scalar functions such as Element or ScalarFunction, which may be multiplied together with
no issue.
'''

## RmV = R*V  # Throws and Error
## RdV = R_V  # Throws and Error

############ Scalar Opertations ###############################
print("########## Scalar Math Opertations #################")

'''
Next we will move on to describe the standard mathmatical functions that can be applied to scalar
valued functions. These encompass most of the standard functions that can be  in python or C math libraries,
such as sin, cos, tan etc. All of these functions are stored inside the VectorFunctions module(which we have imported as vf),
and can be called as shown below. A complete list of functions is given in the table below (ADD IN THE TABLE).
'''      
      
xvals = np.array([1,2,3,4,5,6])
X = Args(6)

a = vf.sin(X[0])
b = vf.cos(X[1])
c = vf.tan(X[1])

d= vf.cosh((X[1]+X[0])*X[1])

e = vf.arctan2(X[0],X[1]/3.14)

f = X[0]**2  # power operator

g = vf.abs(X[0])

h  = vf.sign(-X[1])  #prints [-1]

############ Unary Opertations ###############################
print("########## Vector Norms Opertations #################")
'''
For Vector Valued functions we also provide member functions that will compute various
useful norms and transformations on vectors. While most of these could be computed using the math opertations
we have already covered, users should always use one of these methods if applicable, as the resulting expresions
will be MUCH faster when evaluated. A complete list of such functions is given below.
'''   
      
X = Args(6)

R = X.head(3)
V = X.tail(3)      
      
r   = R.norm()
r   = vf.sqrt(R[0]**2 + R[1]**2 + R[2]**2)  # Same as above but slower

v2 =  V.squared_norm()
v2 = V[0]**2 + V[1]**2 + V[2]**2 # Same as above but slower


Vhat = V.normalized()
Vhat = V/V.norm()        # Same as above but slower

r3 =  R.cubed_norm()


Grav = - R.normalized_power3()  # R/|R|^3
Grav2 = - R/r3         # Same as above but slower


############ Vector math Opertations ######################
print("########## Vector math Opertations #################")

'''
In addition to the standard binary math operations supported via operator overlaods,
we also provide member functions and free functions for performing various common vector.
The most commonly used are the dot , cross, quaternion, and coefficent wise products,
A few examples of how these can be used are shown below. All functions appearing in these expressions must
obvisouly have the correct output size, otherwise an error will be immediately thrown. You may also
mix and match constant numpy arrays and vectorfunctions as needed to define you function. It should be noted
that our quaternion products assume that the vector part of the quaternion are the fiest three elements of the output 
while the real part is the 4th element(ie: q =[qv,q4])
'''

R,V,N,K = Args(14).tolist([(0,3),(3,3),(6,4),(10,4)])

C2 = np.array([1.0,1.0])
C3 = np.array([1.0,1.0,2.0])
C4 = np.array([1.0,1.0,2.0,3.0])


dRV = R.dot(V)
dRV = vf.dot(R,V)

dRC = R.dot(C3)     # use .dot with a constant vector of size 3
dRC = vf.dot(C3,R)  # Or do it with a free function

#dRC = R.dot(C4)  # throws ERROR because vector is incorrect size


RcrossV = R.cross(V)
RcrossV = vf.cross(R,V)
RcrossC3 = vf.cross(R,C3)

RcVcNdC3 = (R.cross(V)).cross(N.head(3)).dot(C3)

#RcrossC4 = vf.cross(R,C4)  # throws an error

KqpN = vf.quatProduct(K,N) # hamiltonian quaternion product
Krn  = vf.quatRotate(K,V)  ## Rotatrs 3x1 vecor V using quaterion K

KpN  = K.cwiseProduct(N)
NpC4 = N.cwiseProduct(C4)



     
      
      
############ Stack Opertations ########################
print("########## Stack Opertations #################")
      

    
############ Matrix Opertations ##############################
print("########## A Standard Math Opertations #################")
      
############ Boolean Opertations ##############################
print("########## A Standard Math Opertations #################")
          
############ Suggested Style ##############################
      

