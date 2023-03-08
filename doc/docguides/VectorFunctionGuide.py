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
and functions defining the lanaguage. From this module we will then import the Arguments type
and give it a shorthand name. The Arguments type is the base expression in the vector function system
and represents a function that simply takes some number of input arguments and returns them as outputs.
It always serves as the starting point for defining a more complicated functional expressions involving
some or all of its outputs.
'''

vf        = ast.VectorFunctions
Args      = vf.Arguments

'''
We can construct the object by simply specifing the number of arguments, in this
case 6. This instance X is now a first class function thats takes any vector of size 6
and returns that vector. Since it is a vectorfunction we can compute is output value using the
()operator,first derivative using the .jacobian method and second derivative using the .adjointhessian method.
To do do this we provide either a numpy vector or python list of real valued inputs, and additionally for the second derivative
a vector of list of lagrange multipliers with same dimensions as the output of the function. One important note, ASSET does not compute
 the full 3D tensor second derivative of vector valued functions, instead it computes the second derivative
dotted with a vector lagrange multiplers, resulting in 2D symmetric matrix with rows and columns equal to the number of inputs.
We refer to this as the adjointhessian, and in the case of a function with a single output is equalvalent to the normal hessian.
Since X here is a simple linear function, the first derivative is simply the identity matrix and the adjointhessian is zero. This is
a rather trivial example, but the same methods can applyed to any asset vector function that we can construct. We should also note that while
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
make a new object referencing one of its elements by using the bracket operator. This will return an object of
another fundamental type, Element, which is itself a function that takes all input arguments and returns the specified
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
distinct scalar elements that will be combined to construct a more complicated expression.
In this case, we can bypass the many lines neccessary to index them individually, by using
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
using the .head(),.tail(), and .segment() methods of Arguments, or standard python (contiguous) list indexing.
For example, if we want to treat the first three arguments of the Arguments below as a singlevector R, we can
use the .head(n) method. The .head(n) method returns a subvector of size n starting at the first element. This syntax mirrors the
Eigen C++ library, which we find to be quite nice, but you may also use standard python list indexing to acomplish the same
goal. Similarly, if we want to adress the last three arguments as a single vector V , we can use the .tail(n) method which returns
the last n elements of some arguments. Finally we can address vectors of length n starting at index i
using the .segment(i,n) method. The return type of all of these methods is the fundmental Segment type, which is a function that returns
as its output the specified subvector of the arguments.

'''

xvals = np.array([1,2,3,4,5,6])

X = Args(6)

R = X.head(3)
R = X[0:3]    # Same as above

print(R(xvals)) #prints [1,2,3]


V = X.tail(3)
V = X[3:6]    # same as above

print(V(xvals))  #prints [4,5,6]


R = X.segment(0,3) # same as R above
V = X.segment(3,3) # same as V above


N = X.segment(1,4) # first argument is starting index, second is size
N = X[1:5]         #same as above but python style

print(N(xvals))    # prints [2,3,4,5]

'''
Paralelling what we did before with elements we can also partion an input argument list
list into Segments and Elements using the .tolist([(start,size), ..]) method. In this case we should
pass a python list of tuples, where the first element of each tuple is the starting idex of the subvector
and the second is the size, subvectors of size one are returned as Elements. Note that this method does not
require you to partition all of the argument set, though this example does. Furthermore, it is not
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
Finally, all of the above indexing methods behave exactly the same when applied
to Segments rather than Arguments, and we can address their individual components
as Elements, and split them in smaller segments. For example, we may split R into
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
Having covered most everything related to constructing arguments, and their elements
and subvectors, we can move on the to combining them together into meaningful mathematical functions.
 We should note that the result of any mathematical non idexing operation will have
the generic type VectorFunction (more than one output) or ScalarFunction (one output),
which themselves may be operated on and combined with the three
fundamental types using the same rules. In general, types will be converted automatically, and
users should not concern themselves with the types of resulting expressions
and should only make sure that their expressions are mathematically consistent.
We may add subtract multiply, and divide functions by other functions and numerical constants using
the standard rules of vector math. For example,
we may add or subtract two functions of the same output size size to together, add or subtract vectors
of constants or constant scalars, multiply functions by constant scalars, multiply functions by Scalar functions, etc.

'''

xvals = np.array([1,2,3,4,5,6])

X = Args(6)

R = X.head(3)
V = X.tail(3)

S = R[0]*V[0]*V[1]*5.0

RpV = R + V

RmC = R - np.array([1.0,1.0,1.0])

Rtv0 = R*V[0]

RtC   = R*2

RdC   = R/2

Vdr0 = V/R[0]

N = Rtv0 + RdC

v1pv0 = (V[1]+V[0] + 9.0)*2.0

inv0 = 1.0/v0

'''
As this is a vector math language, certain operations involving vectors are not
allowed via standard multpily and divide operator overloads. For example one may
not multiply two VectorFunctions together using the * operator as is possible with two arrays in numpy.
This is an explicit choice because in our opinion, for the types of expressions written using asset,
allowing elementwise vector mutlplication creates more problems in terms incorrect problem formulation than it solves.
However, these operations can be acomplished using methods we describe later.Note,
this does not apply to scalar functions such as Element or ScalarFunction, which may be multiplied together with
no issue, and may also scale in VectorFunction.
'''

## RmV = R*V  # Throws and Error
## RdV = R/V  # Throws and Error

############ Scalar Opertations ###############################
print("########## Scalar Math Operations #################")

'''
Next we will move on to describe the standard mathmatical functions that can be applied to scalar
valued functions. These encompass most of the standard functions that can be found in python or C math libraries,
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

############ Unary Operations ###############################
print("########## Vector Norms Operations #################")
'''
For Vector Valued functions we also provide member functions that will compute various
useful norms and transformations on vectors. While most of these could be computed using the math operations
we have already covered, users should always use one of these methods if applicable, as the resulting expresions
will be MUCH faster when evaluated. A few examples are illustrated here,
 and a complete list of such functions is given in the table below.
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


############ Vector math Operations ######################
print("########## Vector math Operations #################")

'''
In addition to the standard binary math operations supported via operator overloads,
we also provide member functions and free functions for performing various common vector operations.
The most commonly used are the dot , cross, quaternion, and coefficent wise products,
A few examples of how these can be used are shown below. All functions appearing in these expressions must
have the correct output size, otherwise an error will be immediately thrown. You may also
mix and match constant numpy arrays and vectorfunctions as needed to define your function. It should be noted
that our quaternion products assume that the vector part of the quaternion is the first three components of the output
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



############ Stack Operations ########################
print("########## Stack Operations #################")
'''
Up to this point, we have looked at partionioning and operating on the outputs
of other functions, and have not addressed how the outputs of functions may be combined together
into a larger single function. This can be accomplished using the VERY IMPORTANT vf.stack() method.
In general stack takes a list of asset function types and produces another function whose output is the concatination
of all the outputs. There are two signatures for stack, The first one (vf.stack([f1,f2,...])) takes a python list
containing only explicit asset function types (ie:Element,ScalarFunction,VectorFunction,Segment etc..).
This version does not allow one to mix in floats or numpy vectors. The second signature (vf.stack(f1,f2,...)) does the
same thing as the first but does not enclose the the objects to be stacked inside of list. Additionally,
for this second signature, you may mix in arbitrary floats and numpy vectors that will be included in the output.
'''

xvals = np.array([1,0,0,
                  0,1,0])

R,V = Args(6).tolist([(0,3),(3,3)])

Rhat = R.normalized()
Nhat = R.cross(V).normalized()
That = Nhat.cross(Rhat).normalized()

RTN = vf.stack([Rhat,That,Nhat])
print(RTN(xvals))  #prints [1. 0. 0. 0. 1. 0. 0. 0. 1.]

#Err = vf.stack([Rhat,That,np.array([1.0,1.0])]) # Throws Error, numpy array not allowed

RTN = vf.stack(Rhat,That,Nhat)  # Same as above

Stuff = vf.stack(7.0, Rhat,42.0,That,Nhat, np.array([2.71,3.14]) )

print(Stuff(xvals))  #prints [ 7., 1., 0.,  0., 42., 0., 1., 0. ,0., 0. ,1. ,2.71,3.14]


############ Matrix Operations ##############################
print("########## Matrix Operations #################")

'''
While ASSET is and always will be a language for defining functions with vector valued
inputs and outputs, we do have limited but growing support for intepreting vector functions
as matrices inside of expressions. This is supported through the vf.ColMatrix and vf.RowMatrix types.
These are types constructed from some vector function and interprets the outputs as nxm matrix.
A ColMatrix will interpret the coefficients of output as a Column major matrix, whereas RowMatrix interprets
them as a row major matrix. Once constructed you may mutlpily matrices by any other appropriately sized
Row/ColMatrix functions in any order, or multply them on the right by appropriately sized VectorFunctions. The result
of all matrix on matrix operations are assumed to be ColMatrix type. The result of Matrix*vector operations are VectorFunctions.
Furthermore, square matrices may be inverted resulting in a Matrix type with same row/col type. For now one, may only add matrics
together if they have the same Row/Col type, though we will support this in the future.
'''

R,V,U = Args(9).tolist([(0,3),(3,3),(6,3)])

## Three orthonormal basis vectors
Rhat = R.normalized()
Nhat = R.cross(V).normalized()
That = Nhat.cross(Rhat).normalized()

RTNcoeffs = vf.stack([Rhat,That,Nhat])

RTNmatC = vf.ColMatrix(RTNcoeffs,3,3)  # Interprate as col major 3x3 Rotation matrix
RTNmatR = vf.RowMatrix(RTNcoeffs,3,3)  # Interprate as row major 3x3 Rotation matrix

M2 = RTNmatC*RTNmatR # Mulitpliy matrices together result is column major

U1 = RTNmatC*U       # Multiply on the right by a VectorFunction of size (3x1)
U2 = RTNmatR*U
U3 = M2*U

ZERO = RTNmatR.inverse()*U -RTNmatC*U


RTNmatC +RTNmatC


############ Boolean Operations ##############################
print("########## Conditional Statement/Operations #################")
'''
Asset's intended use case is for defining constraint,objectives, and dynamical
models that will eventually be put to use inside of a second derivative optimizer. As a
general rule of thumb, it is a bad idea for such functions to contain conditional statements,
as this could potentially result non-smooth derivatives. In these cases we always recomend considering
whether what you were trying to accomplish with the conditional statement can be reformulated in another way.
However if this is not possible, or you are writing a function that will not see the inside of an optimizer,
we do offer support for simple conditional statemants and boolean operations with vector function expressions.
To be precise,we suport constructing boolean statements involving the outputs of scalar valued functions, and then
using those as conditional statements to control the output of another expression. Conditional statements are constructed by
applying the comparison operators (>,<,<=,>=) to the outputs of scalar functions. This can be used to dispatch one of
two functions using the vf.ifelse() function as shown below. Note that the output sizes of both the true and false functions
MUST be the same. Conditional statements may also be combined together using the bitwise or and and opertors (|,&).
'''
x0,x1,x2 = Args(3).tolist()

condition = x0<1.0

output_if_true = x1*2
output_if_false = x1+x2

func = vf.ifelse(condition,output_if_true,output_if_false)


print(func([0,  2,3]))  # prints [4.0]
print(func([1.5,2,3]))  # prints [5.0]


Fine = vf.ifelse(condition,vf.stack(x1,x2),vf.stack(x2,x1))
#Error = vf.ifelse(condition,vf.stack(x1,x2),output_if_false)


combo_condition = (x0<1.0)|(x0>x1)

func = vf.ifelse(combo_condition,output_if_true,output_if_false)


print(func([0,  2,3]))  # prints [4.0]
print(func([1.5,2,3]))  # prints [5.0]
print(func([2.5,2,3]))  # prints [4.0]

############ Constraints on Input Arguments #####################################
print("########## Constraints on Input Arguments #################")

'''
Before, moving on any further, we need to make one very important note about how the vector
function type system works. In all of our previous examples, we have created and partitioned
one set of arguments of a certain size, from which we constructed other functions. You might
ask, what happens if we try to mix expressions formulated out of arguments of different sizes.
This is strictly not allowed, as our entire type system predicated on the fact that expressions can
only be combined if they have the same sized input arguments. For example, the follwing code will
throw an error to alert you that you have made a mistake. However, we should also note as shown below,
that there is nothing unique about any two sets of arguments of the same size. Thus you may (though it is pointless)
combine expressions derived from two arguments objects of the same size.
'''

X1 = Args(9)
X2 = Args(12)
X3 = Args(12)

R1,V1,U1 = X1.tolist([(0,3),(3,3),(6,3)])
R2,V2,U2 = X2.tolist([(0,3),(3,3),(6,3)])
R3,V3,U3 = X3.tolist([(0,3),(3,3),(6,3)])

#Error = R1 + R2
#Error = R1.dot(V2)

## These two functions do identical things
Fine = R2.dot(V3)
Fine = R3.dot(V2)

############ Suggested Style and Organization #################################
print("########## STYL E#############")
'''
At this point we have covered most all of the operations one can and cant perform with asset
vector functions, with the important exception of function composition
(which we will cover in the next section). As you might have noticed, in all of
our scratch pad examples, we simply created a single set of arguments and operated on them
in the same scope. Everyone of these functions is a fully formed asset type and can be immeditaly passed
off to other parts of the library to be used as contraints/ODEs/controllers etc. However, obviously it is not a recipe
for longterm success to simply write expressions inline wherever they are needed. How you package or
encapuslate the construction of asset vectorfunctions is up to you, but we suggest one of the following two methods.

Method one involves simply writing a standard python function that takes as arguments
any meta data or constants, needed to define the function, then writing and returning your asset
vector function. A trivial example of this is shown below, and you can find many others throught our
problem specific examples contained in other sections.
'''

def FuncMeth(a,b,c):
    x0,x1,x2 = Args(3).tolist()
    eq1 = x0 +a - x1
    eq2 = x2*b + x1*c
    return vf.stack(eq1,eq2)

func = FuncMeth(1,2,3)

print(func([1,1,1]))  # prints [1,5]

'''
Method two involves defining a new class that inherits from the apropriate
ASSET type (vf.VectorFunction if output size is >1, vf.ScalarFunction of output size =1)
and then defining and initializing the expression in the constructor. This method should only
be preffered if you need to store the metadata as part of the class
or add additional methods to the object. Otherwise, this method is functionally identical to
the one above.
'''

class FuncClass(vf.VectorFunction):
    def __init__ (self,a,b,c):
        self.a =a
        self.b =b
        self.c =c

        x0,x1,x2 = Args(3).tolist()
        eq1 = x0 +a - x1
        eq2 = x2*b + x1*c

        super().__init__(vf.stack(eq1,eq2)) #Do not forget to call CTOR of Base!!

    def get_a(self):return self.a

func = FuncClass(1,2,3)

print(func([1,1,1]))  # prints [1,5]
print(func.get_a())   # prints 1


############ Function Composition ########################
print("########## Function Composition  #################")

'''
Now that we have a good understanding of the rules and style for defining
single vectorfunctions, we can cover how to call them inside of other functions.
For this final example let us tackle a concrete problem that occurs
in Astrodynamics: Frame conversions. Specifically, we wish to write a function that takes
the position and velocity of some object in cartesian coordinates, as well as some other vector,
and then transforms that vector into the RTN frame. The RTN basis vectors can be computed purely as a function
of position and velocity, so let us first write a function that does just that.
'''

def RTNBasis():

    R,V = Args(6).tolist([(0,3),(3,3)])

    Rhat = R.normalized()
    Nhat = R.cross(V).normalized()
    That = Nhat.cross(R).normalized()

    return vf.stack(Rhat,Nhat,That)


'''

We can then write another function that takes position and velocity as well as the vector
to be transformed. We then instantiate our prevously defined function that
computes basis vectors and then "call" it with the position and velocity arguments
defined inside our new function. Calling the already instantiated function, can be accomplished
using the providing a correctly sized function to the () call operator the same way we do for
real number arguments. In this case, providing the contigous segment of size 6 RV, is the most efficient
way to the define the expression. However, if this were not the case, we could also the other
call signatures shown. We can provide two seperate functions, in this case R and V
either as individual arguments or grouped together in a python list. These will be implictly
stacked using the same rules governing vf.stack and then forwarded to the function.

'''

def RTNTransform():


    X = Args(9)

    RV,U = X.tolist([(0,6),(6,3)])

    R,V = X.tolist([(0,3),(3,3)])

    RTNBasisFunc = RTNBasis() # Instanatiate function object


    RTNcoeffs = RTNBasisFunc(RV)  ### Call Function at new vectorfunction arguments

    RTNcoeffs = RTNBasisFunc(R,V) # Same effect as original but slower
    RTNcoeffs = RTNBasisFunc(vf.stack(R,V)) # Does Exactly the same thing as above calls stack on R,V explicitly

    RTNcoeffs = RTNBasisFunc([R,V]) # Same effect as original but slower
    RTNcoeffs = RTNBasisFunc(vf.stack([R,V])) # Does Exactly the same thing as above calls stack on [R,V] explicitly



    RTNmat = vf.RowMatrix(RTNcoeffs,3,3)

    U_RTN = RTNmat*U

    return U_RTN




print("########## Repreated Sub Expressions #################")


'''
Being a functional programming language, it is important to note that an asset
expression is evaluated every where it appears in a statement. There is no notion
of assigning it to a temporary variable and then resusing it later without recalculating it.
For example,in the following code, just because we bind the complicated expression to
the name expensive, the function answer will still require actually
evaluating expensive three times.
'''

R,V = X.tolist([(0,3),(3,3)])

expensive = 1.0/(R.normalized().cross(V.normalized_power3()).dot(R+V.cross(R).normalized()))**3.14

answer = R+ vf.stack(expensive,expensive+1,expensive)

'''
In the vast majority of cases you should not worry about the cost of reevaluating subexpressions,
as the run time hit is marginal. There is however, one way to explictly ensure to reduce the cost of expensive repeated
sub expressions should you need to. You can do this by writing a second function where the subexpression apears
linearly as additional arguments or segments and then using the call operator to compose this new function
and the original arguments and subexpression together. For example, the following code will producde the same output
as above while only ever evaluating expensive once.
'''

R,V = X.tolist([(0,3),(3,3)])

expensive = 1.0/(R.normalized().cross(V.normalized_power3()).dot(R+V.cross(R).normalized()))**3.14


## New args for defining function of only R and expensive
R_temp, expensive_tmp = Args(4).tolist([(0,3),(3,1)])

answer_tmp = R_temp+ vf.stack(expensive_tmp,expensive_tmp+1,expensive_tmp)

answer = answer_tmp([R,expensive])




print("########## Size of Arguments #################")
'''
The vectorfunction type system has been designed to have good performance for evaluating
the value and derivatives of dense vector functions with a small number of arguements (<50).
It will work for larger expressions, but performance will begin to degrade considerably. This may seem
strange since it ostensibly designed to be used to defign constraitns and objective inside of large
sparse non-linear programs. Howerver, in out experience these problems are almost never composed
of single monolithic functions, and can generally be decomposed into smaller dense functions that only
take a partial subsets of the problem variables. In that case we can define our functions in terms of
only the arguments they take, and then under the hood, asset will ensure that the inputs and outputs are gathered and
scattered to the correct locations inside the larger problem. The specifics of how this works will be handled in later
sections.
'''

X = Args(1000) # Bad

print("########## Tables #################")

############ 1D Tables ##############################
#####################################################
import matplotlib.pyplot as plt
import time

tt = time.perf_counter()
'''
One dimensonal
'''
ts = np.linspace(0,2*np.pi,90)

VecDat = np.array([ [np.sin(t),np.cos(t)] for t in ts])

kind = 'cubic' # or 'linear' 

Tab = vf.InterpTable1D(ts,VecDat,axis=0,kind=kind)
print(Tab(np.pi/2.0)) #prints [1,.0]

# Or if data is transposed
Tab = vf.InterpTable1D(ts,VecDat.T,axis=1,kind=kind)
print(Tab(np.pi/2.0)) #prints [1,.0]

# Or if data is a list of arrays or lists with time included as one the elements
VecList = [ [np.sin(t), np.cos(t), t] for t in ts]

Tab = vf.InterpTable1D(VecList,tvar=2,kind=kind)
print(Tab(np.pi/2.0)) #prints [1,.0]

###########################################
ScalDat = [np.sin(t) for t in ts]
STab =vf.InterpTable1D(ts,ScalDat,kind=kind)
print(STab(np.pi/2.0)) # prints [1.0]


#############################################
Tab.WarnOutOfBounds=True   # By default
print(Tab(-.00001))        # prints a warning
Tab.ThrowOutOfBounds=True
#print(Tab(-.00001))       # throws exception


x,V,t = Args(4).tolist([(0,1),(1,2),(3,1)])

f1 = STab(t) + x # STab(t) is an asset scalar function
f2 = Tab(t) + V  # Tab(t) is an asset vector function

print(f1([1,1,1,np.pi/2])) #prints



TabFunc = Tab.vf() # Now it is a vector function with input size 1
STabFunc = STab.sf() # Now a scalar function with input size 1





ts2 = np.linspace(0,2*np.pi,1000)
Vals = Tab(ts2)



plt.plot(ts2,Vals[0],label='sin')
plt.plot(ts2,Vals[1],label='cos')
plt.plot(ts2,np.cos(ts2),label='cos')
plt.show()

############ 2D Tables ##############################
#####################################################

nx =500
ny =800

lim = 2*np.pi

xs = np.linspace(-lim,lim,nx)
ys = np.linspace(-lim,lim,ny)

def f(x,y):
    return np.sin(x)*np.cos(y) 


X, Y = np.meshgrid(xs, ys)
Z    = f(X,Y)             #Scalar data defined on 2-D meshgrid

kind = 'cubic' # or 'linear'

Tab2D = vf.InterpTable2D(xs,ys,Z,kind=kind)

print(Tab2D(np.pi/2,0))  #prints 1.0


#############################################
Tab2D.WarnOutOfBounds=True   # By default
print(Tab2D(-6.3,0))        # prints a warning
Tab2D.ThrowOutOfBounds=True
#print(Tab2D(-6.3,0))       # throws exception
#############################################

xy,c= Args(3).tolist([(0,2),(2,1)])
x,y = xy.tolist()

# Use it as scalar function inside a statement
Tab2sf = Tab2D(xy)
Tab2sf = Tab2D(x,y)             # Functionally same as above but slower
Tab2sf = Tab2D(vf.stack([x,y])) # Functionally same as above but slower

Func = Tab2sf + c   # Use it as a normal scalar function

print(Func([np.pi/2,0,1.0]))  # prints [2.0]

############ 3D Tables ##############################
#####################################################
def f(x,y,z):return np.cos(x)*np.cos(y)*np.cos(z)

nx = 100
ny = 100
nz = 100

xlim = np.pi
ylim = np.pi
zlim = np.pi

xs = np.linspace(-xlim,xlim,nx)
ys = np.linspace(-ylim,ylim,ny)
zs = np.linspace(-zlim,zlim,nz)

X,Y,Z = np.meshgrid(xs, ys,zs,indexing = 'ij')
Fs    = f(X,Y,Z)    #Scalar data defined on 3-D meshgrid in ij format!!!

kind = 'cubic' # or 'linear', defaults to 'cubic'
cache = False # defaults to False
#cache = True # will precalcuate and cache all interpolation coeffs

Tab3D = vf.InterpTable3D(xs,ys,zs,Fs,kind=kind,cache=cache)

print(Tab3D(0,0,0))  #prints 1.0 

#############################################
Tab3D.WarnOutOfBounds=True   # By default
print(Tab3D(-10,0,0))        # prints a warning
print(Tab3D(0,-10,0))        # prints a warning
print(Tab3D(0,0,-10))        # prints a warning

Tab3D.ThrowOutOfBounds=True
#print(Tab3D(-10,0,0))       # throws exception
#############################################

xyz,c= Args(4).tolist([(0,3),(3,1)])
x,y,z = xyz.tolist()

# Use it as scalar function inside a statement
Tab3sf = Tab3D(xyz)
Tab3sf = Tab3D(x,y,z)             # Functionally same as above but slower
Tab3sf = Tab3D(vf.stack([x,y,z])) # Functionally same as above but slower

Func = Tab3sf + c

print(Func([0,0,0,1]))  # prints [2.0]

print("########## Binding Raw Python (DONT READ THIS) #################")


# A vector function
def VFunc(X,a,b):
    return np.array([a*X[0]**2,X[1]*b])

InputSize = 2
OutputSize =2

PyVfunc =vf.PyVectorFunction(InputSize,OutputSize,
                             VFunc,
                             Jstepsize=1.0e-6,Hstepsize=1.0e-4,
                             args = (3,7)) ## a and b will be 2 

print(PyVfunc([2,2]))  # prints [12,14]


# A scalar function
def SFunc(X,a,b,c):
    return np.array([a*X[0]**2 + X[1]*b + c]) # output is 1x1 array

InputSize = 2

PySfunc =vf.PyScalarFunction(InputSize,
                             SFunc,
                             Jstepsize=1.0e-6,Hstepsize=1.0e-5,
                             args = (1,2,3))  

print(PySfunc([2,2])) #prints [11]

print(PySfunc.adjointhessian([2,2],[1])) #prints [11]





