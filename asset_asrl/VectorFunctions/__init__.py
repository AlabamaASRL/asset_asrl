from asset.VectorFunctions import *
import asset as _asset
import inspect



Arguments = _asset.VectorFunctions.Arguments
ColMatrix = _asset.VectorFunctions.ColMatrix
Comparative = _asset.VectorFunctions.Comparative
Conditional = _asset.VectorFunctions.Conditional
ConstantScalar = _asset.VectorFunctions.ConstantScalar
ConstantVector = _asset.VectorFunctions.ConstantVector
Element = _asset.VectorFunctions.Element
PyScalarFunction = _asset.VectorFunctions.PyScalarFunction
PyVectorFunction = _asset.VectorFunctions.PyVectorFunction
RowMatrix = _asset.VectorFunctions.RowMatrix
ScalarFunction = _asset.VectorFunctions.ScalarFunction
Segment = _asset.VectorFunctions.Segment
Segment2 = _asset.VectorFunctions.Segment2
Segment3 = _asset.VectorFunctions.Segment3
Stack = _asset.VectorFunctions.Stack
StackScalar = _asset.VectorFunctions.StackScalar
Sum = _asset.VectorFunctions.Sum
SumElems = _asset.VectorFunctions.SumElems
SumScalar = _asset.VectorFunctions.SumScalar
VectorFunction = _asset.VectorFunctions.VectorFunction

arccos = _asset.VectorFunctions.arccos
arcsin = _asset.VectorFunctions.arcsin
arctan = _asset.VectorFunctions.arctan
arctan2 = _asset.VectorFunctions.arctan2
cos = _asset.VectorFunctions.cos
cosh = _asset.VectorFunctions.cosh
cross = _asset.VectorFunctions.cross
cwiseProduct = _asset.VectorFunctions.cwiseProduct
cwiseQuotient = _asset.VectorFunctions.cwiseQuotient
dot = _asset.VectorFunctions.dot
doublecross = _asset.VectorFunctions.doublecross
exp = _asset.VectorFunctions.exp
ifelse = _asset.VectorFunctions.ifelse
inverse_norm = _asset.VectorFunctions.inverse_norm
log = _asset.VectorFunctions.log
matmul = _asset.VectorFunctions.matmul
norm = _asset.VectorFunctions.norm
normalize = _asset.VectorFunctions.normalize
pow = _asset.VectorFunctions.pow
quatProduct = _asset.VectorFunctions.quatProduct
sin = _asset.VectorFunctions.sin
sinh = _asset.VectorFunctions.sinh
sqrt = _asset.VectorFunctions.sqrt
squared = _asset.VectorFunctions.squared
stack = _asset.VectorFunctions.stack
stack_scalar = _asset.VectorFunctions.stack_scalar
sum = _asset.VectorFunctions.sum
tan = _asset.VectorFunctions.tan
tanh = _asset.VectorFunctions.tanh


from .Extensions.DerivChecker import FDDerivChecker


if __name__ == "__main__":
    mlist = inspect.getmembers(_asset.VectorFunctions)
    for m in mlist:print(m[0],'= _asset.VectorFunctions.'+str(m[0]))
    
    