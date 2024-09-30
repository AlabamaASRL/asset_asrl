
import sympy as sp
import numpy as np
import re
from sympy.codegen.cfunctions import Sqrt

def find_pow_expressions(s):
    expressions = []
    start_index = None
    parentheses_count = 0
    
    for i, c in enumerate(s):
        if s[i:i+4] == 'pow(':
            if start_index is None:
                start_index = i
            parentheses_count += 1
            print(parentheses_count)

        elif c == '(' and start_index is not None:
            parentheses_count += 1
            print(parentheses_count)

        elif c == ')' and start_index is not None:
            parentheses_count -= 1
            print(parentheses_count)
            if parentheses_count == 1:
                parentheses_count -= 1
                expressions.append(s[start_index:i+1])
                start_index = None
    return expressions

def find_innermost_pow_expressions(s):
    expressions = []
    start_index = None
    parentheses_count = 0
    pow_count = 0
    
    for i, c in enumerate(s):
        if s[i:i+4] == 'pow(':
            if start_index is None:
                start_index = i
            parentheses_count += 1
            pow_count += 1
        elif c == '(' and start_index is not None:
            parentheses_count += 1
        elif c == ')' and start_index is not None:
            parentheses_count -= 1
            if parentheses_count == 0:
                expressions.append(s[start_index:i+1])
                start_index = None
                pow_count = 0
            elif pow_count > 1 and parentheses_count == pow_count - 1:
                expressions.append(s[start_index:i+1])
                start_index = None
                pow_count -= 1
                
    return expressions

class AssetHeaderGen:
    
    def __init__(self,Name,
                 F,
                 Xs,
                 ScalarParams=[], # [(Symbol,Description),]
                 VectorParams=[], # [(Vec,Cppname,Description),]
                 MatrixParams=[], # [(Mat,Cppname,Description),]
                 docstr = "A doc string"
                 ):
        
        self.Name = Name
        self.ninputs  = len(Xs)
        self.noutputs = len(F)
        
        self.Func   = sp.Matrix(list(F))
        self.Inputs = sp.Matrix(list(Xs))
        self.ScalarParams = ScalarParams
        self.VectorParams = VectorParams 
        self.MatrixParams = MatrixParams 

        self.docstr = docstr

        print("Generating Jacobian")
        self.Jac = self.Func.jacobian(self.Inputs)
        print("Finished Jacobian")
        self.LMults =sp.Matrix(sp.symbols('LM:'+str(self.noutputs)))
        print("Generating Gradient")
        self.Grad = self.Jac.transpose()*self.LMults
        print("Finished Gradient")
        print("Generating Hessian")
        self.Hess = self.Grad.jacobian(self.Inputs)
        print("Finished Hessian")

    
    
    
    def powsimp(self,expr):
        
        powcount = expr.count("pow")
        commacount = expr.count(",")
        
        if(powcount>0):
            for passes in range(0,1):
                ## Convert integer pows to multiplies
                for i in range(2,16):
                    regex = re.compile(rf'pow\((?P<var>[A-Za-z_]\w*)\s*,\s*{i}\s*\)')
                    repl=r'\g<var>'
                    for j in range(1,i):
                        repl +=r'*\g<var>'
                    expr = re.sub(regex, repl, expr)
        
                ## Convert pow(x,a.5)  to  (sqrt(x)*x**a)
                for i in range(0,7):
                    regex = re.compile(rf'pow\((?P<var>[A-Za-z_]\w*)\s*,\s*{i}.5\s*\)')
                    repl=r'(sqrt(\g<var>)'
                    for j in range(1,i+1):
                        repl +=r'*\g<var>'
                    repl+=r')'
                    expr = re.sub(regex, repl, expr)
                    
                ## Convert pow(x, i.0 / 2.0)  to  (sqrt(x)*x**a)
                for i in range(1,10,2):
                    
                    regex = re.compile(rf'pow\((?P<var>[A-Za-z_]\w*)\s*,\s*{i}.0/2.0\s*\)')
                    repl=r'(sqrt(\g<var>)'
                    for j in range(0,int((i-1)/2)):
                        repl +=r'*\g<var>'
                    repl+=r')'
                    expr = re.sub(regex, repl, expr)
        
                ## Convert pow(x,-a.5)  to  1/(sqrt(x)*x**a)
                for i in range(0,7):
                    regex = re.compile(rf'pow\((?P<var>[A-Za-z_]\w*)\s*,\s*-{i}.5\s*\)')
                    repl=r'(Scalar(1.0)/(sqrt(\g<var>)'
                    for j in range(1,i+1):
                        repl +=r'*\g<var>'
                    repl+=r'))'
                    expr = re.sub(regex, repl, expr)
                    
                ## Convert pow(x,-a)  to  1/(x**a)
                for i in range(1,16):
                    regex = re.compile(rf'pow\((?P<var>[A-Za-z_]\w*)\s*,\s*-{i}\s*\)')
                    repl=r'(Scalar(1.0)/(\g<var>'
                    for j in range(1,i):
                        repl +=r'*\g<var>'
                    repl+=r'))'
                    expr = re.sub(regex, repl, expr)
                    
                ## Convert pow(x, -i.0 / 2.0)  to  1/(sqrt(x)*x**a)
                for i in range(1,10,2):
                     regex = re.compile(rf'pow\((?P<var>[A-Za-z_]\w*)\s*,\s*-{i}.0/2.0\s*\)')
                     repl=r'(Scalar(1.0)/(sqrt(\g<var>)'
                     for j in range(0,int((i-1)/2)):
                         repl +=r'*\g<var>'
                     repl+=r'))'
                     expr = re.sub(regex, repl, expr)
            
            powcount = expr.count("pow")
            commacount = expr.count(",")
            
            if(powcount==1 and commacount==1):
                
                ## Convert pow(x,-a.5)  to  1/(sqrt(x)*x**a)
                for i in range(0,7):
                    regex = re.compile(rf'pow\((?P<var>\s*([^,]+)),\s*-{i}.5\s*\)')
                    repl=r'(Scalar(1.0)/(sqrt(Scalar(\g<var>))'
                    for j in range(1,i+1):
                        repl +=r'*Scalar(\g<var>)'
                    repl+=r'))'
                    expr = re.sub(regex, repl, expr)
        return expr
    
    
                     
                
                 
        
    
    def fixup(self,expr):
       
        ## Scalarize params
        
        wrapscalar = False
        
        for Param,Descr in self.ScalarParams:
            Name = str(Param)
            if Name in expr:
                wrapscalar = True
                
        for Vec,Name,Descr in self.VectorParams:
            for i,elem in enumerate(Vec):
                symname = str(elem)
                if symname in expr:
                    replname = "({0:}[{1:}])".format(Name,i)
                    regex = re.compile(rf'\b{symname}\b')
                    expr = re.sub(symname,replname,expr)
                    wrapscalar = True
                    
        for Mat,Name,Descr in self.MatrixParams:
            rows = len(Mat)
            cols = len(Mat[0])
            for i in range(0,rows):
                for j in range(0,cols):
                    symname = str(Mat[i,j])
                    if symname in expr:
                        replname = "({0:}({1:},{2:}))".format(Name,i,j)
                        regex = re.compile(rf'\b{symname}\b')
                        expr = re.sub(regex,replname,expr)
                        wrapscalar = True

        if(wrapscalar):
            expr = 'Scalar('+expr+")"
            
            
        ## the pow simplifier is broken, needs to be fixed
        #expr = self.powsimp(expr) 
        
        
        return expr
        
        
        
        
    def gen_class_header(self):
        
        classname = "struct {0:} : VectorFunction<{0:},{1:},{2:},Analytic,Analytic> {{\n".format(self.Name,self.ninputs,self.noutputs)
        usingbase = "  using Base =  VectorFunction<{0:},{1:},{2:},Analytic,Analytic>;\n".format(self.Name,self.ninputs,self.noutputs)
        message = " // THIS CLASS WAS AUTOGENERATED //"
        basetypes = "  DENSE_FUNCTION_BASE_TYPES(Base);\n"
        
        members =""
        dctor = "{0:}(){{}}\n\n".format(self.Name)
        ctorheader = "{0:}(".format(self.Name)
        typedarglist=""
        untypearglist=""
        vflambda ="[]("
        ctorbody = ""
        
        nargs = len(self.ScalarParams)+len(self.VectorParams) + len(self.MatrixParams)
        osize = 1 if self.noutputs==1 else -1
        
       
        
        i=0
        for P,Descr in self.ScalarParams:
            delim= "" if i ==nargs-1 else ","
            typedarglist+= "double {0:}{1:}".format(str(P),delim)
            untypearglist+= " {0:}{1:}".format(str(P),delim)

            ctorbody+= "\n\t this->{0:}={0:};".format(str(P))
            members += "\n\t double {0:}; // {1:}".format(str(P),Descr)
            i+=1
            
        for Vec,Name,Descr in self.VectorParams:
            delim= "" if i == nargs-1 else ","
            vsize = len(Vec)
            typedarglist+= "const Eigen::Matrix<double,{1:},1> & {0:} {2:}".format(str(Name),vsize,delim)
            untypearglist+= "{0:} {2:}".format(str(Name),vsize,delim)

            ctorbody+= "\n\t this->{0:}={0:};".format(str(Name))
            members += "\n\t Eigen::Matrix<double,{1:},1> {0:}; // {2:}".format(str(Name),vsize,Descr)
            i+=1
        
        for Mat,Name,Descr in self.MatrixParams:
            delim= "" if i == nargs-1 else ","
            rows = len(Mat)
            cols = len(Mat[0])
            typedarglist+= "const Eigen::Matrix<double,{1:},{2:}> & {0:} {3:}".format(str(Name),rows,cols,delim)
            untypearglist+= " {0:}{3:}".format(str(Name),rows,cols,delim)
            ctorbody+= "\n\t this->{0:}={0:};".format(str(Name))
            members += "\n\t Eigen::Matrix<double,{1:},{2:}> {0:}; // {3:}".format(str(Name),rows,cols,Descr)
            i+=1
            
            
        ctor=ctorheader + typedarglist + "){" + ctorbody +"\n}\n\n"
        vflambda+=")"
        members+="\n\n"
        
        buildfunc = "static void Build(py::module & m){\n"
        buildfunc+= "\tm.def(\"{0:}\",\n".format(self.Name)
        buildfunc+="\t[]("+typedarglist +"){\n"
        buildfunc+="\t\t return GenericFunction<-1,{1:}>({0:}( {2:} ));\n".format(self.Name,osize,untypearglist)
        buildfunc+="\t},\n"
        buildfunc+="\t\"{0:}\"".format(self.docstr)
        buildfunc+=");\n}"
        
        
        return classname +  usingbase + basetypes + members+ ctor + buildfunc
        
    
    def gen_compute_impl(self):
        
        compute_impl = """
        
        template<class InType, class OutType>
        inline void compute_impl(ConstVectorBaseRef<InType> x_, ConstVectorBaseRef<OutType> fx_) const {
            
         typedef typename InType::Scalar Scalar;
         VectorBaseRef<OutType> _fx_ = fx_.const_cast_derived();
        """
        
        body = self.input_names() + "\n"
        cses,func = sp.cse(self.Func)
        body+=self.cselist(cses)
        body+="\n"
        
        for i,F in enumerate(func[0]):
            body+=self.assigment(f"_fx_[{i}]",str(sp.ccode(F)),False)
        
        return compute_impl + body + "\n\t}\n"
    
    def gen_compute_jacobian_impl(self):
        
        compute_jac_impl = """
        template<class InType, class OutType, class JacType>
        inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x_,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
                                          
         typedef typename InType::Scalar Scalar;
         VectorBaseRef<OutType> _fx_ = fx_.const_cast_derived();
         MatrixBaseRef<JacType> _jx_ = jx_.const_cast_derived();
        """
        
        
        
        body = self.input_names() + "\n"
        
        cses,funcs = sp.cse([self.Func,self.Jac])
        
        body+=self.cselist(cses)
        
        body+="\n"
        
        for i,F in enumerate(funcs[0]):
            expr = F
            body+=self.assigment(f"_fx_[{i}]",str(sp.ccode(expr)),False)
        
        
        body+="\n"

        
        for col in range(0,self.ninputs):
            for row in range(0,self.noutputs):
                expr =funcs[1][row,col]
                body+=self.assigment(f"_jx_({row},{col})",str(sp.ccode(expr)),False)
        
        
        return compute_jac_impl + body + "\n\t}\n"   
    
    def gen_compute_all(self):
        
        compute_all_impl = """
        template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x_,
        ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_,
        ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_,
        ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> _fx_ = fx_.const_cast_derived();
      MatrixBaseRef<JacType> _jx_ = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> _gx_ = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> _hx_ = adjhess_.const_cast_derived();
        """
        
        
        
        body = self.input_names() + "\n" + self.mult_names() + "\n"
        
        cses,funcs = sp.cse([self.Func,self.Jac,self.Grad,self.Hess])
        
        body+=self.cselist(cses)
        
        body+="\n"
        for i,F in enumerate(funcs[0]):
            expr = F
            body+=self.assigment(f"_fx_[{i}]",str(sp.ccode(expr)),False)
        body+="\n"
        for col in range(0,self.ninputs):
            for row in range(0,self.noutputs):
                expr = funcs[1][row,col]
                body+=self.assigment(f"_jx_({row},{col})",str(sp.ccode(expr)),False)
        body+="\n"
        for row in range(0,self.ninputs):
            expr = funcs[2][row]
            body+=self.assigment(f"_gx_[{row}]",str(sp.ccode(expr)),False)
        
        body+="\n"
        for col in range(0,self.ninputs):
            for row in range(0,self.ninputs):
                expr = funcs[3][row,col]                
                body+=self.assigment(f"_hx_({row},{col})",str(sp.ccode(expr)),False)
        
        return compute_all_impl + body + "\n\t}\n"   
    
    
    
    def assigment(self,Lhs,Rhs, Scalar = False):
        scalar = " Scalar" if Scalar else ""
        return "\n\t {2:} {0:} = {1:};".format(Lhs,self.fixup(Rhs),scalar) 
    def input_names(self):
        argnames = ""
        for i,X in enumerate(self.Inputs):
            argnames +=self.assigment(str(X),f"x_[{i}]",True)
        return argnames
    def mult_names(self):
        argnames = ""
        for i,L in enumerate(self.LMults):
            argnames +=self.assigment(str(L),f"adjvars[{i}]",True)
        return argnames
    def cselist(self,cses):
        cseassign = ""
        for cse in cses:
            expr = cse[1]
            cseassign += self.assigment(str(cse[0]),str(sp.ccode(expr)),True)
        return cseassign
    
    def print_header(self):
        
        a = self.gen_class_header()
        b = self.gen_compute_impl()
        c = self.gen_compute_jacobian_impl()
        d = self.gen_compute_all()
        e = r'};'
        
        
        print(a+b+c+d+e)
        
    def make_header(self):
        
        include = "#include \"ASSET_VectorFunctions.h\" \n namespace ASSET { \n"
        a = self.gen_class_header()
        b = self.gen_compute_impl()
        c = self.gen_compute_jacobian_impl()
        d = self.gen_compute_all()
        e = r'};}'
        
        
        text = include + a+b+c+d+e
        
        fname = self.Name +".h"
        with open(fname, "w") as text_file:
            text_file.write(text)
        

        
    
    
    
    

    
        
