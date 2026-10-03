#! /bin/sh
rm -f testmatlab.m
echo "addpath('../src/')" >testmatlab.m
echo "mex -v -I../../src/ ../src/interfacemex.c -L../../src/.libs/ -lcalceph" >>testmatlab.m
echo "success=true;" >>testmatlab.m
#grep '%!' *.m | sed -e 's/%!//g' | awk -F ":"  -v q="'" '{ printf "'\''test...%s'\''\nsuccess&=%s\n", $1, $2 }' >>testmatlab.m
grep '%!' *.m | sed -e 's/%!//g' | awk -F ":"  -v q="'" '{ printf "'\''test...%s'\''\n%s\n", $1, $2 }' >>testmatlab.m
cat - >>testmatlab.m <<__EOF
if (success==true)
disp('all tests are OK')
%    exit(0)
else
disp('some tests are BAD')
%    exit(1)
end   
__EOF
