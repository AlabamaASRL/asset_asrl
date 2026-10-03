function calceph_compilemex()
     x = exist ('OCTAVE_VERSION', 'builtin');
     cd (fileparts(which('calceph_compilemex')))
     if (x)
        mkoctfile --mex interfacemex.c -I'../../../include' -L'../../../lib' -lcalceph
     else
        mex interfacemex.c -I'../../../include' -L'../../../lib' -lcalceph
     end

end
