% /*-----------------------------------------------------------------*/
% /*! 
%   \file mcomputeorder23.m
%  \brief Check the output of the function calceph_compute_order for the order 2 and 3.
% 
%   \author  M. Gastineau 
%            Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
% 
%    Copyright, 2018, CNRS
%    email of the author : Mickael.Gastineau@obspm.fr
% */
% /*-----------------------------------------------------------------*/
%  
% /*-----------------------------------------------------------------*/
% /* License  of this file :
%  This file is 'triple-licensed', you have to choose one  of the three licenses 
%  below to apply on this file.
%  
%     CeCILL-C
%     	The CeCILL-C license is close to the GNU LGPL.
%     	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
%    
%  or CeCILL-B
%         The CeCILL-B license is close to the BSD.
%         (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
%   
%  or CeCILL v2.1
%       The CeCILL license is compatible with the GNU GPL.
%       ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
%  
% 
%  This library is governed by the CeCILL-C, CeCILL-B or the CeCILL license under 
%  French law and abiding by the rules of distribution of free software.  
%  You can  use, modify and/ or redistribute the software under the terms 
%  of the CeCILL-C,CeCILL-B or CeCILL license as circulated by CEA, CNRS and INRIA  
%  at the following URL 'http://www.cecill.info'. 
%  
%  As a counterpart to the access to the source code and  rights to copy,
%  modify and redistribute granted by the license, users are provided only
%  with a limited warranty  and the software's author,  the holder of the
%  economic rights,  and the successive licensors  have only  limited
%  liability. 
%  
%  In this respect, the user's attention is drawn to the risks associated
%  with loading,  using,  modifying and/or developing or reproducing the
%  software by the user in light of its specific status of free software,
%  that may mean  that it is complicated to manipulate,  and  that  also
%  therefore means  that it is reserved for developers  and  experienced
%  professionals having in-depth computer knowledge. Users are therefore
%  encouraged to load and test the software's suitability as regards their
%  requirements in conditions enabling the security of their systems and/or 
%  data to be ensured and,  more generally, to use and operate it in the 
%  same conditions as regards security. 
%  
%  The fact that you are presently reading this means that you have had
%  knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its terms.
%  */
%  /*-----------------------------------------------------------------*/

% /*-----------------------------------------------------------------*/
% /* main program */
% /*-----------------------------------------------------------------*/
function res = mcomputeorder23(filear, prefetch)
        res = 1;
        peph = CalcephBin.open(openfiles(filear));
        if (prefetch==0)
            peph.prefetch();
        end;
        
        AU = peph.getconstant('AU');
        
        datacheck = dlmread(openfilesbinary('example1_tests_23_derivative.dat'));

        for p=1:size(datacheck, 1)/4
            aline = datacheck(p,:);
            jd0 = aline(1);
            dt     = aline(2);
            target =  round(aline(3));
            center = round(aline(4));
            unit   = round(aline(5));
            order  = round(aline(6));
            ncomp  = 3*(order+1);
            PVcheck = aline(6+1:6+ncomp);
            PV = peph.compute_order(jd0, dt, target, center, unit, order);
            seuil = 1E-13;

           % /* check with references coordinates */
            for j=1:ncomp
                if (abs(PVcheck(j) - PV(j)) >= seuil)
                    res  = 0;
                    format longEng
                    printf('failure for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                    aline
                    PVcheck
                    PV
                    diff= PVcheck-PV
                    printf('prefetch=%d\n', prefetch);
                    error('Internal failure')
                end
             end   
        end            
        peph.close();
end        
   
%!assert(mcomputeorder23('../../examples/example1.dat',0)==1)
