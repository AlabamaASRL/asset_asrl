% /*-----------------------------------------------------------------*/
% /*! 
%   \file rotangmomunit.m
%  \brief Check the output of the function calceph_rotangmom_unit.
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
function res = rotangmomunit(filear, prefetch)
        res = 1;
        peph = CalcephBin.open(openfiles(filear));
        if (prefetch==0)
            peph.prefetch();
        end    
        
        datacheck = dlmread(openfiles('../../tests/example2_rotangmom_tests.dat'));
        
        for p=1:size(datacheck, 1)
            aline = datacheck(p,:);
            jd0 = aline(1);
            target =  round(aline(2));
            center = round(aline(3));
            PVcheck = [0,0,0];
            for k=1:3
                PVcheck(k) = aline(k+3);
            end    
            dt = jd0 - round(jd0);
            jd0 = (round(jd0)) + 2.4515450000000000000E+06;
 
            seuil = 3E-10;
            %/* check DAY */
            unit = Constants.UNIT_DAY;
            if (target>=100)
                unit = unit + Constants.USE_NAIFID;
            end
            
            n = 3;
            PV = peph.rotangmom_unit(jd0, dt, target, unit);
            for j=1:n
                if (abs(PVcheck(j) - PV(j)) >= seuil)
                    res  = 0;
                    format longEng
                    printf('failure DAY for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                    aline
                    PVcheck
                    PV
                    diff= PVcheck-PV
                    printf('prefetch=%d\n', prefetch);
                    error('Internal failure')
                end
            end

            %/* check SEC */
            unit = Constants.UNIT_SEC;
            if (target>=100)
                unit = unit + Constants.USE_NAIFID;
            end
            
            PV = peph.rotangmom_unit(jd0, dt, target, unit);
            for j=1:n
                PVcheck(j) = PVcheck(j)/86400E0;
            end
            
            for j=1:n
                if (abs(PVcheck(j) - PV(j)) >= seuil)
                    res  = 0;
                    format longEng
                    printf('failure DAY for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
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
   
%!assert(rotangmomunit('../../examples/example2_rotangmom.dat',0)==1)
%!assert(rotangmomunit('../../examples/example2_rotangmom.dat',1)==1)
