% /*-----------------------------------------------------------------*/
% /*! 
%   \file mcomputeorder01.m
%  \brief Check the output of the function calceph_compute_order for the order 0 and 1.
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
function res = mcomputeorder01(filear, prefetch)
        res = 1;
        peph = CalcephBin.open(openfiles(filear));
        if (prefetch==0)
            peph.prefetch();
        end;
        
        AU = peph.getconstant('AU');
        
        datacheck = dlmread(openfiles('../../tests/example1_tests.dat'));
        
        for p=1:size(datacheck, 1)/4
            aline = datacheck(p,:);
            jd0 = aline(1);
            target =  round(aline(2));
            center = round(aline(3));
            PVcheck = [0,0,0,0,0,0];
            for k=1:6
                PVcheck(k) = aline(k+3);
            end    
            dt = jd0 - round(jd0);
            jd0 = (round(jd0)) + 2.4515450000000000000E+06;
            PV = peph.compute(jd0, dt, target, center);
            PV2 = PV;

            seuil = 1E-8;
            % /* for libration , return angles between 0 and 2*pi */
            if (target == 15)
                seuil = 1E-7;
                j = 3;
                pi2 = 6.283185307179586E0;
                while (PV(j) >= pi2)
                    PV(j) = PV(j) - pi2;
                end
                while (PV(j) <= 0) 
                   PV(j) = PV(j) + pi2;
                end
            end

            % /* check with references coordinates */
            n = 6;
            for j=1:6
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

            %  loop on the order
            for order=0:1

                n = 3*(order+1);
                seuil = 3E-15;
                if (target ~= 15 && target ~= 16)
                    %/* check UA/DAY */
                    unit = Constants.UNIT_AU + Constants.UNIT_DAY;
                    PVcheck = peph.compute_order(jd0, dt, target, center, unit, order);
                    if n~=size(PVcheck)
                        order
                        n
                        size(PVcheck)
                        PVcheck
                        PV
                        error('Invalid length')
                    end    
                    
                    for j=1:n
                        if (abs(PVcheck(j) - PV(j)) >= seuil)
                            format longEng
                            printf('failure AU/DAY  for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                            aline
                            PVcheck
                            PV
                            diff= PVcheck-PV
                            printf('prefetch=%d\n', prefetch);
                            error('Internal failure')
                        end
                    end        

                    % /* check UA/SEC */
                    unit = Constants.UNIT_AU + Constants.UNIT_SEC;
                    PVcheck = peph.compute_order(jd0, dt, target, center, unit, order);
                    if (n>3)
                        for j=4:6
                            PVcheck(j) = PVcheck(j)*86400E0;
                        end    
                    end
                    
                    for j=1:n
                        if (abs(PVcheck(j) - PV(j)) >= seuil)
                            format longEng
                            printf('failure AU/SEC  for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                            aline
                            PVcheck
                            PV
                            diff= PVcheck-PV
                            printf('prefetch=%d\n', prefetch);
                            error('Internal failure')
                        end
                    end        

                    %/* check KM/DAY */
                    seuil = 3E-14;
                    unit = Constants.UNIT_KM + Constants.UNIT_DAY;
                    PVcheck = peph.compute_order(jd0, dt, target, center, unit, order);
                    for j=1:n
                        PVcheck(j) = PVcheck(j)/AU;
                    end
                     
                    for j=1:n
                        if (abs(PVcheck(j) - PV(j)) >= seuil)
                            format longEng
                            printf('failure KM/DAY for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                            aline
                            PVcheck
                            PV
                            diff= PVcheck-PV
                            printf('prefetch=%d\n', prefetch);
                            error('Internal failure')
                        end
                    end        
           
                    %/* check KM/SEC */
                    seuil = 3E-14;
                    unit = Constants.UNIT_KM + Constants.UNIT_SEC;
                    PVcheck = peph.compute_order(jd0, dt, target, center, unit, order);
                    for j=1:n
                        PVcheck(j) = PVcheck(j)/AU;
                    end
                    if (n>3)
                        for j=4:n
                            PVcheck(j) = PVcheck(j)*86400E0;
                        end    
                    end
                    
                    for j=1:n
                        if (abs(PVcheck(j) - PV(j)) >= seuil)
                            format longEng
                            printf('failure KM/SEC for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                            aline
                            PVcheck
                            PV
                            diff= PVcheck-PV
                            printf('prefetch=%d\n', prefetch);
                            error('Internal failure')
                        end
                    end        
                end
                
                if (target == 15)
                    % /* check RAD/DAY */
                    unit = Constants.UNIT_RAD + Constants.UNIT_DAY;
                    PVcheck = peph.compute_order(jd0, dt, target, center, unit, order);
                    for j=1:n
                        if (abs(PVcheck(j) - PV2(j)) >= seuil)
                            format longEng
                            printf('failure RAD/DAY for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                            aline
                            PVcheck
                            PV2
                            diff= PVcheck-PV2
                            printf('prefetch=%d\n', prefetch);
                            error('Internal failure')
                        end
                    end        

                    % /* check RAD/SEC */
                    unit = Constants.UNIT_RAD + Constants.UNIT_SEC;
                    PVcheck = peph.compute_order(jd0, dt, target, center, unit, order);
                    if (n>3)
                        for j=4:n
                            PVcheck(j) = PVcheck(j)*86400E0;
                        end    
                    end
                
                    for j=1:n
                        if (abs(PVcheck(j) - PV2(j)) >= seuil)
                            format longEng
                            printf('failure RAD/SEC for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                            aline
                            PVcheck
                            PV2
                            diff= PVcheck-PV2
                            printf('prefetch=%d\n', prefetch);
                            error('Internal failure')
                        end
                    end        

                    %/* check orient RAD/DAY */
                    unit = Constants.UNIT_RAD + Constants.UNIT_DAY;
                    PVcheck = peph.orient_order(jd0, dt, target, unit, order);
                    if n~=size(PVcheck)
                        order
                        n
                        size(PVcheck)
                        error('Invalid length')
                    end
                    for j=1:n
                        if (abs(PVcheck(j) - PV2(j)) >= seuil)
                            format longEng
                            printf('failure RAD/DAY for %d %d at time %23.16E %23.16E\n',target, center, jd0, dt);
                            aline
                            PVcheck
                            PV2
                            diff= PVcheck-PV2
                            printf('prefetch=%d\n', prefetch);
                            error('Internal failure')
                        end
                    end
                end
             end
        end     
        peph.close();
    end    
   
%!assert(mcomputeorder01('../../examples/example1.dat',0)==1)
%!assert(mcomputeorder01('../../examples/example1.dat',1)==1)
%!assert(mcomputeorder01(cellstr({'../../examples/example1.bsp', '../../examples/example1.tpc', '../../examples/example1.tf', '../../examples/example1.bpc', '../../examples/example1spk_time.bsp'}), 0)==1)
%!assert(mcomputeorder01(cellstr({'../../examples/example1.bsp', '../../examples/example1.tpc', '../../examples/example1.tf', '../../examples/example1.bpc', '../../examples/example1spk_time.bsp'}), 1)==1)

