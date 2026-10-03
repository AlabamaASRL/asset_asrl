% /*-----------------------------------------------------------------*/
% /*! 
%   \file ttime_cal_to_jd.m
%   \brief Check if calceph_time_cal_to_jd works.
% 
%   \author  M. Gastineau 
%            Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 
% 
%    Copyright, 2025, CNRS
%    email of the author : Mickael.Gastineau@obspm.fr
% */
% /*-----------------------------------------------------------------*/
%  
% /*-----------------------------------------------------------------*/
% /* License  of this file :
%  This file is "triple-licensed", you have to choose one  of the three licenses 
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
%  at the following URL "http://www.cecill.info". 
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
function res = ttime_cal_to_jd() 
        
        peph = CalcephBin.open(openfiles('../../tests/example_lsk.tls'));
        [jd0, jdfrac] = peph.time_cal_to_jd(Constants.TT, 2016, 2, 18, 14, 50, 26.2);
        res1 = check_jd(2457437.0, 0.118358796299, jd0, jdfrac)
        [jd0, jdfrac] = peph.time_cal_to_jd(Constants.UTC, 2000, 1, 12, 13, 5, 35.0);
        res3 = check_jd(2451556.0, 0.04554398148148146, jd0, jdfrac)
        res = 1-(res1+res3)
        peph.close();
end


% check the result for the julian day
function res = check_jd(jd0e, jdfrace, jd0c, jdfracc)
        res = 0
        if (abs((jd0c-jd0e)+(jdfracc-jdfrace))>1E-6)
            printf('expected date: %23.16E %23.16E\n', jd0c, jdfracc);
            printf('computed date: %23.16E %23.16E\n', jd0e, jdfrace);
            res = 1;
            error("invalid julian date")
        end
end


%!assert (ttime_cal_to_jd()==1)
 
