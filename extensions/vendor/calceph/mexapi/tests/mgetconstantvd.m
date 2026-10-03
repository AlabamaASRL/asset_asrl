% /*-----------------------------------------------------------------*/
% /*! 
%   \file mgetconstantvd.m
%   \brief Check if calceph_getconstantvd works.
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
function res = mgetconstantvd()
        res = 1;
        peph = CalcephBin.open(openfiles('../../tests/checktpc_11627.tpc'));

        v1 = peph.getconstantsd('BODY000_GMLIST1');
        v1_ref = 699.;
        if (~isequal(v1,v1_ref))
             print(isequal(v1,v1_ref))
             print(v1)
             print(v1_ref)
             error('The test BODY000_GMLIST1 fails');
        end     

        v1 = peph.getconstantvd('BODY000_GMLIST1');
        v1_ref = [699.];
        if (~isequal(v1,v1_ref))
             print(isequal(v1,v1_ref))
             print(v1)
             print(v1_ref)
             error('The test BODY000_GMLIST1 fails');
        end     
      
        v2 = peph.getconstantvd('BODY000_GMLIST2');
        v2_ref = [499.,599.];
        if (~isequal(v2,v2_ref))
             print(v2)
             error('The test BODY000_GMLIST2 fails')
        end
         
        v4 = peph.getconstantvd('BODY000_GMLIST4');
        v4_ref = [199,299,301,399];
        if (~isequal(v4,v4_ref))
             print(v4)
             error('The test BODY000_GMLIST4 fails')
        end
        peph.close();
 
end
     
%!assert (mgetconstantvd()==1)
