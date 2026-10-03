% /*-----------------------------------------------------------------*/
% /*! 
%   \file mgetidbyname.m
%   \brief Check if calceph_getidbyname works.
% 
%   \author  M. Gastineau 
%            Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
% 
%    Copyright, 2023, CNRS
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
function res = mgetidbyname()
        res = 0;
        peph = CalcephBin.open(openfiles('../../examples/example1.tpc'));
        
        res = res +  check_failure(peph, "", 0);
        res = res +  check_failure(peph, "JUPITERBARYCENTER", 0);
        res = res +  check_failure(peph, "", Constants.USE_NAIFID);
        res = res +  check_failure(peph, "JUPITERBARYCENTER", Constants.USE_NAIFID);
        res = res +  check_failure(peph, "XX", 0);
        res = res +  check_failure(peph, "XX", Constants.USE_NAIFID);

        res = res +  check_valid(peph, "JUPITER BARYCENTER", 0, 5);
        res = res +  check_valid(peph, "Jupiter Barycenter", 0, 5);
        res = res +  check_valid(peph, "   Jupiter Barycenter", 0, 5);
        res = res +  check_valid(peph, "Jupiter    Barycenter", 0, 5);
        res = res +  check_valid(peph, "   Jupiter Barycenter   ", 0, 5);

        res = res +  check_valid(peph, "SATURN BARYCENTER", Constants.USE_NAIFID, 6);
        res = res +  check_valid(peph, "SaTURN Barycenter", Constants.USE_NAIFID, 6);
        res = res +  check_valid(peph, "   Saturn Barycenter", Constants.USE_NAIFID, 6);
        res = res +  check_valid(peph, "Saturn    Barycenter", Constants.USE_NAIFID, 6);
        res = res +  check_valid(peph, "   Saturn Barycenter   ", Constants.USE_NAIFID, 6);

        res = res +  check_valid(peph, "SATURN ", Constants.USE_NAIFID, 699);
        res = res +  check_valid(peph, "SaTURN    ", Constants.USE_NAIFID, 699);
        res = res +  check_valid(peph, "   Saturn", Constants.USE_NAIFID, 699);
        res = res +  check_valid(peph, "Saturn", Constants.USE_NAIFID, 699);
        res = res +  check_valid(peph, "   Saturn   ", Constants.USE_NAIFID, 699);

        res = res +  check_valid(peph, "Mercury Barycenter", 0, 1);
        res = res +  check_valid(peph, "Venus Barycenter", 0, 2);
        res = res +  check_valid(peph, " Earth ", 0, 3);
        res = res +  check_valid(peph, "Mars Barycenter", 0, 4);
        res = res +  check_valid(peph, "Saturn Barycenter", 0, 6);
        res = res +  check_valid(peph, "Uranus Barycenter", 0, 7);
        res = res +  check_valid(peph, "Neptune Barycenter", 0, 8);
        res = res +  check_valid(peph, "Pluto Barycenter", 0, 9);
        res = res +  check_valid(peph, "Moon", 0, 10);
        res = res +  check_valid(peph, "Sun", 0, 11);
        res = res +  check_valid(peph, "Solar   System   barycenter", 0, 12);
        res = res +  check_valid(peph, " Earth Moon   barycenter ", 0, 13);

        res = res +  check_valid(peph, " Earth ", Constants.USE_NAIFID, 399);
        res = res +  check_valid(peph, " Moon ", Constants.USE_NAIFID, 301);
        res = res +  check_valid(peph, "Sun", Constants.USE_NAIFID, 10);
        res = res +  check_valid(peph, "Solar   System   barycenter", Constants.USE_NAIFID, 0);
        res = res +  check_valid(peph, " Earth Moon   barycenter ", Constants.USE_NAIFID, 3);

        res = res +  check_valid(peph, "2 PALLAS", Constants.USE_NAIFID, 2000002);
        res = res +  check_valid(peph, "1 CERES", Constants.USE_NAIFID, 2000001);

        if (res>0)
            error("%d tests fail\n", res)
            res = 0;
        else
            res = 1;
        end

        peph.close();
 
end

%/*-----------------------------------------------------------------*/
%/* check the invalid name */
%/*-----------------------------------------------------------------*/
function ret = check_failure(peph, name,  unit)
    [res, id] = peph.getidbyname(name, unit);
    if (res!=0)
        printf("failure : find valid id but the name is invalid !!!\n");
        printf("          name='%s' unit=%d id=%d\n", name, unit, id);
        ret = 1;
    else
        ret = 0;
    end    
end

%/*-----------------------------------------------------------------*/
%/* check the valid name */
%/*-----------------------------------------------------------------*/
function ret = check_valid(peph, name, unit, id_expected)
    ret = 0; 
    [res, id] = peph.getidbyname(name, unit);
    if ((res==0) || (id_expected != id))
        printf("failure : find invalid id but the name is valid !!!\n");
        printf("          name='%s' unit=%d id=%d id_expected=%d\n", name, unit, id, id_expected);
        ret = 1;
    end
end



%!assert (mgetidbyname()==1)
