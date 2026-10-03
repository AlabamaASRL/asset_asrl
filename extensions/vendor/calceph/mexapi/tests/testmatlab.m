addpath('../src/')
mex -v -I../../src/ ../src/interfacemex.c -L../../src/.libs/ -lcalceph
success=true;
'test...getorientrecordindex.m'
assert(getorientrecordindex()==1)
'test...getpositionrecordindex.m'
assert(getpositionrecordindex()==1)
'test...maxsupportedderivativeorder.m'
assert (maxsupportedderivativeorder()==1)
'test...mcheck.m'
assert (mcheck(0)==1)
'test...mcheck.m'
assert (mcheck(1)==1)
'test...mcomputearray.m'
assert(mcomputearray(0)==1)
'test...mcomputeorder01.m'
assert(mcomputeorder01('../../examples/example1.dat',0)==1)
'test...mcomputeorder01.m'
assert(mcomputeorder01('../../examples/example1.dat',1)==1)
'test...mcomputeorder01.m'
assert(mcomputeorder01(cellstr({'../../examples/example1.bsp', '../../examples/example1.tpc', '../../examples/example1.tf', '../../examples/example1.bpc', '../../examples/example1spk_time.bsp'}), 0)==1)
'test...mcomputeorder01.m'
assert(mcomputeorder01(cellstr({'../../examples/example1.bsp', '../../examples/example1.tpc', '../../examples/example1.tf', '../../examples/example1.bpc', '../../examples/example1spk_time.bsp'}), 1)==1)
'test...mcomputeorder23.m'
assert(mcomputeorder23('../../examples/example1.dat',0)==1)
'test...mcomputeorderarray.m'
assert(mcomputeorderarray(0)==1)
'test...mcomputeunit.m'
assert(mcomputeunit('../../examples/example1.dat',0)==1)
'test...mcomputeunit.m'
assert(mcomputeunit('../../examples/example1.dat',1)==1)
'test...mcomputeunit.m'
assert(mcomputeunit(cellstr({'../../examples/example1.bsp', '../../examples/example1.tpc', '../../examples/example1.tf', '../../examples/example1.bpc', '../../examples/example1spk_time.bsp'}), 0)==1)
'test...mcomputeunit.m'
assert(mcomputeunit(cellstr({'../../examples/example1.bsp', '../../examples/example1.tpc', '../../examples/example1.tf', '../../examples/example1.bpc', '../../examples/example1spk_time.bsp'}), 1)==1)
'test...mcomputeunitarray.m'
assert(mcomputeunitarray(0)==1)
'test...mcomputeunitfail.m'
assert (mcomputeunitfail()==1)
'test...mcomputeunitnaifid.m'
assert(mcomputeunitnaifid('../../examples/example1.dat',0)==1)
'test...mcomputeunitnaifid.m'
assert(mcomputeunitnaifid('../../examples/example1.dat',1)==1)
'test...mcomputeunitnaifid.m'
assert(mcomputeunitnaifid(cellstr({'../../examples/example1.bsp', '../../examples/example1.tpc', '../../examples/example1.tf', '../../examples/example1.bpc', '../../examples/example1spk_time.bsp'}), 0)==1)
'test...mcomputeunitnaifid.m'
assert(mcomputeunitnaifid(cellstr({'../../examples/example1.bsp', '../../examples/example1.tpc', '../../examples/example1.tf', '../../examples/example1.bpc', '../../examples/example1spk_time.bsp'}), 1)==1)
'test...mgetconstant.m'
assert (mgetconstant()==1)
'test...mgetconstantindex.m'
assert (mgetconstantindex()==1)
'test...mgetconstantvd.m'
assert (mgetconstantvd()==1)
'test...mgetconstantvs.m'
assert (mgetconstantvs()==1)
'test...mgetidbyname.m'
assert (mgetidbyname()==1)
'test...mgetnamebyidss.m'
assert (mgetnamebyidss()==1)
'test...morientorderarray.m'
assert(morientorderarray(0)==1)
'test...morientunitarray.m'
assert(morientunitarray(0)==1)
'test...morientunitfail.m'
assert (morientunitfail()==0)
'test...mrotangmomorder0array.m'
assert(mrotangmomorder0array(0)==1)
'test...mrotangmomunitarray.m'
assert(mrotangmomunitarray(0)==1)
'test...rotangmomorder0.m'
assert(rotangmomorder0('../../examples/example2_rotangmom.dat',0)==1)
'test...rotangmomorder0.m'
assert(rotangmomorder0('../../examples/example2_rotangmom.dat',1)==1)
'test...rotangmomunit.m'
assert(rotangmomunit('../../examples/example2_rotangmom.dat',0)==1)
'test...rotangmomunit.m'
assert(rotangmomunit('../../examples/example2_rotangmom.dat',1)==1)
'test...terror.m'
assert(terror()==1)
'test...tfileversion.m'
assert (tfileversion()==1)
'test...tgetfov.m'
assert (tgetfov()==1)
'test...tgetversionstr.m'
assert (tgetversionstr())
'test...tisthreadsafe.m'
assert (tisthreadsafe()==1)
'test...topen1.m'
assert (topen1()==1)
'test...topen2.m'
assert (topen2()==1)
'test...topen3.m'
assert (topen3()==1)
'test...topenfail.m'
assert (topen1()==1)
'test...ttime_cal_to_jd.m'
assert (ttime_cal_to_jd()==1)
'test...ttime_cal_utc_to_jd_tdb.m'
assert (ttime_cal_utc_to_jd_tdb()==1)
'test...ttime_jd_tdb_to_cal_utc.m'
assert (ttime_jd_tdb_to_cal_utc()==1)
'test...ttime_jd_to_cal.m'
assert (ttime_jd_to_cal()==1)
'test...ttime_jd_tt_to_jd_tdb.m'
assert (ttime_jd_tt_to_jd_tdb()==1)
'test...ttime_jd_tcb_to_jd_tdb.m'
assert (ttime_jd_tcb_to_jd_tdb()==1)
'test...ttime_str_any_to_jd_tdb.m'
assert (ttime_str_any_to_jd_tdb()==1)
'test...ttime_str_to_jd.m'
assert (ttime_str_to_jd()==1)
'test...ttime_str_utc_to_jd_tdb.m'
assert (ttime_str_utc_to_jd_tdb()==1)
'test...ttime_str_spacecraft_clock_to_jd_tdb.m'
assert (ttime_str_spacecraft_clock_to_jd_tdb()==1)
'test...ttime_jd_tdb_to_str_spacecraft_clock.m'
assert (ttime_jd_tdb_to_str_spacecraft_clock()==1)
'test...ttimescale.m'
assert (ttimescale()==1)
'test...ttimespan.m'
assert (ttimespan()==1)
if (success==true)
disp('all tests are OK')
%    exit(0)
else
disp('some tests are BAD')
%    exit(1)
end   
