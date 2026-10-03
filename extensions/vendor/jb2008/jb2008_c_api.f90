! ASSET C ABI for the public JB2008 integration-form reference.
! JB2008.f is kept byte-identical; this wrapper carries no model equations.
subroutine asset_jb2008(amjd, sun_ra, sun_dec, sat_ra, sat_lat, sat_alt, &
                        f10, f10b, s10, s10b, xm10, xm10b, y10, y10b, &
                        dstdtc, tinf, tlocal, rho) bind(C, name="asset_jb2008")
  use iso_c_binding, only: c_double
  implicit none
  real(c_double), value :: amjd, sun_ra, sun_dec, sat_ra, sat_lat, sat_alt
  real(c_double), value :: f10, f10b, s10, s10b, xm10, xm10b
  real(c_double), value :: y10, y10b, dstdtc
  real(c_double), intent(out) :: tinf, tlocal, rho
  real(c_double) :: sun(2), sat(3), temp(2)

  sun = [sun_ra, sun_dec]
  sat = [sat_ra, sat_lat, sat_alt]
  call JB2008(amjd, sun, sat, f10, f10b, s10, s10b, xm10, xm10b, &
              y10, y10b, dstdtc, temp, rho)
  tinf = temp(1)
  tlocal = temp(2)
end subroutine asset_jb2008
