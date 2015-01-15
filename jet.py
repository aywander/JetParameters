
import jet_parameters as jp
import physconst as pc
import norm

p = jp.JetParams(
    power=1.e45,
    chi=1.,
    pratio=1.,
    lorentz=5.,
    rjet=0.01,
    gamma_RHD=1.6666666666,
    dens_ambient=1.0,
    temp_ambient=1.e7,
    gamma_HD=1.6666666666,
    input='chi',
    norm=norm.PhysNorm(x=pc.kpc, v=pc.c, dens=0.6165*pc.amu,
                       temp=pc.c**2*pc.amu/pc.kboltz, curr=1)
)


p.print_all()
p.print_scalings()

