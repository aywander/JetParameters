import jet_parameters as jp
import physconst as pc
import norm

p = jp.JetParams(
    power=1.e45,
    chi=1.0,
    pratio=2.44254679e+00,
    lorentz=5.,
    rjet=0.05,
    gamma_rhd=1.6666666666,
    dens_ambient=1.0,
    temp_ambient=1.2e7,
    gamma_hd=1.6666666666,
    input_param='pratio',
    norm_code=norm.PhysNorm(x=pc.kpc, v=pc.c, dens=0.6165 * pc.amu,
                            temp=pc.c ** 2 * pc.amu / pc.kboltz, curr=1)
)

p.print_all()
p.print_scalings()

