import jet_parameters as jp
import physconst as pc
import norm

p = jp.JetParameters(
    power=1.2e41,
    chi=1.6,
    pratio=1.0,
    lorentz=5.0,
    r_jet=0.3,
    alpha=30,
    gamma_rhd=1.33333333333,
    dens_ambient=0.1,
    temp_ambient=1.e7,
    gamma_hd=1.6666666666,
    input_param='chi',
    norm_code=norm.PhysNorm(x=pc.pc, v=pc.c, dens=0.6034 * pc.amu,
                            temp=pc.c ** 2 * pc.amu / pc.kboltz, curr=1)
)

p.print_all()
# p.print_scalings()
