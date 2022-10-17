import jet_parameters as jp
import physconst as pc
import norm

p = jp.JetParameters(
    power=1.2e44,
    chi=1.6,
    pratio=1.0,
    lorentz=1.01,
    r_jet=0.3,
    alpha=15,
    gamma_rhd=5.,
    dens_ambient=0.1,
    temp_ambient=3.e6,
    gamma_hd=1.6666666666,
    input_param='chi',
    norm_code=norm.PhysNorm(x=pc.pc, v=pc.c, dens=0.6034 * pc.amu,
                            temp=pc.c ** 2 * pc.amu / pc.kboltz, curr=1)
)

print("pratio = " + str(p.eqn_pratio()))
print("dratio = " + str(p.eqn_dratio()))

p.print_all()
# p.print_scalings()
