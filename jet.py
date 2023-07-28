import jet_parameters as jp
import physconst as pc
import norm

p = jp.JetParameters(
    power=1.0e45,             # cgs
    chi=0.69444444,           # dimensionless
    pratio=1.0,               # dimensionless
    lorentz=5.0,              # dimensionless
    r_jet=40.,               # code units
    alpha=0,                  # degrees
    gamma_rhd=1.33333333,     # dimensionless
    dens_ambient=0.3,         # code units
    temp_ambient=1.e7,        # Kelvin
    gamma_hd=1.6666666666,    # dimensionless
    input_param='chi',        # char

    # Define code units
    norm_code=norm.PhysNorm(x=pc.kpc, v=pc.c, dens=0.6034 * pc.amu,
                            temp=pc.c ** 2 * pc.amu / pc.kboltz, curr=1)
)

p.print_all()
# p.print_scalings()
