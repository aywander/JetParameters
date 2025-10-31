import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import mpl_aesth
import physconst as pc
import norm

# ______________________________________________________________________________________________________________________
# Set plotting style and unit normalization
mu = 0.6034
mpl_aesth.adjust_rcParams(style='seaborn-v0_8', use_kpfonts=False, grid=False, dark_mode=False)

# Normalization for this problem (from code to cgs units)
nm = norm.PhysNorm(x=pc.kpc, v=pc.c, dens=mu * pc.amu,
                   temp=pc.c ** 2 * pc.amu / pc.kboltz, curr=1)

# ______________________________________________________________________________________________________________________
# Parameter space searched. Note, for a given jet pressure and jet radius, only any two of these are independent.
# Also, the jet Lorentz factor, internal Mach number, and proper density parameter are related to each other.

# Note, all the below are linear values

# Jet power (erg s^-1)
pow_jet_arr = np.logspace(41, 45, 100)
pow_jet_set = 10 ** np.array([41, 41.5, 42, 42.5, 43, 43.5])

# Jet Lorentz factor
lor_jet_arr = np.logspace(np.log10(1.1), 1, 400)
lor_jet_set = [1.1, 1.3, 1.5, 2, 3, 5, 10]

# Jet chi
chi_jet_arr = np.logspace(0., 1., 400)
chi_jet_set = [1.1, 1.3, 1.5, 2, 3, 5, 10]

# Jet Mach number
mch_jet_arr = np.logspace(0, 2, 100)
mch_jet_set = [2, 3, 5, 10, 20, 30, 50]

# ______________________________________________________________________________________________________________________
# Other parameters

prs_jet = 1.        # Jet pressure normalized by ambient pressure
tmp_amb = 1.e7      # Ambient temperature in K
rho_amb = 0.1       # Ambient temperature in code units
gamma_rhd = 4./3    # Adiabatic index for fluid consisting of relativistic particles
rad_jet = 0.03      # Jet radius (code units)

# ______________________________________________________________________________________________________________________
# Functions

# Lorentz factor given internal Mach number and chi
def lor_mch_chi(mch, chi):
    return 1. + np.sqrt(mch * mch * (1. - gamma_rhd) / (gamma_rhd - 2. - chi))

# Internal Mach number given chi and Lorentz factor
def mch_chi_lor(chi, lor):
    return np.sqrt((2. + chi - gamma_rhd) * (lor * lor - 1.)/(gamma_rhd - 1.))

# Chi given Lorentz factor and internal Mach number
def chi_lor_mch(lor, mch):
    return (gamma_rhd - 1.) + (gamma_rhd - 1.) * mch * mch / (lor * lor - 1.)

# Beta from Lorentz factor
def voc_lor(lor):
    return np.sqrt(1. - 1. / (lor * lor))

def pow_lor_chi(lor, chi, prs_jet, area):
    voc = voc_lor(lor)
    gmm_inv_rhd = 1. / (gamma_rhd - 1.)
    return gmm_inv_rhd * area * gamma_rhd * voc * lor * lor * prs_jet * (1. + chi * (lor - 1.) / lor)


# ______________________________________________________________________________________________________________________
# Actual calculations

# Jet pressure in code units
prs_jet = prs_jet * tmp_amb / nm.temp * rho_amb / mu

# Jet area in code units
area_jet = rad_jet * rad_jet * np.pi

# Mach number
mch = mch_chi_lor(chi_jet_arr[:,None], lor_jet_arr[None,:])

# Jet power
pow = pow_lor_chi(lor_jet_arr[None,:], chi_jet_arr[:,None], prs_jet, area_jet)
pow *= nm.epwr

# Create the plot
fig, ax = plt.subplots()

is_extents_log = np.log10([lor_jet_arr[0], lor_jet_arr[-1], chi_jet_arr[0], chi_jet_arr[-1]])

im = ax.imshow(mch, extent=is_extents_log, origin='lower', cmap='tab20c', vmin=0, vmax=50, alpha=0.5)
ax.set_xlabel(r'$\Gamma$')
ax.set_xticks(np.log10(lor_jet_set), lor_jet_set)
ax.set_ylabel(r'$\chi$')
ax.set_yticks(np.log10(chi_jet_set), chi_jet_set)

divider_right = make_axes_locatable(ax)
caxr = divider_right.append_axes("right", size="2%", pad="2%")
cb = plt.colorbar(im, cax=caxr)
cb.set_label(r'$\mathcal{M}$')


# plt.contour(np.log10(lor_jet_arr), np.log10(chi_jet_arr), np.log10(mch), np.log10(mch_jet_set))
# cb = plt.colorbar(cax=cb.ax)
# cb.set_ticks(ticks=np.log10(mch_jet_set), labels=mch_jet_set)

cn = ax.contour(np.log10(lor_jet_arr), np.log10(chi_jet_arr), np.log10(pow), np.log10(pow_jet_set), linewidths=3)

divider_top = make_axes_locatable(ax)
caxt = divider_top.append_axes("top", size="2%", pad="2%")
cb = plt.colorbar(cn, cax=caxt, location='top')
cb.set_label(r'$\log P_\mathrm{jet}$')
cb.set_ticks(ticks=np.log10(pow_jet_set), labels=np.log10(pow_jet_set))
cb.lines[0].set_linewidth(3.0)


# Save figs and show
fig.savefig('mach_lorentz_chi.pdf', bbox_inches='tight')
fig.savefig('mach_lorentz_chi.png', dpi=300, bbox_inches='tight')
fig.savefig('mach_lorentz_chi.pdf', bbox_inches='tight')
fig.savefig('mach_lorentz_chi.png', dpi=300, bbox_inches='tight')
fig.show()
