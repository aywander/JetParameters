# Requires python >= 2.7 because of OrderedDict
from collections import OrderedDict

import numpy as np

import norm
import physconst as pc
import eos


class CompositionJet(eos.CompositionBase):
    """
    Currently identical class to Eos/CompositionBase
    """

    def __init__(self, mu=0.6034):
        eos.CompositionBase.__init__(self, mu)


class CompositionISM(eos.CompositionBase):
    """
    Currently identical class to Eos/CompositionBase
    """

    def __init__(self, mu=0.6034):
        eos.CompositionBase.__init__(self, mu)


class JetParameters:
    """ 
    This class contains functions to calculate parameters of a relativistic jet
    and corresponding non-relativistic jet. Everything is in cgs. If other units
    are required, PhysNorm classes should be used.
    """

    def __init__(self, power=1.e45, chi=1., pratio=1., lorentz=5, r_jet=0.01, alpha=0.0, gamma_rhd=1.3333333333333,
                 gamma_hd=1.6666666666, dens_ambient=1.0, temp_ambient=1.e7, input_param='chi',
                 norm_code=norm.PhysNorm(x=pc.kpc, v=pc.c, dens=0.6034 * pc.amu,
                                         temp=pc.c ** 2 * pc.amu / pc.kboltz, curr=1)):
        """
        By default, internally, everything is then converted into and handled in units of 

        unit density = mean mass per particle, mu*amu
        unit length = kpc
        unit speed = c
        unit temperature = c**2*amu/kboltz

        unless a user-defined norm_code parameter is given.
        To recover cgs units or other units, use norm.PhysNorm class and physconst module.

        :param power:            Jet power (erg s^-1)
        :param chi:              Jet chi
        :param pratio:           Jet to ambient medium pressure ratio
        :param lorentz:          Jet lorentz factor
        :param r_jet:             Jet radius (kpc)
        :param alpha:            Jet half opening angle (degrees)
        :param gamma_rhd:        Adiabiatic index relativistic
        :param dens_ambient:     Reference temperature of background ISM (K).
        :param temp_ambient:     Reference density of background ISM (amu*mu).
        :param gamma_hd:         Adiabiatic index non-relativistic
        :param input_param:      "chi" or "pratio", which is used as input parameters
        :param norm_code:        Normalization object for internal calculations
        """

        # Argument variables
        self.power = power
        self.chi = chi
        self.pratio = pratio
        self.lorentz = lorentz
        self.r_jet = r_jet
        self.alpha= alpha
        self.gamma_rhd = gamma_rhd
        self.dens_ambient = dens_ambient
        self.temp_ambient = temp_ambient
        self.gamma_hd = gamma_hd
        self.input_param = input_param
        self.norm_code = norm_code

        # Dictionaries where variables and their values are stored
        self.vars_cgs = OrderedDict()
        self.vars_code = OrderedDict()

        # List of input variable names (via class init arguments)
        self.input_param_vars = [
            'power',
            'lorentz',
            'r_jet',
            'gamma_rhd',
            'temp_ambient',
            'dens_ambient',
            'gamma_hd',
            input_param
        ]

        # List of variable names and their type of units
        self.defs = OrderedDict([
            ('power', ('epwr', 'Jet power')),
            ('chi', ('none', 'Jet chi')),
            ('lorentz', ('none', 'Jet lorentz factor')),
            ('r_jet', ('x', 'Jet radius')),
            ('gamma_rhd', ('none', 'Adiabiatic index, relativistic')),
            ('temp_ambient', ('temp', 'Reference temperature of background ISM.')),
            ('dens_ambient', ('dens', 'Reference density of background ISM.  ')),
            ('pres_ambient', ('pres', 'Reference pressure of background ISM.  ')),
            ('gamma_hd', ('none', 'Adiabiatic index, non-relativistic')),
            ('vel', ('v', 'Jet velocity')),
            ('beta', ('none', 'Ratio of jet speed to speed of light')),
            ('eint_ambient', ('eint', 'Ambient internal energy')),
            ('vsnd_ambient', ('v', 'Ambient sound speed')),
            ('pres', ('pres', 'Jet pressure')),
            ('dens', ('dens', 'Jet density')),
            ('temp', ('temp', 'Jet temperature')),
            ('vsnd_rhd', ('v', 'Jet sound speed, relativistic')),
            ('mach_rhd_sb', ('none', 'Jet mach number, relativistic, SB2007.')),
            ('mach_rhd_kf', ('none', 'Jet mach number, relativistic, KF1996')),
            ('mach_hd_sb', ('none', 'Jet mach number, non-relativistic, SB2007')),
            ('mach_hd_kf', ('none', 'Jet mach number, non-relativistic, KF1996.')),
            ('dens_hd_sb', ('dens', 'Equivalent density of non-relativistic jet, SB2007.')),
            ('dens_hd_kf', ('dens', 'Equivalent density of non-relativistic jet, KF19996.')),
            ('eflx', ('eflx', 'Jet energy flux, non-relativistic.')),
            ('pratio', ('none', 'Pressure ratio (jet/ISM).')),
            ('dratio', ('none', 'Density ratio (jet/ISM).')),
            ('enth_rhd', ('eint', 'Jet Specific enthalpy, relativistic.')),
            ('eint_rhd', ('eint', 'Jet Specific internal energy, relativistic.')),
            ('pflx_rhd', ('pres', 'Jet Momentum flux, relativistic.')),
            ('pflx_rhd_from_e', ('pres', 'Jet Momentum flux, relativistic, calculated from power.')),
            ('pdot_rhd', ('pdot', 'Jet Momentum injection rate, relativistic.')),
            ('pdot_rhd_from_e', ('pdot', 'Jet Momentum injection rate, relativistic  calculated from power.')),
            ('v_head', ('v', 'Jet head advance speed as calculated from ram pressure balance (Safouris et al 2008)'))
        ])

        self.defs_hd = OrderedDict([
            ('vsnd', ('v', 'Jet sound speed')),
            ('enth_hd', ('eint', 'Jet Specific enthalpy, non-relativistic.')),
            ('eint_hd', ('eint', 'Jet Specific internal energy, non-relativistic.')),
            ('pdot_hd', ('pdot', 'Jet Momentum injection rate, non-relativistic.')),
            ('pflx_hd', ('pres', 'Jet Momentum flux, non-relativistic.')),
            ('power_hd', ('edot', 'Jet power, non-relativistic.'))
        ])

        # Change all input parameters into code units here first.
        self.power = self.power / getattr(self.norm_code, self.defs['power'][0])
        self.temp_ambient = self.temp_ambient / getattr(self.norm_code, self.defs['temp_ambient'][0])
        self.alpha = np.deg2rad(alpha)

        # Create and keep a composition object for jet
        self.jc = CompositionJet()
        self.muj = self.jc.mu

        # Create and keep a composition object for ambient gas
        self.ic = CompositionISM()
        self.mua = self.ic.mu

        # Create an EOS object for jet
        self.eosj = eos.EOSIdeal(comp=self.jc, inorm=norm_code, onorm=norm_code)

        # Create an EOS object for ambient gas
        self.eosa = eos.EOSIdeal(comp=self.ic, inorm=norm_code, onorm=norm_code)


        # Create update functions for all variables, and update all 
        for var in self.defs:
            if var not in self.input_param_vars:
                setattr(self, 'upd_' + var, self.create_upd_fn(var))
                getattr(self, 'upd_' + var)()
        for var in self.defs_hd:
            if var not in self.input_param_vars:
                setattr(self, 'upd_' + var, self.create_upd_fn(var))
                getattr(self, 'upd_' + var)()

        # Create dictionaries
        self.update_all_dictionaries()

    def update_all(self):
        """ 
        Updates all attributes
        except those that have changed.
        This logic doesn't work yet, because
        I need to exclude those vars that have
        been changed. Perhaps some logic as to how to
        treat the cases where more vars than equations
        have been changed needs to be included.
        Need to at least create a dictionary of
        changed flags. 
        
        When changing attributes directly, these 
        changes need to be registered to the 
        changed-dict with a function that compares
        new and old dict. An old dict would need to
        be maintained. Setter functions could also
        be used/generated.

        You must also make sure that the decision of 
        retrieval of values from the dictionary or from 
        attributes is consistent throughout the code.

        Remember that the use of sympy needs to be
        trialed at some point.
        """
        for var in self.defs:
            if var not in self.input_param_vars:
                getattr(self, 'upd_' + var)()
        for var in self.defs_hd:
            if var not in self.input_param_vars:
                getattr(self, 'upd_' + var)()
        self.update_all_dictionaries()

    def update_all_dictionaries(self):
        self.update_dictionary_code()
        self.update_dictionary_cgs()

    def update_dictionary_code(self):

        # Create/update dictionary of all variables in code units
        for k in self.defs:
            self.vars_code[k] = getattr(self, k)
        for k in self.defs_hd:
            self.vars_code[k] = getattr(self, k)

    def update_dictionary_cgs(self):

        # Create/update dictionary of all vars in cgs
        for k in self.defs:
            self.vars_cgs[k] = getattr(self, k) * self.norm_code.scalings[self.defs[k][0]]
        for k in self.defs_hd:
            self.vars_cgs[k] = getattr(self, k) * self.norm_code.scalings[self.defs_hd[k][0]]

    def print_scalings(self):
        self.norm_code.print_scalings()

    def print_all_code(self):

        self.update_dictionary_code()
        for var, val in self.vars_code.items():
            print(format(var, '16s') + format(val, '>16.8e'))

    def print_all_cgs(self):

        self.update_dictionary_cgs()
        for var, val in self.vars_cgs.items():
            print(format(var, '16s') + format(val, '>16.8e'))

    def print_all(self):

        self.update_all_dictionaries()
        for var, val_code, val_cgs in zip(self.vars_code.keys(),
                                          self.vars_code.values(),
                                          self.vars_cgs.values()):
            print(format(var, '16s') + format(val_code, '>16.8e') + 3 * ' ' + format(val_cgs, '>16.8e'))

    def create_upd_fn(self, var):
        eqn_fn = getattr(self, 'eqn_' + var)

        def upd_fn(**kwargs):
            """ 
            This function is auto-generated. It calculates the specific variable 
            in its name and updates the respective attribute with the value. The
            function also updates the variable dictionaries.

            See equivalent eqn_ function for list of arguments
            """
            setattr(self, var, eqn_fn(**kwargs))
            return getattr(self, var)

        return upd_fn

    def _default_attributes(self, vardict):
        """
        vardict         Dictionary of variables to be set to self.<var> if <var> == None
                        Usually vardict is generated in calling function with locals()

        Note: A dictionary is mutable so this should work.
        """
        for (k, v) in vardict.items():
            if v is None: vardict[k] = getattr(self, k)

    def eqn_pres_ambient(self, dens_ambient=None, temp_ambient=None):
        """
        Ideal gas surrounding
        """
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if dens_ambient is None: dens_ambient = self.dens_ambient
        return self.eosa.pres_from_dens_temp(dens_ambient, temp_ambient)

    def eqn_eint_ambient(self, dens_ambient=None, temp_ambient=None, gamma_hd=None):
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if gamma_hd is None: gamma_hd = self.gamma_hd
        pres_ambient = self.eqn_pres_ambient(dens_ambient, temp_ambient)
        return pres_ambient / (dens_ambient * (gamma_hd - 1.))

    def eqn_vsnd_ambient(self, dens_ambient=None, temp_ambient=None, gamma_hd=None):
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if gamma_hd is None: gamma_hd = self.gamma_hd
        pres_ambient = self.eqn_pres_ambient(dens_ambient, temp_ambient)
        return np.sqrt(gamma_hd * pres_ambient / dens_ambient)

    def eqn_beta(self, lorentz=None):
        if lorentz is None: lorentz = self.lorentz
        return np.sqrt(1. - 1. / lorentz ** 2)

    def eqn_vel(self, lorentz=None):
        if lorentz is None: lorentz = self.lorentz
        return (pc.c / self.norm_code.v) * np.sqrt(1. - 1. / lorentz ** 2)

    def eqn_area(self, r_jet=None, alpha=None):
        if r_jet is None: r_jet = self.r_jet
        if alpha is None: alpha = self.alpha
        if alpha > 1.e-30:
            return 2. * np.pi * (1. - np.cos(alpha)) * pow(r_jet / np.sin(alpha), 2)
        else:
            return np.pi * r_jet * r_jet

    def eqn_pres(self, power=None, lorentz=None, chi=None, r_jet=None, alpha=None, gamma_rhd=None, pratio=None,
                 dens_ambient=None, temp_ambient=None, input_param=None):
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if alpha is None: alpha = self.alpha
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        if input_param == 'pratio':
            pres_ambient = self.eqn_pres_ambient(dens_ambient, temp_ambient)
            return pratio * pres_ambient
        elif input_param == 'chi':
            beta = self.eqn_beta(lorentz)
            area = self.eqn_area(r_jet=r_jet, alpha=alpha)
            return ((gamma_rhd - 1.) / gamma_rhd * power /
                    ((pc.c / self.norm_code.v) * area * lorentz ** 2 * beta
                     * (1. + (lorentz - 1.) / lorentz * chi)))
        else:
            raise ValueError('Unknown value for input_param, ' + input_param)

    def eqn_chi(self, power=None, lorentz=None, chi=None, r_jet=None, alpha=None, gamma_rhd=None, pratio=None,
                dens_ambient=None, temp_ambient=None, input_param=None):
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if alpha is None: alpha = self.alpha
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        if input_param == 'pratio':
            beta = self.eqn_beta(lorentz)
            area = self.eqn_area(r_jet=r_jet, alpha=alpha)
            pres = self.eqn_pres(power, lorentz, chi, r_jet, gamma_rhd=gamma_rhd, pratio=pratio,
                                 dens_ambient=dens_ambient, temp_ambient=temp_ambient, input_param=input_param)
            return lorentz / (lorentz - 1.) * ((gamma_rhd - 1.) / gamma_rhd * power / (
                (pc.c / self.norm_code.v) * pres * area * lorentz ** 2 * beta) - 1.)
        elif input_param == 'chi':
            return chi
        else:
            raise ValueError('Unknown value for input_param, ' + input_param)

    def eqn_eflx(self, power=None, r_jet=None, alpha=None):
        if power is None: power = self.power
        if r_jet is None: r_jet = self.r_jet
        area = self.eqn_area(r_jet, alpha)
        return power / area

    def eqn_dens(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                 pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Relativistic proper density
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        pres = self.eqn_pres(power, lorentz, chi, r_jet, gamma_rhd=gamma_rhd, pratio=pratio, dens_ambient=dens_ambient,
                             temp_ambient=temp_ambient, input_param=input_param)
        return chi * gamma_rhd / (gamma_rhd - 1) * pres / (pc.c / self.norm_code.v) ** 2

    def eqn_temp(self, power=None, lorentz=None, chi=None, r_jet=None,
                 gamma_rhd=None, muj=None,
                 pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if muj is None: muj = self.muj
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        pres = self.eqn_pres(power, lorentz, chi, r_jet, gamma_rhd=gamma_rhd, pratio=pratio, dens_ambient=dens_ambient,
                             temp_ambient=temp_ambient, input_param=input_param)
        dens = self.eqn_dens(power, lorentz, chi, r_jet, gamma_rhd, pratio,
                             dens_ambient, temp_ambient, input_param)
        return self.eosj.temp_from_dens_pres(dens, pres, muj)

    def eqn_vsnd(self, power=None, lorentz=None, chi=None, r_jet=None, alpha=None, gamma_rhd=None,
                 pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):

        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if alpha is None: alpha = self.alpha
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param

        pres = self.eqn_pres(power, lorentz, chi, r_jet, alpha, gamma_rhd, pratio, dens_ambient, temp_ambient, input_param)
        dens = self.eqn_dens(power, lorentz, chi, r_jet, gamma_rhd, pratio, dens_ambient, temp_ambient, input_param)
        return np.sqrt(gamma_rhd * pres / dens)

    def eqn_vsnd_rhd(self, power=None, lorentz=None, chi=None, r_jet=None, alpha=None, gamma_rhd=None, pratio=None,
                     dens_ambient=None, temp_ambient=None, input_param=None):
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param

        vsnd = self.eqn_vsnd(power, lorentz, chi, r_jet, alpha, gamma_rhd, pratio, dens_ambient, temp_ambient, input_param)
        return vsnd / np.sqrt(1. - vsnd ** 2 / (pc.c / self.norm_code.v) ** 2.)

    def eqn_mach_rhd_sb(self, lorentz=None, chi=None, gamma_rhd=None):
        """
        Relativistic Mach number 
        Expression Sutherland & Bicknell 2007, Sect. 3
        """
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        beta = self.eqn_beta(lorentz)
        return np.sqrt((2. - gamma_rhd) / (gamma_rhd - 1.) \
                       * lorentz ** 2 * beta ** 2 * (1. + chi / (2. - gamma_rhd)))

    def eqn_mach_hd_sb(self, lorentz=None, chi=None, gamma_rhd=None):
        """
        Corresponding non-relativistic Mach number
        Expression Sutherland & Bicknell 2007, Sect. 3
        """
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        return np.sqrt(2 / (gamma_rhd - 1.)
                       * (1. + chi * lorentz / (lorentz + 1.))
                       * (lorentz ** 2 - 1.))

    def eqn_mach_rhd_kf(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                        pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Relativistic Mach number
        Expression in Komissarov & Falle 1996, ASP 100, p173
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        vel = self.eqn_vel(lorentz)
        vsnd_rhd = self.eqn_vsnd_rhd(power, lorentz, chi, r_jet, gamma_rhd=gamma_rhd, pratio=pratio,
                                     dens_ambient=dens_ambient, temp_ambient=temp_ambient, input_param=input_param)
        return lorentz * vel / vsnd_rhd

    def eqn_mach_hd_kf(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                       pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Corresponding non-relativistic Mach number
        Expression in Komissarov & Falle 1996, ASP 100, p173
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        mach_rhd = self.eqn_mach_rhd_kf(power, lorentz, chi, r_jet, gamma_rhd,
                                        pratio, dens_ambient, temp_ambient, input_param)
        return mach_rhd * np.sqrt(2 * (lorentz / (lorentz + 1.)
                                       + gamma_rhd / (gamma_rhd - 1.) / (4 * chi))
                                  / (1. + gamma_rhd * (2. - gamma_rhd) / (gamma_rhd - 1.) / (4 * chi)))

    def eqn_dens_hd_sb(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                       pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Equivalent jet density for non-relativistic jet that has the same jet kinetic 
        power as that of a relativistic jet with the same velocity and pressure.
        Expression from S&B 2007. Doesn't work in the limit lorentz->1 and gamma=5/3
        though gamma=4/3 it works!?
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        pres = self.eqn_pres(power, lorentz, chi, r_jet, gamma_rhd=gamma_rhd, pratio=pratio, dens_ambient=dens_ambient,
                             temp_ambient=temp_ambient, input_param=input_param)
        return 2 * gamma_rhd / (gamma_rhd - 1.) * pres / (pc.c / self.norm_code.v) ** 2 \
               * lorentz ** 2 * (1. + chi * lorentz / (lorentz + 1.))

    def eqn_dens_hd_kf(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None):
        """
        Equivalent jet density for non-relativistic jet assuming same power,
        velocity and pressure. Expression from Komissarov & Falle (1996).
        This works.
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        dens = self.eqn_dens(power, lorentz, chi, r_jet, gamma_rhd)
        return 2 * dens * lorentz ** 2 * (lorentz / (lorentz + 1.)
                                          + gamma_rhd / (gamma_rhd - 1.) / (4 * chi))

    def eqn_power_rhd(self, pres, lorentz=None, chi=None, r_jet=None, alpha=None, gamma_rhd=None):
        """
        Jet power as calculated from relativistic parameters
        :param alpha:
        """

        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        beta = self.eqn_beta(lorentz)
        area = self.eqn_area(r_jet, alpha)
        return gamma_rhd / (gamma_rhd - 1.) * pc.c / self.norm_code.v * pres * area * lorentz ** 2 * beta \
               * (1. + (lorentz - 1.) / lorentz * chi)

    def eqn_eflx_rhd(self, lorentz=None, chi=None, r_jet=None, alpha=None, pres=None, gamma_rhd=None):
        """
        Jet energy flux as calculated from relativistic parameters
        :param alpha:
        """
        if pres is None: pres = self.pres
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        area = self.eqn_area(r_jet, alpha)
        power = self.eqn_power_rhd(lorentz, chi, r_jet, pres=pres, gamma_rhd=gamma_rhd)
        return power / area

    def eqn_mach(self, dens=None, pres=None, vel=None, gamma_rhd=None):
        """
        Non-relativistic mach number from non-relativistic primitives
        """
        if pres is None: pres = self.pres
        if dens is None: dens = self.dens
        if vel is None: vel = self.vel
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        vsnd = self.eqn_vsnd()
        return vel / vsnd

    def eqn_power_hd(self, vel=None, r_jet=None, alpha=None, dens=None, pres=None, gamma_rhd=None):
        """
        Jet power as calculated from non-relativistic parameters
        :param alpha:
        """
        if pres is None: pres = self.pres
        if dens is None: dens = self.dens
        if vel is None: vel = self.vel
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        area = self.eqn_area(r_jet, alpha)
        mach = self.eqn_mach(dens, pres, vel, gamma_rhd)
        return gamma_rhd / (gamma_rhd - 1.) * pres * vel * area \
               * (1. + (gamma_rhd - 1.) / 2 * mach ** 2)

    def eqn_eflx_hd(self, vel=None, r_jet=None, alpha=None, dens=None, pres=None, gamma_rhd=None):
        """
        Jet energy flux as calculated from non-relativistic parameters
        :param alpha:
        """
        if pres is None: pres = self.pres
        if dens is None: dens = self.dens
        if vel is None: vel = self.vel
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        area = self.eqn_area(r_jet, alpha)
        power_hd = self.eqn_power_hd(vel, r_jet, dens=dens, pres=pres, gamma_rhd=gamma_rhd)
        return power_hd / area

    def eqn_pratio(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                   pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Pressure ratios (jet/ism)
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        if input_param == 'pratio':
            return pratio
        elif input_param == 'chi':
            pres = self.eqn_pres(power, lorentz, chi, r_jet, gamma_rhd=gamma_rhd, pratio=pratio,
                                 dens_ambient=dens_ambient, temp_ambient=temp_ambient, input_param=input_param)
            pres_ambient = self.eqn_pres_ambient(dens_ambient, temp_ambient)
            return pres / pres_ambient
        else:
            raise ValueError('Unknonw value for input_param, ' + input_param)

    def eqn_dratio(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                   pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Density ratios (jet/ism)
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        dens = self.eqn_dens(power, lorentz, chi, r_jet, gamma_rhd, pratio,
                             dens_ambient, temp_ambient, input_param)
        return dens / dens_ambient

    def eqn_enth_rhd(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                     pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Specific relativistic enthalpy
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        pres = self.eqn_pres(power, lorentz, chi, r_jet, gamma_rhd=gamma_rhd, pratio=pratio, dens_ambient=dens_ambient,
                             temp_ambient=temp_ambient, input_param=input_param)
        dens = self.eqn_dens(power, lorentz, chi, r_jet, gamma_rhd, pratio,
                             dens_ambient, temp_ambient, input_param)
        return (pc.c / self.norm_code.v) ** 2 + gamma_rhd * pres / (dens * (gamma_rhd - 1.))

    def eqn_enth_hd(self, dens=None, pres=None, gamma_hd=None):
        """
        Specific enthalpy
        """
        if pres is None: pres = self.pres
        if dens is None: dens = self.dens
        if gamma_hd is None: gamma_hd = self.gamma_hd
        return gamma_hd * pres / (dens * (gamma_hd - 1.))

    def eqn_eint_rhd(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                     pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Relativistic jet internal energy density
        Not sure if expression is consistent with that used in PLUTO.
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        pres = self.eqn_pres(power, lorentz, chi, r_jet, gamma_rhd=gamma_rhd, pratio=pratio, dens_ambient=dens_ambient,
                             temp_ambient=temp_ambient, input_param=input_param)
        dens = self.eqn_dens(power, lorentz, chi, r_jet, gamma_rhd, pratio,
                             dens_ambient, temp_ambient, input_param)
        return pres / (dens * (gamma_rhd - 1.))

    def eqn_eint_hd(self, dens=None, pres=None, gamma_hd=None):
        """
        Non-relativistic jet internal energy density
        """
        if pres is None: pres = self.pres
        if dens is None: dens = self.dens
        if gamma_hd is None: gamma_hd = self.gamma_hd
        return pres / (dens * (gamma_hd - 1.))

    def eqn_pflx_rhd(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                     pratio=None, dens_ambient=None, temp_ambient=None, input_param=None):
        """
        Relativistic jet momentum flux
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_param is None: input_param = self.input_param
        beta = self.eqn_beta(lorentz)
        enth = self.eqn_enth_rhd(power, lorentz, chi, r_jet, gamma_rhd,
                                 pratio, dens_ambient, temp_ambient, input_param)
        dens = self.eqn_dens(power, lorentz, chi, r_jet, gamma_rhd, pratio,
                             dens_ambient, temp_ambient, input_param)

        return beta ** 2 * lorentz ** 2 * dens * enth

    def eqn_pflx_rhd_from_e(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None):
        """
        A close estimate of the momentum flux with  pflux = eflux/c
        """
        if power is None: power = self.power
        if lorentz is None:
            pass
        if chi is None:
            pass
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None:
            pass
        return self.eqn_eflx(power, r_jet) / (pc.c / self.norm_code.v)

    def eqn_pdot_rhd(self, power=None, lorentz=None, chi=None, r_jet=None, alpha=None, gamma_rhd=None):
        """
        Momentum injection rate
        :param alpha:
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        area = self.eqn_area(r_jet, alpha)
        return self.eqn_pflx_rhd_from_e(power, lorentz, chi, r_jet, gamma_rhd) * area

    def eqn_pdot_rhd_from_e(self, power=None):
        """
        A close estimate of the momentum injection rate with  pdot = power/c
        """
        if power is None: power = self.power
        return power / (pc.c / self.norm_code.v)

    def eqn_pflx_hd(self, dens=None, vel=None):
        """
        Non-relativistic jet momentum flux
        """
        if dens is None: dens = self.dens
        if vel is None: vel = self.vel
        return dens * vel

    def eqn_pdot_hd(self, dens=None, vel=None, r_jet=None, alpha=None):
        """
        Non-relativistic jet momentum injection rate
        :param alpha:
        """
        if dens is None: dens = self.dens
        if vel is None: vel = self.vel
        if r_jet is None: r_jet = self.r_jet
        area = self.eqn_area(r_jet, alpha)
        return self.eqn_pflx_hd(dens, vel) * area

    def eqn_v_head(self, power=None, lorentz=None, chi=None, r_jet=None, gamma_rhd=None,
                   pratio=None, dens_ambient=None, temp_ambient=None, input_refactor=None):
        """
        A close estimate of the momentum flux with pflux = eflux/c
        """
        if power is None: power = self.power
        if lorentz is None: lorentz = self.lorentz
        if chi is None: chi = self.chi
        if r_jet is None: r_jet = self.r_jet
        if gamma_rhd is None: gamma_rhd = self.gamma_rhd
        if dens_ambient is None: dens_ambient = self.dens_ambient
        if temp_ambient is None: temp_ambient = self.temp_ambient
        if pratio is None: pratio = self.pratio
        if input_refactor is None: input_refactor = self.input_param

        # Jet speed
        vjet = self.eqn_vel(lorentz)

        # Ratio of jet density to ISM density
        zeta = self.eqn_dratio(power, lorentz, chi, r_jet, gamma_rhd,
                               pratio, dens_ambient, temp_ambient, input_refactor)

        return vjet / (1. + 1. / lorentz * np.sqrt(chi / ((1 + chi) * zeta)))


    


