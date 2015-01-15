# Requires python >= 2.7 because of OrderedDict
import physconst as pc
import numpy as np
import norm
import eos
from collections import OrderedDict


class CompositionJet(eos.CompositionBase):
    '''
    Currently identical class to Eos/CompositionBase
    '''
    def __init__(self, mu=0.6165):
        eos.CompositionBase.__init__(self, mu)


class CompositionISM(eos.CompositionBase):
    '''
    Currently identical class to Eos/CompositionBase
    '''
    def __init__(self, mu=0.6165):
        eos.CompositionBase.__init__(self, mu)

 
class JetParams():
    ''' 
    This class contains functions to calculate parameters of a relativistic jet
    and corresponding non-relativistic jet. Everything is in cgs. If other units
    are required, PhysNorm classes should be used.
    '''
    def __init__(self,
                 power=1.e45,
                 chi=1.,
                 pratio=1.,
                 lorentz=5,
                 rjet=0.01,
                 gamma_RHD=1.6666666666,
                 dens_ambient=1.0,
                 temp_ambient=1.e7,
                 gamma_HD=1.6666666666,
                 input='chi',
                 norm=norm.PhysNorm(x=pc.kpc, v=pc.c, dens=0.6165*pc.amu,
                                    temp=pc.c**2*pc.amu/pc.kboltz, curr=1)
                ):
        '''
        Parameters

          power                Jet power (erg s^-1)
          chi                  Jet chi
          lorentz              Jet lorentz factor
          rjet                 Jet radius (kpc)
          gamma_RHD            Adiabiatic index relativistic
          temp_ambient         Reference temperature of background ISM (K).
          dens_ambient         Reference density of background ISM (amu*mu). 
          gamma_HD             Adiabiatic index non-relativistic
          input                "chi" or "pratio", which is used as input
          norm                 Normalization object for internal calculations 

        By default, internally, everything is then converted into and handled in units of 

        unit density = mean mass per particle, mu*amu
        unit length = kpc
        unit speed = c
        unit temperature = c**2*amu/kboltz

        unless a user-defined norm parameter is given.
        To recover cgs units or other units, use norm.PhysNorm class and physconst module.

        '''

        # All attributes in class are stored in dictionary
        # self.__dict__ whose update function can be used to 
        # append it with a list of local variables.
        self.__dict__.update(locals()); del self.__dict__['self']

        # List of variable names that were given
        args = self.__dict__.copy()

        # List of input variable names (via class init arguments)
        self.input_vars = [
            'power',
            'lorentz',
            'rjet',
            'gamma_RHD',
            'temp_ambient',
            'dens_ambient',
            'gamma_HD',
            input
        ]

        # List of variable names and their type of units
        self.defs = OrderedDict([
            ('power'          , ('epwr', 'Jet power')),
            ('chi'            , ('none', 'Jet chi')),
            ('lorentz'        , ('none', 'Jet lorentz factor')),
            ('rjet'           , ('x'   , 'Jet radius')),
            ('gamma_RHD'      , ('none', 'Adiabiatic index, relativistic')),
            ('temp_ambient'   , ('temp', 'Reference temperature of background ISM.')),
            ('dens_ambient'   , ('dens', 'Reference density of background ISM.  ')),
            ('pres_ambient'   , ('pres', 'Reference pressure of background ISM.  ')),
            ('gamma_HD'       , ('none', 'Adiabiatic index, non-relativistic')),
            ('vel'            , ('v'   , 'Jet velocity')),
            ('beta'           , ('none', 'Ratio of jet speed to speed of light')),
            ('eint_ambient'   , ('eint', 'Ambient internal energy')),
            ('vsnd_ambient'   , ('v'   , 'Ambient sound speed')),
            ('pres'           , ('pres', 'Jet pressure')),
            ('dens'           , ('dens', 'Jet density')),
            ('temp'           , ('temp', 'Jet temperature')),
            ('vsnd_RHD'       , ('v',    'Jet sound speed, relativistic')),
            ('mach_RHD_SB'    , ('none', 'Jet mach number, relativistic, SB2007.')),
            ('mach_RHD_KF'    , ('none', 'Jet mach number, relativistic, KF1996')),
            ('mach_HD_SB'     , ('none', 'Jet mach number, non-relativistic, SB2007')),
            ('mach_HD_KF'     , ('none', 'Jet mach number, non-relativistic, KF1996.')),
            ('dens_HD_SB'     , ('dens', 'Equivalent density of non-relativistic jet, SB2007.')),
            ('dens_HD_KF'     , ('dens', 'Equivalent density of non-relativistic jet, KF19996.')),
            ('eflx'           , ('eflx', 'Jet energy flux, non-relativistic.')),
            ('pratio'         , ('none', 'Pressure ratio (jet/ISM).')),
            ('dratio'         , ('none', 'Density ratio (jet/ISM).')),
            ('enth_RHD'       , ('eint', 'Jet Specific enthalpy, relativistic.')),
            ('eint_RHD'       , ('eint', 'Jet Specific internal energy, relativistic.')),
            ('pflx_RHD'       , ('pres', 'Jet Momentum flux, relativistic.')),
            ('pflx_RHD_fromE' , ('pres', 'Jet Momentum flux, relativistic, calculated from power.')),
            ('pdot_RHD'       , ('pdot', 'Jet Momentum injection rate, relativistic.')),
            ('pdot_RHD_fromE' , ('pdot', 'Jet Momentum injection rate, relativistic  calculated from power.')),
            ('vhead'          , ('v'   , 'Jet head advance speed as calculated from ram pressure balance (Safouris et al 2008)'))
        ])

        self.defs_HD = OrderedDict([
            ('vsnd'           , ('v',     'Jet sound speed')),
            ('enth_HD'        , ('eint',  'Jet Specific enthalpy, non-relativistic.')),
            ('eint_HD'        , ('eint',  'Jet Specific internal energy, non-relativistic.')),
            ('pdot_HD'        , ('pdot',  'Jet Momentum injection rate, non-relativistic.')),
            ('pflx_HD'        , ('pres',  'Jet Momentum flux, non-relativistic.')),
            ('power_HD'       , ('edot',  'Jet power, non-relativistic.'))
        ])


        # Capture normalization object
        self.norm = norm

        # Create and keep a composition object for jet
        self.jc = CompositionJet()
        self.muj = self.jc.mu

        # Create and keep a composition object for ambient gas
        self.ic = CompositionISM()
        self.mua = self.ic.mu

        # Create an EOS object for jet
        self.eosj = eos.EOSIdeal(comp=self.jc, inorm=norm, onorm=norm)

        # Create an EOS object for ambient gas
        self.eosa = eos.EOSIdeal(comp=self.ic, inorm=norm, onorm=norm)

        # Change all input parameters into code units here first.
        args['power'] = args['power']/getattr(self.norm, self.defs['power'][0])
        args['temp_ambient'] = args['temp_ambient']/getattr(self.norm, self.defs['temp_ambient'][0])
        self.__dict__.update(args)
        self.args = args


        # Create update functions for all variables, and update all 
        for var in self.defs:
            if var not in self.input_vars:
                setattr(self, 'upd_'+var, self.create_upd_fn(var))
                getattr(self, 'upd_'+var)()
        for var in self.defs_HD:
            if var not in self.input_vars:
                setattr(self, 'upd_'+var, self.create_upd_fn(var))
                getattr(self, 'upd_'+var)()

        # Create dictionaries
        self.update_all_dictionaries()

    def update_all(self):
        """ 
        Updates all attributes
        excpet those that have changed.
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
        attributes is consistent throughtout the code.

        Remember that the use of pysym needs to be
        trialed at some point.
        """
        for var in self.defs:
            if not self.args.has_key(var): 
                getattr(self, 'upd_'+var)()
        for var in self.defs_HD:
            if not self.args.has_key(var): 
                getattr(self, 'upd_'+var)()
        self.update_all_dictionaries()

    def update_all_dictionaries(self):
        self.update_dictionary_code()
        self.update_dictionary_cgs()

    def update_dictionary_code(self):

        # Create/update dictionary of all variables in code units
        self.vars_code = OrderedDict()
        for k in self.defs: self.vars_code[k] =  getattr(self, k)
        for k in self.defs_HD: self.vars_code[k] =  getattr(self, k)

    def update_dictionary_cgs(self):

        # Create/update dictionary of all vars in cgs
        self.vars_cgs = OrderedDict()
        for k in self.defs: self.vars_cgs[k] =  getattr(self, k)*self.norm.scalings[self.defs[k][0]]
        for k in self.defs_HD: self.vars_cgs[k] = getattr(self, k)*self.norm.scalings[self.defs_HD[k][0]]


    def print_scalings(self):
        self.norm.print_scalings()

    def print_all_code(self):

        self.update_dictionary_code()
        for var, val in self.vars_code.items():
            print(format(var,'16s')+format(val,'>16.8e'))

    def print_all_cgs(self):

        self.update_dictionary_cgs()
        for var, val in self.vars_cgs.items():
            print(format(var,'16s')+format(val,'>16.8e'))

    def print_all(self):

        self.update_all_dictionaries()
        for var, val_code, val_cgs in zip(self.vars_code.keys(),
                                          self.vars_code.values(), 
                                          self.vars_cgs.values()):
            print(format(var,'16s')+format(val_code,'>16.8e')+3*' '+format(val_cgs,'>16.8e'))

    def create_upd_fn(self, var):
        eqn_fn = getattr(self, 'eqn_'+var)

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
        '''
        vardict         Dictionary of varables to be set to self.<var> if <var> == None
                        Usually vardict is generated in calling funciton with locals()

        Note: A dictionary is mutable so this should work.
        '''
        for (k, v) in vardict.items(): 
            a = 1
            if v == None: vardict[k] = getattr(self, k)

    def eqn_pres_ambient(self, dens_ambient=None, temp_ambient=None):
        '''
        Ideal gas surrounding
        '''
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if dens_ambient == None: dens_ambient = self.dens_ambient
        return self.eosa.pres_from_dens_temp(dens_ambient, temp_ambient)

    def eqn_eint_ambient(self, dens_ambient=None, temp_ambient=None, gamma_HD=None):
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if gamma_HD == None: gamma_HD = self.gamma_HD
        pres_ambient =  self.eqn_pres_ambient(dens_ambient, temp_ambient)
        return pres_ambient/(dens_ambient*(gamma_HD - 1.))

    def eqn_vsnd_ambient(self, dens_ambient=None, temp_ambient=None, gamma_HD=None):
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if gamma_HD == None: gamma_HD = self.gamma_HD
        pres_ambient =  self.eqn_pres_ambient(dens_ambient, temp_ambient)
        return np.sqrt(gamma_HD*pres_ambient/dens_ambient)

    def eqn_beta(self, lorentz=None):
        if lorentz == None: lorentz = self.lorentz
        return np.sqrt(1. - 1./lorentz**2)

    def eqn_vel(self, lorentz=None):
        if lorentz == None: lorentz = self.lorentz
        return (pc.c/self.norm.v)*np.sqrt(1. - 1./lorentz**2)

    def eqn_pres(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                 pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        if input == 'pratio':
            pres_ambient = self.eqn_pres_ambient(dens_ambient, temp_ambient)
            return pratio*pres_ambient
        elif input == 'chi':
            beta = self.eqn_beta(lorentz)
            area = rjet**2*np.pi
            return (gamma_RHD - 1.)/gamma_RHD*power/((pc.c/self.norm.v)*area*lorentz**2*beta
                                                     *(1. + (lorentz - 1.)/lorentz*chi))
        else:
            raise ValueError('Unknonw value for input, ' + input)

    def eqn_chi(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        if input == 'pratio':
            beta = self.eqn_beta(lorentz)
            area = rjet**2*np.pi
            pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, 
                            pratio, dens_ambient, temp_ambient, input)
            return lorentz/(lorentz - 1.)*((gamma_RHD - 1.)/gamma_RHD*power/(
                (pc.c/self.norm.v)*pres*area*lorentz**2*beta))
        elif input == 'chi':
            return chi
        else:
            raise ValueError('Unknonw value for input, ' + input)


    def eqn_eflx(self, power=None, rjet=None):
        if power == None: power = self.power
        if rjet == None: rjet = self.rjet
        return  power/(rjet**2*np.pi)

    def eqn_dens(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None, 
                 pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Relativistic proper density
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, 
                             pratio, dens_ambient, temp_ambient, input)
        return chi*gamma_RHD/(gamma_RHD - 1)*pres/(pc.c/self.norm.v)**2

    def eqn_temp(self, power=None, lorentz=None, chi=None, rjet=None,
                 gamma_RHD=None, muj=None,
                 pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if muj == None: muj = self.muj
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        dens = self.eqn_dens(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        return self.eosj.temp_from_dens_pres(dens, pres, muj)
    
    def eqn_vsnd(self, pres=None, dens=None, gamma_RHD=None):
        if pres == None: pres = self.pres
        if dens == None: dens = self.dens
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        return np.sqrt(gamma_RHD*pres/dens)

    def eqn_vsnd_RHD(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                     pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        dens = self.eqn_dens(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        vsnd = self.eqn_vsnd(pres, dens, gamma_RHD)
        return vsnd/np.sqrt(1. - vsnd**2/(pc.c/self.norm.v)**2.)

    def eqn_mach_RHD_SB(self, lorentz=None, chi=None, gamma_RHD=None):
        '''
        Relativistic Mach number 
        Expression Sutherland & Bicknell 2007, Sect. 3
        '''
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        beta = self.eqn_beta(lorentz)
        return np.sqrt((2. - gamma_RHD)/(gamma_RHD - 1.) \
                       *lorentz**2*beta**2*(1. + chi/(2. - gamma_RHD)))

    def eqn_mach_HD_SB(self, lorentz=None, chi=None, gamma_RHD=None):
        '''
        Corresponding non-relativistic Mach number
        Expression Sutherland & Bicknell 2007, Sect. 3
        '''
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        return np.sqrt(2/(gamma_RHD - 1.) \
                       *(1. + chi*lorentz/(lorentz + 1.)) \
                       *(lorentz**2 - 1.))

    def eqn_mach_RHD_KF(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                        pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Expression in Komissarov & Falle 1996, ASP 100, p173
        - gives different answer as that by S&B 2007
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        vel = self.eqn_vel(lorentz)
        vsnd_RHD = self.eqn_vsnd_RHD(power, lorentz, chi, rjet, gamma_RHD, 
                                     pratio, dens_ambient, temp_ambient, input)
        return lorentz*vel/vsnd_RHD

    def eqn_mach_HD_KF(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                       pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Corresponding non-relativistic Mach number
        Expression in Komissarov & Falle 1996, ASP 100, p173
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        dens = self.eqn_dens(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        mach_RHD = self.eqn_mach_RHD_KF(power, lorentz, chi, rjet, gamma_RHD,
                                        pratio, dens_ambient, temp_ambient, input)
        return mach_RHD*np.sqrt(2*(lorentz/(lorentz + 1.) \
                                   + gamma_RHD/(gamma_RHD - 1.)/(4*chi)) \
                                /(1. + gamma_RHD*(2. - gamma_RHD)/(gamma_RHD - 1.)/(4*chi)))

    def eqn_dens_HD_SB(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                       pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Equivalent jet density for non-relativistic jet that has the same jet kinetic 
        power as that of a relativistic jet with the same velocity and pressure.
        Expression from S&B 2007. Doesn't work in the limit lorentz->1 and gamma=5/3
        though gamma=4/3 it works!?
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        return 2*gamma_RHD/(gamma_RHD - 1.)*pres/(pc.c/self.norm.v)**2 \
                *lorentz**2*(1. + chi*lorentz/(lorentz + 1.))

    def eqn_dens_HD_KF(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None):
        '''
        Equivalent jet density for non-relativistic jet assuming same power,
        velocity and pressure. Expression from Komissarov & Falle (1996).
        This works.
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        dens = self.eqn_dens(power, lorentz, chi, rjet, gamma_RHD)
        return 2*dens*lorentz**2*(lorentz/(lorentz + 1.) \
                                  + gamma_RHD/(gamma_RHD - 1.)/(4*chi))

    def eqn_power_RHD(self, lorentz=None, chi=None, rjet=None, pres=None, gamma_RHD=None):
        '''
        Jet power as calculated from relativistic parameters
        '''
        if pres == None: pres = self.pres
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        beta = self.eqn_beta(lorentz)
        area = rjet**2*np.pi
        return gamma_RHD/(gamma_RHD - 1.)*pc.c/self.norm.v*pres*area*lorentz**2*beta \
                *(1. + (lorentz - 1.)/lorentz*chi)

    def eqn_eflx_RHD(self, lorentz=None, chi=None, rjet=None, pres=None, gamma_RHD=None):
        '''
        Jet energy flux as calculated from relativistic parameters
        '''
        if pres == None: pres = self.pres
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        area = rjet**2*np.pi
        power = self.eqn_power_RHD(lorentz, chi, rjet, pres, gamma_RHD)
        return power/area

    def eqn_mach(self, dens=None, pres=None, vel=None, gamma_RHD=None):
        '''
        Non-relativistic mach number from non-relativistic primitives
        '''
        if pres == None: pres = self.pres
        if dens == None: dens = self.dens
        if vel == None: vel = self.vel
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        vsnd = self.eqn_vsnd(pres, dens, gamma_RHD)
        return vel/vsnd

    def eqn_power_HD(self, vel=None, rjet=None, dens=None, pres=None, gamma_RHD=None):
        '''
        Jet power as calculated from non-relativistic parameters
        '''
        if pres == None: pres = self.pres
        if dens == None: dens = self.dens
        if vel == None: vel = self.vel
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        area = rjet**2*np.pi
        mach = self.eqn_mach(dens, pres, vel, gamma_RHD)
        return gamma_RHD/(gamma_RHD - 1.)*pres*vel*area \
                *(1. + (gamma_RHD - 1.)/2*mach**2)

    def eqn_eflx_HD(self, vel=None, rjet=None, dens=None, pres=None, gamma_RHD=None):
        '''
        Jet energy flux as calculated from non-relativistic parameters
        '''
        if pres == None: pres = self.pres
        if dens == None: dens = self.dens
        if vel == None: vel = self.vel
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        area = rjet**2*np.pi
        power_HD = self.eqn_power_HD(vel, rjet, dens, pres, gamma_RHD)
        return power_HD/area

    def eqn_pratio(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None, 
                   pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Pressure ratios (jet/ism)
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        if input == 'pratio':
            return pratio
        elif input == 'chi':
            pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD,
                                 pratio, dens_ambient, temp_ambient, input)
            pres_ambient = self.eqn_pres_ambient(dens_ambient, temp_ambient)
            return pres/pres_ambient
        else:
            raise ValueError('Unknonw value for input, ' + input)

    def eqn_dratio(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None, 
                   pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Density ratios (jet/ism)
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        dens = self.eqn_dens(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        return dens/dens_ambient

    def eqn_enth_RHD(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                     pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Specific relativistic enthalpy
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        dens = self.eqn_dens(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        return (pc.c/self.norm.v)**2 + gamma_RHD*pres/(dens*(gamma_RHD - 1.))

    def eqn_enth_HD(self, dens=None, pres=None, gamma_HD=None):
        '''
        Specific enthalpy
        '''
        if pres == None: pres = self.pres
        if dens == None: dens = self.dens
        if gamma_HD == None: gamma_HD = self.gamma_HD
        return gamma_HD*pres/(dens*(gamma_HD - 1.))

    def eqn_eint_RHD(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                     pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Relativistic jet internal energy density
        Not sure if expression is consistent with that used in PLUTO.
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        pres = self.eqn_pres(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        dens = self.eqn_dens(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)
        return pres/(dens*(gamma_RHD - 1.))

    def eqn_eint_HD(self, dens=None, pres=None, gamma_HD=None):
        '''
        Non-relativistic jet internal energy density
        '''
        if pres == None: pres = self.pres
        if dens == None: dens = self.dens
        if gamma_HD == None: gamma_HD = self.gamma_HD
        return pres/(dens*(gamma_HD - 1.))

    def eqn_pflx_RHD(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                     pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        Relativistic jet momentum flux
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input
        beta = self.eqn_beta(lorentz)
        enth = self.eqn_enth_RHD(power, lorentz, chi, rjet, gamma_RHD, 
                                 pratio, dens_ambient, temp_ambient, input)
        dens = self.eqn_dens(power, lorentz, chi, rjet, gamma_RHD, pratio,
                             dens_ambient, temp_ambient, input)

        return beta**2*lorentz**2*dens*enth

    def eqn_pflx_RHD_fromE(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None):
        '''
        A close estimate of the momentum flux with  pflux = eflux/c
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        return self.eqn_eflx(power, rjet)/(pc.c/self.norm.v)

    def eqn_pdot_RHD(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None):
        '''
        Momentum injection rate
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        area = rjet**2*np.pi
        return self.eqn_pflx_RHD_fromE(power, lorentz, chi, rjet, gamma_RHD)*area

    def eqn_pdot_RHD_fromE(self, power=None):
        '''
        A close estimate of the momentum injection rate with  pdot = power/c
        '''
        if power == None: power = self.power
        return power/(pc.c/self.norm.v)

    def eqn_pflx_HD(self, dens=None, vel=None):
        '''
        Non-relativistic jet momentum flux
        '''
        if dens == None: dens = self.dens
        if vel == None: vel = self.vel
        return dens*vel

    def eqn_pdot_HD(self, dens=None, vel=None, rjet=None):
        '''
        Non-relativistic jet momentum injection rate
        '''
        if dens == None: dens = self.dens
        if vel == None: vel = self.vel
        if rjet == None: rjet = self.rjet
        area = rjet**2*np.pi
        return self.eqn_pflx_HD(dens, vel)*area

    def eqn_vhead(self, power=None, lorentz=None, chi=None, rjet=None, gamma_RHD=None,
                  pratio=None, dens_ambient=None, temp_ambient=None, input=None):
        '''
        A close estimate of the momentum flux with  pflux = eflux/c
        '''
        if power == None: power = self.power
        if lorentz == None: lorentz = self.lorentz
        if chi == None: chi = self.chi
        if rjet == None: rjet = self.rjet
        if gamma_RHD == None: gamma_RHD = self.gamma_RHD
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if pratio == None: pratio = self.pratio
        if input == None: input = self.input

        # Jet speed
        vjet = self.eqn_vel(lorentz)

        # Ratio of jet density to ISM density
        zeta = self.eqn_dratio(power, lorentz, chi, rjet, gamma_RHD, 
                               pratio, dens_ambient, temp_ambient, input)
 
        return vjet/(1. + 1./lorentz*np.sqrt(chi/((1 + chi)*zeta)))


    


