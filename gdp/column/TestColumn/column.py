"""Distillation column model for 2018 PSE conference"""

from __future__ import division

from pyomo.environ import (
    Block, ConcreteModel, Constraint, log, minimize, NonNegativeReals, Objective, RangeSet, Set, Var,TransformationFactory,SolverFactory,value )
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.infeasible import log_infeasible_constraints
from initialize import initialize
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.data_class import MasterProblemResult
from pyomo.opt import SolutionStatus
from pyomo.opt import TerminationCondition as tc, SolverResults
import os



def build_column(min_trays, max_trays, xD, xB,x_input, provide_init=False, init={}):
    """Builds the column model."""
    m = ConcreteModel('benzene-toluene column')
    m.comps = Set(initialize=['benzene', 'toluene'])
    min_T, max_T = 300, 400
    m.T_ref = 298.15
    max_flow = 500
    m.condens_tray = max_trays
    m.feed_tray = int(round(max_trays / 2))
    m.reboil_tray = 1
    m.distillate_purity = xD
    m.bottoms_purity = xB
    m.pvap_const = {
        'benzene': {'A': -6.98273, 'B': 1.33213, 'C': -2.62863,
                    'D': -3.33399, 'Tc': 562.2, 'Pc': 48.9},
        'toluene': {'A': -7.28607, 'B': 1.38091, 'C': -2.83433,
                    'D': -2.79168, 'Tc': 591.8, 'Pc': 41.0}}
    m.vap_Cp_const = {
        'benzene': {'A': -3.392E1, 'B': 4.739E-1, 'C': -3.017E-4,
                    'D': 7.130E-8, 'E': 0},
        'toluene': {'A': -2.435E1, 'B': 5.125E-1, 'C': -2.765E-4,
                    'D': 4.911E-8, 'E': 0}}
    m.liq_Cp_const = {
        'benzene': {'A': 1.29E5, 'B': -1.7E2, 'C': 6.48E-1,
                    'D': 0, 'E': 0},
        'toluene': {'A': 1.40E5, 'B': -1.52E2, 'C': 6.95E-1,
                    'D': 0, 'E': 0}}
    m.dH_vap = {'benzene': 33.770E3, 'toluene': 38.262E3}  # J/mol


    m.trays = RangeSet(max_trays, doc='Set of potential trays')
    m.conditional_trays = Set(
        initialize=m.trays - [m.condens_tray, m.feed_tray, m.reboil_tray],
        doc="Trays that may be turned on and off.")    
    m.tray = Disjunct(m.conditional_trays, doc='Disjunct for tray existence')
    m.no_tray = Disjunct(m.conditional_trays, doc='Disjunct for tray absence')



    @m.Disjunction(m.conditional_trays, doc='Tray exists or does not')
    def tray_no_tray(b, t):
        return [b.tray[t], b.no_tray[t]]
    m.minimum_num_trays = Constraint(
        expr=sum(m.tray[t].indicator_var
                 for t in m.conditional_trays) + 1  # for feed tray
        >= min_trays)

    if provide_init:
	    m.T_feed = Var(
		doc='Feed temperature [K]', domain=NonNegativeReals,
		bounds=(min_T, max_T), initialize=init['T_feed'])
	    m.feed_vap_frac = Var(
		doc='Vapor fraction of feed',
		initialize=init['feed_vap_frac'], bounds=(0, 1))
	    m.feed = Var(
		m.comps, doc='Total component feed flow [mol/s]',initialize=init['feed'])
	    m.x = Var(m.comps, m.trays, doc='Liquid mole fraction',
		      bounds=(0, 1), domain=NonNegativeReals, initialize=init['x'])
	    m.y = Var(m.comps, m.trays, doc='Vapor mole fraction',
		      bounds=(0, 1), domain=NonNegativeReals, initialize=init['y'])
	    m.L = Var(m.comps, m.trays,
		      doc='component liquid flows from tray in kmol',
		      domain=NonNegativeReals, bounds=(0, max_flow),
		      initialize=init['L'])
	    m.V = Var(m.comps, m.trays,
		      doc='component vapor flows from tray in kmol',
		      domain=NonNegativeReals, bounds=(0, max_flow),
		      initialize=init['V'])
	    m.liq = Var(m.trays, domain=NonNegativeReals,
		        doc='liquid flows from tray in kmol', initialize=init['liq'],
		        bounds=(0, max_flow))
	    m.vap = Var(m.trays, domain=NonNegativeReals,
		        doc='vapor flows from tray in kmol', initialize=init['vap'],
		        bounds=(0, max_flow))
	    m.B = Var(m.comps, domain=NonNegativeReals,
		      doc='bottoms component flows in kmol',
		      bounds=(0, max_flow), initialize=init['B'])
	    m.D = Var(m.comps, domain=NonNegativeReals,
		      doc='distillate component flows in kmol',
		      bounds=(0, max_flow), initialize=init['D'])
	    m.bot = Var(domain=NonNegativeReals, initialize=init['bot'], bounds=(0, 100),
		        doc='bottoms flow in kmol')
	    m.dis = Var(domain=NonNegativeReals, initialize=init['dis'],
		        doc='distillate flow in kmol', bounds=(0, 100))
	    m.reflux_ratio = Var(domain=NonNegativeReals, bounds=(0.5, 4),
		                 doc='reflux ratio', initialize=init['reflux_ratio'])
	    m.reboil_ratio = Var(domain=NonNegativeReals, bounds=(1.3, 4),
		                 doc='reboil ratio', initialize=init['reboil_ratio'])
	    m.reflux_frac = Var(domain=NonNegativeReals, bounds=(0, 1 - 1E-6),
		                doc='reflux fractions',initialize=init['reflux_frac'])
	    m.boilup_frac = Var(domain=NonNegativeReals, bounds=(0, 1 - 1E-6),
		                doc='boilup fraction',initialize=init['boilup_frac'])
	    m.Kc = Var(
		m.comps, m.trays, doc='Phase equilibrium constant',
		domain=NonNegativeReals, initialize=init['Kc'], bounds=(0, 1000))
	    m.T = Var(m.trays, doc='Temperature [K]',
		      domain=NonNegativeReals,
		      bounds=(min_T, max_T),initialize=init['T'])
	    m.P = Var(doc='Pressure [bar]',
		      bounds=(0, 5),initialize=init['P'])
	    m.gamma = Var(
		m.comps, m.trays,
		doc='liquid activity coefficent of component on tray',
		domain=NonNegativeReals, bounds=(0, 10), initialize=init['gamma'])
	    m.Pvap = Var(
		m.comps, m.trays,
		doc='pure component vapor pressure of component on tray in bar',
		domain=NonNegativeReals, bounds=(1E-3, 5), initialize=init['Pvap'])
	    m.Pvap_X = Var(
		m.comps, m.trays,
		doc='Related to fraction of critical temperature (1 - T/Tc)',
		bounds=(0.25, 0.5), initialize=init['Pvap_X'])
	    m.H_L = Var(
		m.comps, m.trays, bounds=(0.1, 16),
		doc='Liquid molar enthalpy of component in tray (kJ/mol)',initialize=init['H_L'])
	    m.H_V = Var(
		m.comps, m.trays, bounds=(30, 16 + 40),
		doc='Vapor molar enthalpy of component in tray (kJ/mol)',initialize=init['H_V'])
	    m.H_L_spec_feed = Var(
		m.comps, doc='Component liquid molar enthalpy in feed [kJ/mol]',
		initialize=init['H_L_spec_feed'], bounds=(0.1, 16))
	    m.H_V_spec_feed = Var(
		m.comps, doc='Component vapor molar enthalpy in feed [kJ/mol]',
		initialize=init['H_V_spec_feed'], bounds=(30, 16 + 40))
	    m.Qb = Var(domain=NonNegativeReals, doc='reboiler duty (MJ/s)',
		       initialize=init['Qb'], bounds=(0, 8))
	    m.Qc = Var(domain=NonNegativeReals, doc='condenser duty (MJ/s)',
		       initialize=init['Qc'], bounds=(0, 8))

    else:

	    m.T_feed = Var(
		doc='Feed temperature [K]', domain=NonNegativeReals,
		bounds=(min_T, max_T), initialize=368)
	    m.feed_vap_frac = Var(
		doc='Vapor fraction of feed',
		initialize=0, bounds=(0, 1))
	    m.feed = Var(
		m.comps, doc='Total component feed flow [mol/s]', initialize=50)
	    m.x = Var(m.comps, m.trays, doc='Liquid mole fraction',
		      bounds=(0, 1), domain=NonNegativeReals, initialize=0.5)
	    m.y = Var(m.comps, m.trays, doc='Vapor mole fraction',
		      bounds=(0, 1), domain=NonNegativeReals, initialize=0.5)
	    m.L = Var(m.comps, m.trays,
		      doc='component liquid flows from tray in kmol',
		      domain=NonNegativeReals, bounds=(0, max_flow),
		      initialize=50)
	    m.V = Var(m.comps, m.trays,
		      doc='component vapor flows from tray in kmol',
		      domain=NonNegativeReals, bounds=(0, max_flow),
		      initialize=50)
	    m.liq = Var(m.trays, domain=NonNegativeReals,
		        doc='liquid flows from tray in kmol', initialize=100,
		        bounds=(0, max_flow))
	    m.vap = Var(m.trays, domain=NonNegativeReals,
		        doc='vapor flows from tray in kmol', initialize=100,
		        bounds=(0, max_flow))
	    m.B = Var(m.comps, domain=NonNegativeReals,
		      doc='bottoms component flows in kmol',
		      bounds=(0, max_flow), initialize=50)
	    m.D = Var(m.comps, domain=NonNegativeReals,
		      doc='distillate component flows in kmol',
		      bounds=(0, max_flow), initialize=50)
	    m.bot = Var(domain=NonNegativeReals, initialize=50, bounds=(0, 100),
		        doc='bottoms flow in kmol')
	    m.dis = Var(domain=NonNegativeReals, initialize=50,
		        doc='distillate flow in kmol', bounds=(0, 100))
	    m.reflux_ratio = Var(domain=NonNegativeReals, bounds=(0.5, 4),
		                 doc='reflux ratio', initialize=1.4)
	    m.reboil_ratio = Var(domain=NonNegativeReals, bounds=(1.3, 4),
		                 doc='reboil ratio', initialize=0.9527)
	    m.reflux_frac = Var(domain=NonNegativeReals, bounds=(0, 1 - 1E-6),
		                doc='reflux fractions')
	    m.boilup_frac = Var(domain=NonNegativeReals, bounds=(0, 1 - 1E-6),
		                doc='boilup fraction')
	    m.Kc = Var(
		m.comps, m.trays, doc='Phase equilibrium constant',
		domain=NonNegativeReals, initialize=1, bounds=(0, 1000))
	    m.T = Var(m.trays, doc='Temperature [K]',
		      domain=NonNegativeReals,
		      bounds=(min_T, max_T))
	    m.P = Var(doc='Pressure [bar]',
		      bounds=(0, 5))
	    m.gamma = Var(
		m.comps, m.trays,
		doc='liquid activity coefficent of component on tray',
		domain=NonNegativeReals, bounds=(0, 10), initialize=1)
	    m.Pvap = Var(
		m.comps, m.trays,
		doc='pure component vapor pressure of component on tray in bar',
		domain=NonNegativeReals, bounds=(1E-3, 5), initialize=0.4)
	    m.Pvap_X = Var(
		m.comps, m.trays,
		doc='Related to fraction of critical temperature (1 - T/Tc)',
		bounds=(0.25, 0.5), initialize=0.4)
	    m.H_L = Var(
		m.comps, m.trays, bounds=(0.1, 16),
		doc='Liquid molar enthalpy of component in tray (kJ/mol)')
	    m.H_V = Var(
		m.comps, m.trays, bounds=(30, 16 + 40),
		doc='Vapor molar enthalpy of component in tray (kJ/mol)')
	    m.H_L_spec_feed = Var(
		m.comps, doc='Component liquid molar enthalpy in feed [kJ/mol]',
		initialize=0, bounds=(0.1, 16))
	    m.H_V_spec_feed = Var(
		m.comps, doc='Component vapor molar enthalpy in feed [kJ/mol]',
		initialize=0, bounds=(30, 16 + 40))
	    m.Qb = Var(domain=NonNegativeReals, doc='reboiler duty (MJ/s)',
		       initialize=1, bounds=(0, 8))
	    m.Qc = Var(domain=NonNegativeReals, doc='condenser duty (MJ/s)',
		       initialize=1, bounds=(0, 8))


    m.partial_cond = Disjunct()
    m.total_cond = Disjunct()
    m.condenser_choice = Disjunction(expr=[m.partial_cond, m.total_cond])

    for t in m.conditional_trays:
        _build_conditional_tray_mass_balance(m, t, m.tray[t], m.no_tray[t])
    _build_feed_tray_mass_balance(m)
    _build_condenser_mass_balance(m)
    _build_reboiler_mass_balance(m)

    @m.Constraint(m.comps,
                  doc="Bottoms flow is equal to liquid leaving reboiler.")
    def bottoms_mass_balance(m, c):
        return m.B[c] == m.L[c, m.reboil_tray]

    @m.Constraint()
    def boilup_frac_defn(m):
        return m.bot == (1 - m.boilup_frac) * m.liq[m.reboil_tray + 1]

    @m.Constraint()
    def reflux_frac_defn(m):
        return m.dis == (1 - m.reflux_frac) * (
            m.vap[m.condens_tray - 1] - m.vap[m.condens_tray])

    @m.Constraint(m.trays)
    def liquid_sum(m, t):
        return sum(m.L[c, t] for c in m.comps) == m.liq[t]

    @m.Constraint(m.trays)
    def vapor_sum(m, t):
        return sum(m.V[c, t] for c in m.comps) == m.vap[t]

    m.bottoms_sum = Constraint(
        expr=sum(m.B[c] for c in m.comps) == m.bot)
    m.distil_sum = Constraint(
        expr=sum(m.D[c] for c in m.comps) == m.dis)

    @m.Constraint(m.trays)
    def monotonoic_temperature(_, t):
        return m.T[t] >= m.T[t + 1] if t < max_trays else Constraint.Skip

    for t in m.conditional_trays:
        _build_tray_phase_equilibrium(m, t, m.tray[t])
    m.feed_tray_phase_eq = Block()
    m.reboiler_phase_eq = Block()
    m.condenser_phase_eq = Block()
    _build_tray_phase_equilibrium(m, m.feed_tray, m.feed_tray_phase_eq)
    _build_tray_phase_equilibrium(m, m.reboil_tray, m.reboiler_phase_eq)
    _build_tray_phase_equilibrium(m, m.condens_tray, m.condenser_phase_eq)
    _build_column_heat_relations(m)

    @m.Constraint()
    def distillate_req(m):
        return m.D['benzene'] >= m.distillate_purity * m.dis

    @m.Constraint()
    def bottoms_req(m):
        return m.B['toluene'] >= m.bottoms_purity * m.bot

    # m.obj = Objective(expr=(m.Qc + m.Qb) * 1E-3, sense=minimize)
    m.obj = Objective(
        expr=(m.Qc + m.Qb) * 1E3 + 1E3 * (
            sum(m.tray[t].indicator_var for t in m.conditional_trays) + 1),
        sense=minimize)
    # m.obj = Objective(
    #     expr=sum(m.tray[t].indicator_var for t in m.conditional_trays) + 1)

    @m.Constraint()
    def reflux_ratio_calc(m):
        return m.reflux_frac * (m.reflux_ratio + 1) == m.reflux_ratio

    @m.Constraint()
    def reboil_ratio_calc(m):
        return m.boilup_frac * (m.reboil_ratio + 1) == m.reboil_ratio

    @m.Constraint(m.conditional_trays)
    def tray_ordering(m, t):
        """Trays close to the feed should be activated first."""
        if t + 1 < m.condens_tray and t > m.feed_tray:
            return m.tray[t].indicator_var >= m.tray[t + 1].indicator_var
        elif t > m.reboil_tray and t + 1 < m.feed_tray:
            return m.tray[t + 1].indicator_var >= m.tray[t].indicator_var
        else:
            return Constraint.NoConstraint
# Fix feed conditions
    m.feed['benzene'].fix(50)
    m.feed['toluene'].fix(50)
    m.T_feed.fix(368)
    m.feed_vap_frac.fix(0.40395)
    m.P.fix(1.01)
    # Fix to be total condenser
    m.partial_cond.deactivate()
    m.total_cond.indicator_var.fix(1)

#-----------End of model declaration. These changes are required to run the DSDA--------------

    #FIX Indicatior variables according to input
    ext_var_1 =  x_input[0]
    ext_var_2 =  x_input[1]
    YR_fixed = {}
    YB_fixed= {}
    
    for n in m.trays - [m.condens_tray, m.reboil_tray]:
        if n == ext_var_1:
            YR_fixed[n] = 1
        else:
            YR_fixed[n] = 0
        if n == ext_var_2:
            YB_fixed[n] = 1
        else:
            YB_fixed[n] = 0
    for n in m.trays - [m.condens_tray, m.reboil_tray]:    
        temp=1-(1-sum(YR_fixed[j] for j in m.trays if j>=n and j<=max_trays-1))-(sum(YB_fixed[j] for j in m.trays if j>=n and j<=max_trays-1)-YB_fixed[n])

        if temp==1 and n!=m.feed_tray:
            m.tray[n].indicator_var.fix(True)
            m.no_tray[n].indicator_var.fix(False)
	elif temp==0 and n!=m.feed_tray:
            m.tray[n].indicator_var.fix(False)
            m.no_tray[n].indicator_var.fix(True)


    # Transform the model

    TransformationFactory('gdp.fix_disjuncts').apply_to(m)


    # Check equation feasibility
    try:
	    fbbt(m)

            # SOLVE
	    if provide_init==False:
	    	initialize(m)
	    log_infeasible_constraints(m, tol=1E-3)
	    results=SolverFactory('ipopt').solve(
		m, tee=True)

            # Save results (for initialization)

            T_feed_init,feed_vap_frac_init,feed_init,x_init,y_init,L_init,V_init,liq_init,vap_init,B_init,D_init,bot_init,dis_init,reflux_ratio_init={},{},{},{},{},{},{},{},{},{},{},{},{},{}
            reboil_ratio_init,reflux_frac_init,boilup_frac_init,Kc_init,T_init,P_init,gamma_init={},{},{},{},{},{},{}
            Pvap_init,Pvap_X_init,H_L_init,H_V_init,H_L_spec_feed_init,H_V_spec_feed_init,Qb_init,Qc_init={},{},{},{},{},{},{},{}

            T_feed_init=value(m.T_feed)
            feed_vap_frac_init=value(m.feed_vap_frac)
	    bot_init=value(m.bot)
	    dis_init=value(m.dis)
	    reflux_ratio_init=value(m.reflux_ratio)
	    reboil_ratio_init=value(m.reboil_ratio)
	    reflux_frac_init=value(m.reflux_frac)
	    boilup_frac_init=value(m.boilup_frac)
            Qb_init=value(m.Qb)
            Qc_init=value(m.Qc)
            P_init=value(m.P)

            for i in m.comps:
		feed_init[i]=value(m.feed[i])
		B_init[i]=value(m.B[i])
		D_init[i]=value(m.D[i])
		H_L_spec_feed_init[i]=value(m.H_L_spec_feed[i])
		H_V_spec_feed_init[i]=value(m.H_V_spec_feed[i])

            for n in m.trays: 
		liq_init[n]=value(m.liq[n])
		vap_init[n]=value(m.vap[n])
		T_init[n]=value(m.T[n])

            for i in m.comps:
		for n in m.trays:
			x_init[i,n]=value(m.x[i,n])
			y_init[i,n]=value(m.y[i,n])
			L_init[i,n]=value(m.L[i,n])
			V_init[i,n]=value(m.V[i,n])
			Kc_init[i,n]=value(m.Kc[i,n])
			gamma_init[i,n]=value(m.gamma[i,n])
			Pvap_init[i,n]=value(m.Pvap[i,n])
			Pvap_X_init[i,n]=value(m.Pvap_X[i,n])
			H_L_init[i,n]=value(m.H_L[i,n])
			H_V_init[i,n]=value(m.H_V[i,n])


            initialization = {'T_feed':T_feed_init,'feed_vap_frac':feed_vap_frac_init,'feed':feed_init,'x':x_init,'y':y_init,'L':L_init,'V':V_init,'liq':liq_init,'vap':vap_init,'B':B_init,'D':D_init,'bot':bot_init,'dis':dis_init,'reflux_ratio':reflux_ratio_init,'reboil_ratio':reboil_ratio_init,'reflux_frac':reflux_frac_init,'boilup_frac':boilup_frac_init,'Kc':Kc_init,'T':T_init,'P':P_init,'gamma':gamma_init,'Pvap':Pvap_init,'Pvap_X':Pvap_X_init,'H_L':H_L_init,'H_V':H_V_init,'H_L_spec_feed':H_L_spec_feed_init,'H_V_spec_feed':H_V_spec_feed_init,'Qb':Qb_init,'Qc':Qc_init}
            
            return m, results.solver.status, initialization
    

    except InfeasibleConstraintException:

            #config.logger.debug("MIP preprocessing detected infeasibility.")
            nlp_result = MasterProblemResult()
            nlp_result.feasible = False
            nlp_result.pyomo_results = SolverResults()
            nlp_result.pyomo_results.solver.termination_condition = tc.error
            print('Try an infeasible')

            return m, 'infeasible', {}



# ---------Other functions do define the model-------------------------------------------------

def _build_conditional_tray_mass_balance(m, t, tray, no_tray):
    """
    t = tray number
    tray = tray exists disjunct
    no_tray = tray absent disjunct
    """
    @tray.Constraint(m.comps)
    def mass_balance(_, c):
        return (
            # Feed in if feed tray
            (m.feed[c] if t == m.feed_tray else 0)
            # Vapor from tray t
            - m.V[c, t]
            # Loss to distillate if condenser
            - (m.D[c] if t == m.condens_tray else 0)
            # Liquid from tray above if not condenser
            + (m.L[c, t + 1] if t < m.condens_tray else 0)
            # Loss to bottoms if reboiler
            - (m.B[c] if t == m.reboil_tray else 0)
            # Liquid to tray below if not reboiler
            - (m.L[c, t] if t > m.reboil_tray else 0)
            # Vapor from tray below if not reboiler
            + (m.V[c, t - 1] if t > m.reboil_tray else 0) == 0)

    @tray.Constraint(m.comps)
    def tray_liquid_composition(_, c):
        return m.L[c, t] == m.liq[t] * m.x[c, t]

    @tray.Constraint(m.comps)
    def tray_vapor_compositions(_, c):
        return m.V[c, t] == m.vap[t] * m.y[c, t]

    @no_tray.Constraint(m.comps)
    def liq_comp_pass_through(_, c):
        return m.x[c, t] == m.x[c, t + 1]

    @no_tray.Constraint(m.comps)
    def liq_flow_pass_through(_, c):
        return m.L[c, t] == m.L[c, t + 1]

    @no_tray.Constraint(m.comps)
    def vap_comp_pass_through(_, c):
        return m.y[c, t] == m.y[c, t - 1]

    @no_tray.Constraint(m.comps)
    def vap_flow_pass_through(_, c):
        return m.V[c, t] == m.V[c, t - 1]


def _build_feed_tray_mass_balance(m):
    t = m.feed_tray

    @m.Constraint(m.comps)
    def feed_mass_balance(_, c):
        return (
            m.feed[c]        # Feed in
            - m.V[c, t]      # Vapor from tray t
            + m.L[c, t + 1]  # Liquid from tray above
            - m.L[c, t]      # Liquid to tray below
            + m.V[c, t - 1]  # Vapor from tray below
            == 0)

    @m.Constraint(m.comps)
    def feed_tray_liquid_composition(_, c):
        return m.L[c, t] == m.liq[t] * m.x[c, t]

    @m.Constraint(m.comps)
    def feed_tray_vapor_composition(_, c):
        return m.V[c, t] == m.vap[t] * m.y[c, t]


def _build_condenser_mass_balance(m):
    t = m.condens_tray

    @m.Constraint(m.comps)
    def condenser_mass_balance(_, c):
        return (
            - m.V[c, t]      # Vapor from tray t
            - m.D[c]         # Loss to distillate
            - m.L[c, t]      # Liquid to tray below
            + m.V[c, t - 1]  # Vapor from tray below
            == 0)

    @m.partial_cond.Constraint(m.comps)
    def condenser_liquid_composition(_, c):
        return m.L[c, t] == m.liq[t] * m.x[c, t]

    @m.partial_cond.Constraint(m.comps)
    def condenser_vapor_composition(_, c):
        return m.V[c, t] == m.vap[t] * m.y[c, t]

    @m.total_cond.Constraint(m.comps)
    def no_vapor_flow(_, c):
        return m.V[c, t] == 0

    @m.total_cond.Constraint()
    def no_total_vapor_flow(_):
        return m.vap[t] == 0

    @m.total_cond.Constraint(m.comps)
    def liquid_fraction_pass_through(_, c):
        return m.x[c, t] == m.y[c, t - 1]

    @m.Constraint(m.comps)
    def condenser_distillate_composition(_, c):
        return m.D[c] == m.dis * m.x[c, t]


def _build_reboiler_mass_balance(m):
    t = m.reboil_tray

    @m.Constraint(m.comps)
    def reboiler_mass_balance(_, c):
        t = m.reboil_tray
        return (
            - m.V[c, t]      # Vapor from tray t
            + m.L[c, t + 1]  # Liquid from tray above
            - m.B[c]         # Loss to bottoms
            == 0)

    @m.Constraint(m.comps)
    def reboiler_liquid_composition(_, c):
        return m.L[c, t] == m.liq[t] * m.x[c, t]

    @m.Constraint(m.comps)
    def reboiler_vapor_composition(_, c):
        return m.V[c, t] == m.vap[t] * m.y[c, t]


def _build_tray_phase_equilibrium(m, t, tray):
    @tray.Constraint(m.comps)
    def raoults_law(_, c):
        return m.y[c, t] == m.x[c, t] * m.Kc[c, t]

    @tray.Constraint(m.comps)
    def phase_equil_const(_, c):
        return m.Kc[c, t] * m.P == (
            m.gamma[c, t] * m.Pvap[c, t])

    @tray.Constraint(m.comps)
    def Pvap_relation(_, c):
        k = m.pvap_const[c]
        x = m.Pvap_X[c, t]
        return (log(m.Pvap[c, t]) - log(k['Pc'])) * (1 - x) == (
            k['A'] * x +
            k['B'] * x ** 1.5 +
            k['C'] * x ** 3 +
            k['D'] * x ** 6)

    @tray.Constraint(m.comps)
    def Pvap_X_defn(_, c):
        k = m.pvap_const[c]
        return m.Pvap_X[c, t] == 1 - m.T[t] / k['Tc']

    @tray.Constraint(m.comps)
    def gamma_calc(_, c):
        return m.gamma[c, t] == 1


def _build_column_heat_relations(m):
    @m.Expression(m.trays, m.comps)
    def liq_enthalpy_expr(_, t, c):
        k = m.liq_Cp_const[c]
        return (
            k['A'] * (m.T[t] - m.T_ref) +
            k['B'] * (m.T[t] ** 2 - m.T_ref ** 2) / 2 +
            k['C'] * (m.T[t] ** 3 - m.T_ref ** 3) / 3 +
            k['D'] * (m.T[t] ** 4 - m.T_ref ** 4) / 4 +
            k['E'] * (m.T[t] ** 5 - m.T_ref ** 5) / 5) * 1E-6

    @m.Expression(m.trays, m.comps)
    def vap_enthalpy_expr(_, t, c):
        k = m.vap_Cp_const[c]
        return (
            m.dH_vap[c] +
            k['A'] * (m.T[t] - m.T_ref) +
            k['B'] * (m.T[t] ** 2 - m.T_ref ** 2) / 2 +
            k['C'] * (m.T[t] ** 3 - m.T_ref ** 3) / 3 +
            k['D'] * (m.T[t] ** 4 - m.T_ref ** 4) / 4 +
            k['E'] * (m.T[t] ** 5 - m.T_ref ** 5) / 5) * 1E-3

    for t in m.conditional_trays:
        _build_conditional_tray_energy_balance(m, t, m.tray[t], m.no_tray[t])
    _build_feed_tray_energy_balance(m)
    _build_condenser_energy_balance(m)
    _build_reboiler_energy_balance(m)


def _build_conditional_tray_energy_balance(m, t, tray, no_tray):
    @tray.Constraint()
    def energy_balance(_):
        return sum(
            m.L[c, t + 1] * m.H_L[c, t + 1]  # heat of liquid from tray above
            - m.L[c, t] * m.H_L[c, t]  # heat of liquid to tray below
            + m.V[c, t - 1] * m.H_V[c, t - 1]  # heat of vapor from tray below
            - m.V[c, t] * m.H_V[c, t]  # heat of vapor to tray above
            for c in m.comps) * 1E-3 == 0

    @tray.Constraint(m.comps)
    def liq_enthalpy_calc(_, c):
        return m.H_L[c, t] == m.liq_enthalpy_expr[t, c]

    @tray.Constraint(m.comps)
    def vap_enthalpy_calc(_, c):
        return m.H_V[c, t] == m.vap_enthalpy_expr[t, c]

    @no_tray.Constraint(m.comps)
    def liq_enthalpy_pass_through(_, c):
        return m.H_L[c, t] == m.H_L[c, t + 1]

    @no_tray.Constraint(m.comps)
    def vap_enthalpy_pass_through(_, c):
        return m.H_V[c, t] == m.H_V[c, t - 1]


def _build_feed_tray_energy_balance(m):
    t = m.feed_tray

    @m.Constraint()
    def feed_tray_energy_balance(_):
        return (
            sum(m.feed[c] * (
                m.H_L_spec_feed[c] * (1 - m.feed_vap_frac) +
                m.H_V_spec_feed[c] * m.feed_vap_frac)
                for c in m.comps) +
            sum(
                # Heat of liquid from tray above
                m.L[c, t + 1] * m.H_L[c, t + 1]
                # heat of liquid to tray below
                - m.L[c, t] * m.H_L[c, t]
                # heat of vapor from tray below
                + m.V[c, t - 1] * m.H_V[c, t - 1]
                # heat of vapor to tray above
                - m.V[c, t] * m.H_V[c, t]
                for c in m.comps)) * 1E-3 == 0

    @m.Constraint(m.comps)
    def feed_tray_liq_enthalpy_calc(_, c):
        return m.H_L[c, t] == m.liq_enthalpy_expr[t, c]

    @m.Constraint(m.comps)
    def feed_tray_vap_enthalpy_calc(_, c):
        return m.H_V[c, t] == m.vap_enthalpy_expr[t, c]

    @m.Expression(m.comps)
    def feed_liq_enthalpy_expr(_, c):
        k = m.liq_Cp_const[c]
        return (
            k['A'] * (m.T_feed - m.T_ref) +
            k['B'] * (m.T_feed ** 2 - m.T_ref ** 2) / 2 +
            k['C'] * (m.T_feed ** 3 - m.T_ref ** 3) / 3 +
            k['D'] * (m.T_feed ** 4 - m.T_ref ** 4) / 4 +
            k['E'] * (m.T_feed ** 5 - m.T_ref ** 5) / 5) * 1E-6

    @m.Constraint(m.comps)
    def feed_liq_enthalpy_calc(_, c):
        return m.H_L_spec_feed[c] == m.feed_liq_enthalpy_expr[c]

    @m.Expression(m.comps)
    def feed_vap_enthalpy_expr(_, c):
        k = m.vap_Cp_const[c]
        return (
            m.dH_vap[c] +
            k['A'] * (m.T_feed - m.T_ref) +
            k['B'] * (m.T_feed ** 2 - m.T_ref ** 2) / 2 +
            k['C'] * (m.T_feed ** 3 - m.T_ref ** 3) / 3 +
            k['D'] * (m.T_feed ** 4 - m.T_ref ** 4) / 4 +
            k['E'] * (m.T_feed ** 5 - m.T_ref ** 5) / 5) * 1E-3

    @m.Constraint(m.comps)
    def feed_vap_enthalpy_calc(_, c):
        return m.H_V_spec_feed[c] == m.feed_vap_enthalpy_expr[c]


def _build_condenser_energy_balance(m):
    t = m.condens_tray

    @m.partial_cond.Constraint()
    def partial_condenser_energy_balance(_):
        return -m.Qc + sum(
            - m.D[c] * m.H_L[c, t]  # heat of liquid distillate
            - m.L[c, t] * m.H_L[c, t]  # heat of liquid to tray below
            + m.V[c, t - 1] * m.H_V[c, t - 1]  # heat of vapor from tray below
            - m.V[c, t] * m.H_V[c, t]  # heat of vapor from partial condenser
            for c in m.comps) * 1E-3 == 0

    @m.total_cond.Constraint()
    def total_condenser_energy_balance(_):
        return -m.Qc + sum(
            - m.D[c] * m.H_L[c, t]  # heat of liquid distillate
            - m.L[c, t] * m.H_L[c, t]  # heat of liquid to tray below
            + m.V[c, t - 1] * m.H_V[c, t - 1]  # heat of vapor from tray below
            for c in m.comps) * 1E-3 == 0

    @m.Constraint(m.comps)
    def condenser_liq_enthalpy_calc(_, c):
        return m.H_L[c, t] == m.liq_enthalpy_expr[t, c]

    @m.partial_cond.Constraint(m.comps)
    def vap_enthalpy_calc(_, c):
        return m.H_V[c, t] == m.vap_enthalpy_expr[t, c]


def _build_reboiler_energy_balance(m):
    t = m.reboil_tray

    @m.Constraint()
    def reboiler_energy_balance(_):
        return m.Qb + sum(
            m.L[c, t + 1] * m.H_L[c, t + 1]  # Heat of liquid from tray above
            - m.B[c] * m.H_L[c, t]  # heat of liquid bottoms if reboiler
            - m.V[c, t] * m.H_V[c, t]  # heat of vapor to tray above
            for c in m.comps) * 1E-3 == 0

    @m.Constraint(m.comps)
    def reboiler_liq_enthalpy_calc(_, c):
        return m.H_L[c, t] == m.liq_enthalpy_expr[t, c]

    @m.Constraint(m.comps)
    def reboiler_vap_enthalpy_calc(_, c):
        return m.H_V[c, t] == m.vap_enthalpy_expr[t, c]