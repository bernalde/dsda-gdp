import pyomo.environ as pe
from pyomo.gdp import (Disjunct, Disjunction)
import numpy as np
import time as time
from pyomo.core.base.misc import display
from pyomo.opt.base.solvers import SolverFactory
import os


def external_init(NT):
    m = pe.ConcreteModel(name='external_initialization')
    m.ext_var_1 = pe.Var(within=pe.NonNegativeIntegers, bounds=(1, NT))
    m.ext_var_2 = pe.Var(within=pe.NonNegativeIntegers, bounds=(1, NT))

    @m.Constraint()
    def feas_constraint(m):
        return m.ext_var_2 - m.ext_var_1 <= 0

    m.z_ext = pe.Var()

    @m.Constraint()
    def z_cons(m):
        return m.z_ext == 1

    def obj_rule(m):
        return m.z_ext

    m.obj = pe.Objective(rule=obj_rule, sense=pe.minimize)

    solvername = 'gams'
    opt = SolverFactory(solvername, solver='cplex')
    results = opt.solve(m, tee=False,
                        add_options=[
                            'option reslim = 20;'
                            'option optcr = 0.0;'
                        ])

    return [pe.value(m.ext_var_1), pe.value(m.ext_var_2)]


def fnlp_gdp(NT, x, provide_init=False, init={}):
    # INPUTS

    # PYOMO MODEL
    m = pe.ConcreteModel(name='gdp_reactors')

    # SETS
    m.I = pe.Set(initialize=['A', 'B'])  # Set of components
    m.N = pe.RangeSet(1, NT)  # Set of units in the superstructure

    # PARAMETERS
    m.k = pe.Param(initialize=2)  # Kinetic constant [L/(mol*s)]
    m.order1 = pe.Param(initialize=1)  # Partial order of reacton 1
    m.order2 = pe.Param(initialize=1)  # Partial order of reaction 2
    m.QF0 = pe.Param(initialize=1)  # Inlet volumetric flow [L/s]
    C0_Def = {'A': 0.99, 'B': 0.01}
    # Initial concentration of reagents [mol/L]
    m.C0 = pe.Param(m.I, initialize=C0_Def)

    # Inlet molar flow [mol/s]

    def F0_Def(m, i):
        return m.C0[i]*m.QF0

    m.F0 = pe.Param(m.I, initialize=F0_Def)

    # BINARY VARIABLE
    m.YF = pe.Var(m.N, within=pe.Binary)

    # BOOLEAN VARIABLES

    # Existence of recycle flow in unit n
    m.YR = pe.BooleanVar(m.N)

    # Unit operation in n (True if unit n is a CSTR, False if unit n is a bypass)
    m.YP = pe.BooleanVar(m.N)

    if provide_init:
        # REAL VARIABLES

        # Network Variables
        # Outlet flow rate of the superstructure unit [L/s]
        m.Q = pe.Var(
            m.N, initialize=init['Q'], within=pe.NonNegativeReals, bounds=(0, 10))

        # Outlet flow rate recycle activation of the superstructure unit [L/s]
        m.QFR = pe.Var(
            m.N, initialize=init['QFR'], within=pe.NonNegativeReals, bounds=(0, 10))

        # Molar flow [mol/s]
        m.F = pe.Var(
            m.I, m.N, initialize=init['F'], within=pe.NonNegativeReals, bounds=(0, 10))

        # Molar flow  recycle activation [mol/s]
        m.FR = pe.Var(
            m.I, m.N, initialize=init['FR'], within=pe.NonNegativeReals, bounds=(0, 10))

        # Reaction rate [mol/(L*s)]
        m.rate = pe.Var(
            m.I, m.N, initialize=init['rate'], within=pe.Reals, bounds=(-10, 10))

        # Reactor volume [L]
        m.V = pe.Var(
            m.N, initialize=init['V'], within=pe.NonNegativeReals, bounds=(0, 10))

        # Volume activation [L]
        m.c = pe.Var(
            m.N, initialize=init['c'], within=pe.NonNegativeReals, bounds=(0, 10))

        # Splitter Variables
        # Recycle flow rate  [L/s]
        m.QR = pe.Var(initialize=init['QR'],
                      within=pe.NonNegativeReals, bounds=(0, 10))

        # Product flow rate  [L/s]
        m.QP = pe.Var(initialize=init['QP'],
                      within=pe.NonNegativeReals, bounds=(0, 10))

        # Recycle molar flow [mol/s]
        m.R = pe.Var(
            m.I, initialize=init['R'], within=pe.NonNegativeReals, bounds=(0, 10))

        # Product molar flow [mol/s]
        m.P = pe.Var(
            m.I, initialize=init['P'], within=pe.NonNegativeReals, bounds=(0, 10))

    else:
        # REAL VARIABLES

        # Network Variables
        # Outlet flow rate of the superstructure unit [L/s]
        m.Q = pe.Var(m.N, within=pe.NonNegativeReals, bounds=(0, 10))

        # Outlet flow rate recycle activation of the superstructure unit [L/s]
        m.QFR = pe.Var(m.N, within=pe.NonNegativeReals, bounds=(0, 10))

        # Molar flow [mol/s]
        m.F = pe.Var(m.I, m.N, within=pe.NonNegativeReals, bounds=(0, 10))

        # Molar flow  recycle activation [mol/s]
        m.FR = pe.Var(m.I, m.N, within=pe.NonNegativeReals, bounds=(0, 10))

        # Reaction rate [mol/(L*s)]
        m.rate = pe.Var(m.I, m.N, within=pe.Reals, bounds=(-10, 10))

        # Reactor volume [L]
        m.V = pe.Var(m.N, within=pe.NonNegativeReals, bounds=(0, 10))

        # Volume activation [L]
        m.c = pe.Var(m.N, within=pe.NonNegativeReals, bounds=(0, 10))

        # Splitter Variables
        # Recycle flow rate  [L/s]
        m.QR = pe.Var(within=pe.NonNegativeReals, bounds=(0, 10))

        # Product flow rate  [L/s]
        m.QP = pe.Var(within=pe.NonNegativeReals, bounds=(0, 10))

        # Recycle molar flow [mol/s]
        m.R = pe.Var(m.I, within=pe.NonNegativeReals, bounds=(0, 10))

        # Product molar flow [mol/s]
        m.P = pe.Var(m.I, within=pe.NonNegativeReals, bounds=(0, 10))

    # CONSTRAINTS

    # Unreacted Feed Balances
    # Unreacted feed unit mole balance

    def unreact_mole_rule(m, i, n):
        if n == NT:
            return m.F0[i] + m.FR[i, n] - m.F[i, n] + m.rate[i, n]*m.V[n] == 0
        else:
            return pe.Constraint.Skip

    m.unreact_mole = pe.Constraint(m.I, m.N, rule=unreact_mole_rule)

    # Unreacted feed unit continuity

    def unreact_cont_rule(m, n):
        if n == NT:
            return m.QF0 + m.QFR[n] - m.Q[n] == 0
        else:
            return pe.Constraint.Skip

    m.unreact_cont = pe.Constraint(m.N, rule=unreact_cont_rule)

    # Reactor Balances
    # Reactor mole balance

    def react_mole_rule(m, i, n):
        if n != NT:
            return m.F[i, n+1] + m.FR[i, n] - m.F[i, n] + m.rate[i, n]*m.V[n] == 0
        else:
            return pe.Constraint.Skip

    m.react_mole = pe.Constraint(m.I, m.N, rule=react_mole_rule)

    # Reactor continuity

    def react_cont_rule(m, n):
        if n != NT:
            return m.Q[n+1] + m.QFR[n] - m.Q[n] == 0
        else:
            return pe.Constraint.Skip

    m.react_cont = pe.Constraint(m.N, rule=react_cont_rule)

    # Splitting Point Balances
    # Splitting point mole balance

    def split_mole_rule(m, i):
        return m.F[i, 1] - m.P[i] - m.R[i] == 0

    m.split_mole = pe.Constraint(m.I, rule=split_mole_rule)

    # Splitting point continuity

    def split_cont_rule(m):
        return m.Q[1] - m.QP - m.QR == 0

    m.split_cont = pe.Constraint(rule=split_cont_rule)

    # Splitting point additional constraints

    def split_add_rule(m, i):
        return m.P[i]*m.Q[1] - m.F[i, 1]*m.QP == 0

    m.split_add = pe.Constraint(m.I, rule=split_add_rule)

    # Product Specification

    def prod_spec_rule(m):
        return m.QP*0.95 - m.P['B'] == 0

    m.prod_spec = pe.Constraint(rule=prod_spec_rule)

    # Volume Constraint

    def vol_cons_rule(m, n):
        if n != 1:
            return m.V[n] - m.V[n-1] == 0
        else:
            return pe.Constraint.Skip

    m.vol_cons = pe.Constraint(m.N, rule=vol_cons_rule)

    # YD Disjunction block equation definition

    def build_cstr_equations(disjunct, n):
        m = disjunct.model()

        # Reaction rates calculation
        @disjunct.Constraint()
        def YPD_rate_calc(disjunct):
            return m.rate['A', n]*((m.Q[n])**m.order1)*((m.Q[n])**m.order2)+m.k*((m.F['A', n])**m.order1)*((m.F['B', n])**m.order2) == 0

        # Reaction rate relation
        @disjunct.Constraint()
        def YPD_rate_rel(disjunct):
            return m.rate['B', n] + m.rate['A', n] == 0

        # Volume activation
        @disjunct.Constraint()
        def YPD_vol_act(disjunct):
            return m.c[n] - m.V[n] == 0

    def build_bypass_equations(disjunct, n):
        m = disjunct.model()

        # FR desactivation
        @disjunct.Constraint(m.I)
        def neg_YPD_FR_desact(disjunct, i):
            return m.FR[i, n] == 0

        # Rate desactivation
        @disjunct.Constraint(m.I)
        def neg_YPD_rate_desact(disjunct, i):
            return m.rate[i, n] == 0

        # QFR desactivation
        @disjunct.Constraint()
        def neg_YPD_QFR_desact(disjunct):
            return m.QFR[n] == 0

        @disjunct.Constraint()
        def neg_YPD_vol_desact(disjunct):
            '''
            Volume desactivation function for defining pyomo model
            args:
                disjunct: pyomo block with disjunct to include the constraint
                n: pyomo set with reactor index
            return: 
                return constraint
            '''
            return m.c[n] == 0

    # YR Disjuction block equation definition

    def build_recycle_equations(disjunct, n):
        m = disjunct.model()

        # FR activation
        @disjunct.Constraint(m.I)
        def YRD_FR_act(disjunct, i):
            return m.FR[i, n] - m.R[i] == 0

        # QFR activation
        @disjunct.Constraint()
        def YRD_QFR_act(disjunct):
            return m.QFR[n] - m.QR == 0

    def build_no_recycle_equations(disjunct, n):
        m = disjunct.model()

        # FR desactivation
        @disjunct.Constraint(m.I)
        def neg_YRD_FR_desact(disjunct, i):
            return m.FR[i, n] == 0

        # QFR desactivation
        @disjunct.Constraint()
        def neg_YRD_QFR_desact(disjunct):
            return m.QFR[n] == 0

    # Create disjunction blocks
    m.YR_is_recycle = Disjunct(m.N, rule=build_recycle_equations)
    m.YR_is_not_recycle = Disjunct(m.N, rule=build_no_recycle_equations)

    m.YP_is_cstr = Disjunct(m.N, rule=build_cstr_equations)
    m.YP_is_bypass = Disjunct(m.N, rule=build_bypass_equations)

    # Create disjunctions

    @m.Disjunction(m.N)
    def YP_is_cstr_or_bypass(m, n):
        return [m.YP_is_cstr[n], m.YP_is_bypass[n]]

    @m.Disjunction(m.N)
    def YR_is_recycle_or_not(m, n):
        return [m.YR_is_recycle[n], m.YR_is_not_recycle[n]]

    # Associate Boolean variables with with disjunctions
    for n in m.N:
        m.YP[n].associate_binary_var(m.YP_is_cstr[n].indicator_var)
        m.YR[n].associate_binary_var(m.YR_is_recycle[n].indicator_var)

    # Logic Constraints
    # Unit must be a CSTR to include a recycle

    def cstr_if_recycle_rule(m, n):
        return m.YR_is_recycle[n].indicator_var <= m.YP_is_cstr[n].indicator_var

    m.cstr_if_recycle = pe.Constraint(m.N, rule=cstr_if_recycle_rule)

    # There is only one unreacted feed

    def one_unreacted_feed_rule(m):
        return sum(m.YF[n] for n in m.N) == 1

    m.one_unreacted_feed = pe.Constraint(rule=one_unreacted_feed_rule)

    # There is only one recycle stream

    def one_recycle_rule(m):
        return sum(m.YR_is_recycle[n].indicator_var for n in m.N) == 1

    m.one_recycle = pe.Constraint(rule=one_recycle_rule)

    # Unit operation in n constraint

    def unit_in_n_rule(m, n):
        return m.YP_is_cstr[n].indicator_var == 1 - (sum(m.YF[n2] for n2 in m.N if n2 <= n) - m.YF[n])

    m.unit_in_n = pe.Constraint(m.N, rule=unit_in_n_rule)

    # External variable fix
    ext_var_1 = x[0]
    ext_var_2 = x[1]

    for n in m.N:
        if n == ext_var_1:
            m.YF[n].fix(1)
        else:
            m.YF[n].fix(0)

        if n == ext_var_2:
            m.YR_is_recycle[n].indicator_var.fix(1)
            m.YR_is_not_recycle[n].indicator_var.fix(0)
        else:
            m.YR_is_recycle[n].indicator_var.fix(0)
            m.YR_is_not_recycle[n].indicator_var.fix(1)

        temp = 1 - (sum(m.YF[n2] for n2 in m.N if n2 <= n) - m.YF[n])
        if temp == 1:
            m.YP_is_cstr[n].indicator_var.fix(1)
            m.YP_is_bypass[n].indicator_var.fix(0)
        else:
            m.YP_is_cstr[n].indicator_var.fix(0)
            m.YP_is_bypass[n].indicator_var.fix(1)

    # OBJECTIVE

    def obj_rule(m):
        return sum(m.c[n] for n in m.N)

    m.obj = pe.Objective(rule=obj_rule, sense=pe.minimize)

    # Transform the model using the BigM relaxation

    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    pe.TransformationFactory('gdp.bigm').apply_to(m)

    # SOLVE
    dir_path = os.path.dirname(os.path.abspath(__file__))
    gams_path = os.path.join(dir_path, "gamsfiles/")
    if not(os.path.exists(gams_path)):
        print('Directory for automatically generated files ' +
              gams_path + ' does not exist. We will create it')
        os.makedirs(gams_path)

    #opt = SolverFactory('ipopt')
    #results = opt.solve(m)
    solvername = 'gams'
    opt = SolverFactory(solvername, solver='msnlp')
    results = opt.solve(m, tee=False,
                        # Uncomment the following lines if you want to save GAMS models
                        keepfiles=True,
                        tmpdir=gams_path,
                        symbolic_solver_labels=True,
                        add_options=[
                            'option reslim = 10;'
                            'option optcr = 0.0;'
                            # Uncomment the following lines to setup IIS computation of BARON through option file
                            # 'GAMS_MODEL.optfile = 1;'
                            # '\n'
                            # '$onecho > baron.opt \n'
                            # 'CompIIS 1 \n'
                            # '$offecho'
                            #'display(execError);'
                        ])

    Q_init, QFR_init, F_init, FR_init, rate_init, V_init, c_init,  R_init, P_init = {
    }, {}, {}, {}, {}, {}, {}, {}, {}

    QR_init = pe.value(m.QR)
    QP_init = pe.value(m.QP)
    for n in m.N:
        Q_init[n] = pe.value(m.Q[n])
        QFR_init[n] = pe.value(m.QFR[n])
        V_init[n] = pe.value(m.V[n])
        c_init[n] = pe.value(m.c[n])

        for i in m.I:
            F_init[i, n] = pe.value(m.F[i, n])
            FR_init[i, n] = pe.value(m.FR[i, n])
            rate_init[i, n] = pe.value(m.rate[i, n])

    for i in m.I:
        R_init[i] = pe.value(m.R[i])
        P_init[i] = pe.value(m.P[i])

    initialization = {'Q': Q_init, 'QFR': QFR_init, 'F': F_init, 'FR': FR_init, 'rate': rate_init,
                      'V': V_init, 'c': c_init, 'QR': QR_init, 'QP': QP_init, 'R': R_init, 'P': P_init}

    return m, results.solver.status, initialization


def complete_enumeration(NT):
    X1, X2, aux, aux2, x = [], [], [], 1, {}

    for i in range(1, NT+1):
        X1.append(i)
        aux.append(i)
        X2.append(aux2)

    for i in range(NT-1):
        aux.pop(0)
        aux2 += 1
        for j in aux:
            X1.append(j)
            X2.append(aux2)

    print('=============================')
    print('%6s %6s %12s' % ('x1', 'x2', 'Objective'))
    print('-----------------------------')

    for i in range(len(X1)):
        x = [X1[i], X2[i]]
        m, _, _ = fnlp_gdp(NT, x)
        print('%6s %6s %12s' % (X1[i], X2[i], round(pe.value(m.obj), 5)))
    print('=============================')


def neighborhood_k_eq_2(num_ext):  # number
    num_neigh = 2*num_ext
    neighbors = np.concatenate((np.eye(num_ext), -np.eye(num_ext)), axis=1)
    directions = {}
    for i in range(num_neigh):
        direct = []
        directions[i+1] = direct
        for j in range(num_ext):
            direct.append(neighbors[j, i])
    return directions  # return dict

def my_neighbors(start, neighborhood, optimize=False, min_allowed={}, max_allowed={}, cheating=False):
    neighbors = {0:start}
    for i in neighborhood.keys():
        neighbors[i] = list(map(sum, zip(start,list(neighborhood[i]))))
        
    if optimize:
        new_neighbors = {}
        num_vars = len(neighbors[0])
        for i in neighbors.keys():
            checked = 0
            for j in range(num_vars):
                if neighbors[i][j] >= min_allowed[j+1] and neighbors[i][j] <= max_allowed[j+1]:
                    checked += 1
            if checked == num_vars:
                new_neighbors[i] = neighbors[i]

        if cheating:
            temp = new_neighbors
            for i in list(temp.keys()):
                if new_neighbors[i][1] - new_neighbors[i][0] > 0:
                    new_neighbors.pop(i,None)

        
        return new_neighbors
    return neighbors


def evaluate_neighbors(ext_vars, init, fmin, tol=0.00001):
    improve =  False
    best_var = ext_vars[0]
    best_dir = 0
    best_init = init
    temp = ext_vars
    temp.pop(0, None)
    
    for i in temp.keys():
        m, status , new_init = fnlp_gdp(NT, temp[i], provide_init=True, init=init)
        
        if status == pe.SolverStatus.ok:
            act_obj = pe.value(m.obj)
            if act_obj + tol < fmin:
                fmin = act_obj
                best_var = ext_vars[i]
                best_dir = i
                best_init = new_init
                improve = True

    return fmin, best_var, best_dir, best_init, improve


def move_and_evaluate(start, init, fmin, direction, optimize=False, min_allowed={}, max_allowed={},tol=0.00001):
    best_var = start
    best_init = init
    moved = False
    
    moved_point = list(map(sum, zip(list(start), list(direction))))
    
    if optimize:
        checked = 0
        for j in range(len(moved_point)):
            if moved_point[j] >= min_allowed[j+1] and moved_point[j] <= max_allowed[j+1]:
                    checked += 1
        if checked == len(moved_point):
            m, status , new_init = fnlp_gdp(NT, moved_point, provide_init=True, init=init)
            if status == pe.SolverStatus.ok:
                    act_obj = pe.value(m.obj)
                    if act_obj + tol < fmin:
                        fmin = act_obj
                        best_var = moved_point
                        best_init = new_init
                        moved = True
    else:
        m, status , new_init = fnlp_gdp(NT, moved_point, provide_init=True, init=init)
        if status == pe.SolverStatus.ok:
            act_obj = pe.value(m.obj)
            if act_obj < fmin:
                fmin = act_obj
                best_var = moved_point
                best_init = new_init
                moved = True
    
    return fmin, best_var, moved, best_init

def dsda(NT, k):
    #Initialize
    route = []
    ext_var = external_init(NT)
    route.append(ext_var)
    m, _, init = fnlp_gdp(NT, ext_var)
    fmin = pe.value(m.obj)

    if k == '2':
        neighborhood = neighborhood_k_eq_2(len(ext_var))

    go_1 = True

    while go_1:

        neighbors = my_neighbors(ext_var, neighborhood, optimize=True, min_allowed={1:1, 2:1}, max_allowed={1:5, 2:5}, cheating=True)
        
        fmin, best_var, best_dir, best_init, improve = evaluate_neighbors(neighbors, init, fmin)

        if improve == True:
            go_2 = True
            route.append(best_var)
            while go_2:
                fmin, best_var, moved, best_init = move_and_evaluate(best_var, best_init, fmin, neighborhood[best_dir], optimize=True, min_allowed={1:1, 2:1}, max_allowed={1:5, 2:5})
                if moved == True:
                    route.append(best_var)
                else:
                    ext_var = best_var
                    go_2 = False
                    
        
        else:
            go_1 = False

    print(route)

if __name__ == "__main__":
    NT = 5
    k = '2'
    #complete_enumeration(NT)
    dsda(NT,k)

