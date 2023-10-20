"""
gdp_small_batch.py

The code builds the GDP model for the small batches problem and solves it using BARON.

"""

import os

import pyomo.environ as pe
from pyomo.core.base.misc import display
from pyomo.core.plugins.transform.logical_to_linear import \
    update_boolean_vars_from_binary
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory


def build_small_batch():
    """
    Function that builds the GDP model for the superstructure of the small batches problem.
    Args:

    Returns:
        m: pyomo model
    """
    NK = 3

    # Model
    m = pe.ConcreteModel()

    # Sets
    m.i = pe.Set(initialize=['a', 'b'])  # Set of products, i = 1, 2
    m.j = pe.Set(initialize=['mixer', 'reactor',
                             'centrifuge'])  # Set of stages, j = 1, 2, 3
    m.k = pe.RangeSet(NK, doc='Set of potential number of parallel units')    # Set of potential number of parallel units

    # Parameters and Scalars

    m.h = pe.Param(initialize=6000, doc= 'Horizon time [hr]')  # Horizon time (available time) [hr]
    m.vlow = pe.Param(initialize=250, doc= 'Lower bound for size of batch unit [L]')  # Lower bound for size of batch unit [L]
    m.vupp = pe.Param(initialize=2500, doc='Upper bound for size of batch unit [L]')  # Upper bound for size of batch unit [L]

    # Production rate of product i [kg/hr]
    m.q = pe.Param(m.i, initialize={'a': 200000, 'b': 150000}, doc='Production rate of product i [kg/hr]')
    # Cost coefficient for batch units
    m.alpha = pe.Param(
        m.j, initialize={'mixer': 250, 'reactor': 500, 'centrifuge': 340}, doc='Cost coefficient for batch units')
    # Cost exponent for batch units
    m.beta = pe.Param(
        m.j, initialize={'mixer': 0.6, 'reactor': 0.6, 'centrifuge': 0.6}, doc='Cost exponent for batch units')

    def coeff_init(m, k):
        """Coefficient for number of parallel units"""
        return pe.log(k)

    # Represent number of parallel units
    m.coeff = pe.Param(m.k, initialize=coeff_init)

    s_init = {('a', 'mixer'): 2, ('a', 'reactor'): 3, ('a', 'centrifuge'): 4,
              ('b', 'mixer'): 4, ('b', 'reactor'): 6, ('b', 'centrifuge'): 3}

    # Size factor for product i in stage j [kg/L]
    m.s = pe.Param(m.i, m.j, initialize=s_init, doc='Size factor for product i in stage j [kg/L]')

    t_init = {('a', 'mixer'): 8, ('a', 'reactor'): 20, ('a', 'centrifuge'): 4,
              ('b', 'mixer'): 10, ('b', 'reactor'): 12, ('b', 'centrifuge'): 3}

    # Processing time of product i in batch j [hr]
    m.t = pe.Param(m.i, m.j, initialize=t_init, doc='Processing time of product i in batch j [hr]')

    # Variables
    m.Y = pe.BooleanVar(m.k, m.j, doc='Stage existence')    # Stage existence
    m.coeffval = pe.Var(m.k, m.j,  within=pe.NonNegativeReals,
                        bounds=(0, pe.log(NK)), doc='Activation of coefficient for number of parallel units')  # Activation of coefficient for number of parallel units
    m.v = pe.Var(m.j, within=pe.NonNegativeReals, bounds=(
        pe.log(m.vlow), pe.log(m.vupp)), doc='Volume of stage j')  # Volume of stage j
    m.b = pe.Var(m.i, within=pe.NonNegativeReals, doc='Batch size of product i')  # Batch size of product i
    m.tl = pe.Var(m.i, within=pe.NonNegativeReals, doc='Cycle time of product i')  # Cycle time of product i
    # Number of units in parallel stage j
    m.n = pe.Var(m.j, within=pe.NonNegativeReals, doc='Number of units in parallel stage j')  # Number of units in parallel stage j

    # Constraints

    # Volume requirement in stage j
    @m.Constraint(m.i, m.j)
    def vol(m, i, j):
        """Volume requirement in stage j"""
        return m.v[j] >= pe.log(m.s[i, j]) + m.b[i]

    # Cycle time for each product i
    @m.Constraint(m.i, m.j)
    def cycle(m, i, j):
        """Cycle time for each product i"""
        return m.n[j] + m.tl[i] >= pe.log(m.t[i, j])

    # Constraint for production time
    @m.Constraint()
    def time(m):
        """Horizon time constraint"""
        return sum(m.q[i]*pe.exp(m.tl[i]-m.b[i]) for i in m.i) <= m.h

    # Relating number of units to 0-1 variables
    @m.Constraint(m.j)
    def units(m, j):
        """Relating number of units to 0-1 variables"""
        return m.n[j] == sum(m.coeffval[k, j] for k in m.k)

    # Only one choice for parallel units is feasible
    @m.LogicalConstraint(m.j)
    def lim(m, j):
        """Only one choice for parallel units is feasible"""
        return pe.exactly(1, m.Y[1, j], m.Y[2, j], m.Y[3, j])

    # _______ Disjunction_________

    def build_existence_equations(disjunct, k, j):
        m = disjunct.model()

        # Coeffval activation
        @disjunct.Constraint()
        def coeffval_act(disjunct):
            return m.coeffval[k, j] == m.coeff[k]

    def build_not_existence_equations(disjunct, k, j):
        m = disjunct.model()

        # Coeffval deactivation
        @disjunct.Constraint()
        def coeffval_deact(disjunct):
            return m.coeffval[k, j] == 0

    # Create disjunction block
    m.Y_exists = Disjunct(m.k, m.j, rule=build_existence_equations)
    m.Y_not_exists = Disjunct(m.k, m.j, rule=build_not_existence_equations)

    # Create disjunction

    @m.Disjunction(m.k, m.j)
    def Y_exists_or_not(m, k, j):
        """Create the disjunction for Y[k, j]"""
        return [m.Y_exists[k, j], m.Y_not_exists[k, j]]

    # Associate Boolean variables with with disjunction
    for k in m.k:
        for j in m.j:
            m.Y[k, j].associate_binary_var(m.Y_exists[k, j].indicator_var)

    # ____________________________

    # Objective
    def obj_rule(m):
        """Investment cost of batch units"""
        return sum(m.alpha[j]*(pe.exp(m.n[j] + m.beta[j]*m.v[j])) for j in m.j)

    m.obj = pe.Objective(rule=obj_rule, sense=pe.minimize)

    return m


def external_ref(m, x, logic_expr=None):
    ext_var = {}
    p = 0
    for j in m.j:
        ext_var[j] = x[p]
        p = p+1

    for k in m.k:
        for j in m.j:
            if k == ext_var[j]:
                m.Y[k, j].fix(True)
                m.Y_exists[k, j].indicator_var.fix(
                    True)  # IS THIS REQUIRED????
                m.Y_not_exists[k, j].indicator_var.fix(
                    False)  # IS THIS REQUIRED????
            else:
                m.Y[k, j].fix(False)
                m.Y_exists[k, j].indicator_var.fix(
                    False)  # IS THIS REQUIRED????
                m.Y_not_exists[k, j].indicator_var.fix(
                    True)  # IS THIS REQUIRED????

    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    pe.TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    pe.TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
        m, tmp=False, ignore_infeasible=True)

    return m


def solve_with_minlp(m, transformation='bigm', minlp='baron', timelimit=10):

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    transformation_string = 'gdp.' + transformation
    pe.TransformationFactory(transformation_string).apply_to(m)

    # Solution step
    dir_path = os.path.dirname(os.path.abspath(__file__))
    gams_path = os.path.join(dir_path, "gamsfiles/")
    if not(os.path.exists(gams_path)):
        print('Directory for automatically generated files ' +
              gams_path + ' does not exist. We will create it')
        os.makedirs(gams_path)

    solvername = 'gams'
    opt = SolverFactory(solvername, solver=minlp)
    m.results = opt.solve(m, tee=True,
                          # Uncomment the following lines if you want to save GAMS models
                          # keepfiles=True,
                          # tmpdir=gams_path,
                          # symbolic_solver_labels=True,
                          add_options=[
                              'option reslim = ' + str(timelimit) + ';'
                              'option optcr = 0.0;'
                              # Uncomment the following lines to setup IIS computation of BARON through option file
                              # 'GAMS_MODEL.optfile = 1;'
                              # '\n'
                              # '$onecho > baron.opt \n'
                              # 'CompIIS 1 \n'
                              # '$offecho'
                              # 'display(execError);'
                          ])
    update_boolean_vars_from_binary(m)
    return m


if __name__ == "__main__":
    m = build_small_batch()
    m_solved = solve_with_minlp(m, transformation='bigm', minlp='baron', timelimit=120)

    # EXTERNAL REF TEST (this test can be deleted)
    newmodel = external_ref(m, [2, 2, 1], logic_expr=None)
    
