"""
main_small_batch.py
This file contains the main function to solve the small batch problem with different approaches.

References:
    - Kocis, G. R.; Grossmann, I. E. Global Optimization of Nonconvex Mixed-Integer Nonlinear Programming (MINLP) Problems in Process Synthesis. Ind. Eng. Chem. Res. 1988, 27 (8), 1407–1421. 
"""
import csv
import logging
import os
from math import ceil, fabs
import time

import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.infeasible import log_infeasible_constraints

from gdp.dsda.dsda_functions import (
    external_ref,
    generate_initialization,
    get_external_information,
    initialize_model,
    solve_complete_external_enumeration,
    solve_subproblem,
    solve_with_dsda,
    solve_with_gdpopt,
    solve_with_minlp,
    visualize_dsda,
)
from gdp.small_batch.gdp_small_batch import build_small_batch


def problem_logic_batch(m):
    """
    This function returns the logic expressions to be used in the disjunctive constraints of the D-SDA approach for the small batch problem.

    Args:
        m (pyomo.ConcreteModel): Pyomo model of the small batch problem

    Return:
        logic_expr (list): List of logic expressions to be used in the disjunctive constraints
    """
    logic_expr = []
    for k in m.k:
        for j in m.j:
            logic_expr.append([m.Y[k, j], m.Y_exists[k, j].indicator_var])
            logic_expr.append([~m.Y[k, j], m.Y_not_exists[k, j].indicator_var])
    return logic_expr


if __name__ == "__main__":
    # Inputs
    timelimit = 900
    model_args = {}
    starting_point = [3, 3, 3]

    globaltee = True
    # Setting logging level to ERROR to avoid printing FBBT warning of some constraints not implemented
    logging.basicConfig(level=logging.ERROR)

    csv_columns = [
        'Method',
        'Approach',
        'Solver',
        'Objective',
        'Time',
        'Status',
        'User_time',
    ]
    dict_data = []
    dir_path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(dir_path, "results", "small_batch_results.csv")

    nlps = ['knitro', 'baron']  # 'msnlp']

    nlp_opts = dict((nlp, {}) for nlp in nlps)
    # nlp_opts['msnlp']['add_options'] = [
    #     'GAMS_MODEL.optfile = 1;'
    #     '\n'
    #     '$onecho > msnlp.opt \n'
    #     'nlpsolver knitro \n'
    #     '$offecho \n'
    # ]

    minlps = ['antigone', 'baron', 'scip', 'dicopt', 'sbb', 'knitro']

    minlps_opts = dict((minlp, {}) for minlp in minlps)
    minlps_opts['dicopt']['add_options'] = [
        'GAMS_MODEL.optfile = 1;'
        '\n'
        '$onecho > dicopt.opt \n'
        'stop 0 \n'
        'relaxed 2 \n'
        'maxcycles 10000 \n'
        'nlpsolver knitro \n'
        '$offecho \n'
    ]

    minlps_opts['sbb']['add_options'] = [
        'GAMS_MODEL.optfile = 1;'
        '\n'
        '$onecho > sbb.opt \n'
        'rootsolver knitro \n'
        'subsolver knitro \n'
        '$offecho \n'
    ]
    transformations = ['bigm', 'hull']
    ks = ['Infinity', '2']
    strategies = ['LOA', 'LBB']

    # Initializations
    json_file = os.path.join(dir_path, "gdp/dsda/", "small_batch_initialization.json")
    if os.path.exists(json_file):
        init_path = json_file
    else:
        m = build_small_batch()
        ext_ref = {m.Y: m.k}
        (
            reformulation_dict,
            number_of_external_variables,
            lower_bounds,
            upper_bounds,
        ) = get_external_information(m, ext_ref, tee=globaltee)
        m_fixed = external_ref(
            m=m,
            x=starting_point,
            extra_logic_function=problem_logic_batch,
            dict_extvar=reformulation_dict,
            tee=globaltee,
        )
        m_solved = solve_subproblem(
            m=m_fixed, subproblem_solver='baron', timelimit=100, tee=globaltee
        )
        init_path = generate_initialization(
            m=m_solved, starting_initialization=True, model_name='small_batch'
        )

    # MINLP
    for solver in minlps:
        for transformation in transformations:
            new_result = {}
            m = build_small_batch()
            m_init = initialize_model(m, json_path=init_path)
            m_solved = solve_with_minlp(
                m_init,
                transformation=transformation,
                minlp=solver,
                minlp_options=minlps_opts[solver],
                timelimit=timelimit,
                gams_output=False,
                tee=globaltee,
            )
            new_result = {
                'Method': 'MINLP',
                'Approach': transformation,
                'Solver': solver,
                'Objective': pe.value(m_solved.obj),
                'Time': m_solved.results.solver.user_time,
                'Status': m_solved.results.solver.termination_condition,
                'User_time': 'NA',
            }
            dict_data.append(new_result)
            print(new_result)

    # # GDPopt
    # for solver in nlps:
    #     for strategy in strategies:
    #         new_result = {}
    #         m = build_small_batch()
    #         m_init = initialize_model(m, json_path=init_path)
    #         m_solved = solve_with_gdpopt(
    #             m_init,
    #             mip='cplex',
    #             nlp=solver,
    #             nlp_options=nlp_opts[solver],
    #             timelimit=timelimit,
    #             strategy=strategy,
    #             tee=globaltee,
    #         )
    #         new_result = {
    #             'Method': 'GDPopt',
    #             'Approach': strategy,
    #             'Solver': solver,
    #             'Objective': pe.value(m_solved.obj),
    #             'Time': m_solved.results.solver.user_time,
    #             'Status': m_solved.results.solver.termination_condition,
    #             'User_time': 'NA',
    #         }
    #         dict_data.append(new_result)
    #         print(new_result)

    # D-SDA - MINLP
    m = build_small_batch()
    ext_ref = {m.Y: m.k}
    get_external_information(m, ext_ref, tee=globaltee)

    for solver in nlps:
        for k in ks:
            for transformation in transformations:
                new_result = {}
                m_solved, _, _ = solve_with_dsda(
                    model_function=build_small_batch,
                    model_args={},
                    starting_point=starting_point,
                    ext_dict=ext_ref,
                    mip_transformation=True,
                    transformation=transformation,
                    ext_logic=problem_logic_batch,
                    k=k,
                    provide_starting_initialization=True,
                    feasible_model='small_batch',
                    subproblem_solver=solver,
                    subproblem_solver_options=nlp_opts[solver],
                    iter_timelimit=timelimit,
                    timelimit=timelimit,
                    gams_output=False,
                    tee=globaltee,
                    global_tee=globaltee,
                )
                new_result = {
                    'Method': str('D-SDA_MIP_' + transformation),
                    'Approach': str('k=' + k),
                    'Solver': solver,
                    'Objective': pe.value(m_solved.obj),
                    'Time': m_solved.dsda_time,
                    'Status': m_solved.dsda_status,
                    'User_time': m_solved.dsda_usertime,
                }
                dict_data.append(new_result)
                print(new_result)

    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    # # Complete enumeration
    # for transformation in transformations:
    #     for solver in ['knitro', 'baron']:
    #         m = build_small_batch()
    #         ext_ref = {m.Y: m.k}
    #         get_external_information(m, ext_ref, tee=False)
    #         m_solved = solve_complete_external_enumeration(
    #             model_function=build_small_batch,
    #             model_args={},
    #             ext_dict=ext_ref,
    #             ext_logic=problem_logic_batch,
    #             feasible_model='small_batch',
    #             mip_transformation=True,
    #             transformation=transformation,
    #             subproblem_solver=solver,
    #             subproblem_solver_options=nlp_opts[solver],
    #             iter_timelimit=900,
    #             gams_output=False,
    #             tee=globaltee,
    #             global_tee=globaltee,
    #             export_csv=True,
    #         )
