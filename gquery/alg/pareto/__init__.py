'''
Pareto CGQ solvers. Each module exposes one entry point:
  pbs.solve_pbs            Pareto Beam Search (beam_size=0 gives the complete frontier)
  pts.solve_pts            Pareto Tree Search (depth first)
  dp.solve_dp              conventional requirement-stage Pareto DP
  epsilon_ilp.solve_epsilon_ilp   repeated epsilon-constraint ILP
  brute_force.solve_brute_force   exhaustive enumeration
metrics holds the objective record and the frontier metrics.
'''
