'''
Weighted CGQ solvers. Each module exposes one entry point:
  pbs.solve_pbs            weighted beam search
  pts.solve_pts            weighted depth-first tree search
  ilp.solve_ilp            exact branch and bound (the reference)
  brute_force.solve_brute_force   exhaustive enumeration (verification only)
objective holds the weighted objective and the shared result record; bounds
holds the MCKP LP bound used by pbs and pts.
'''
