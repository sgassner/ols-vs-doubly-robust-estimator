"""
Monte Carlo Study: OLS vs. Doubly Robust Estimator

Author: Sandro Gassner

Date: 01.05.2022
"""

# load standard functions
import sys
from numpy.random import seed

# set working directory
PATH = 'YOUR_PATCH'
sys.path.append(PATH)

# load own functions
import ols_vs_dre_functions as pc

# set seed to ensure replicability
seed(404040)

# define simulation parameters
n_sim = 1000   # number of simulations
n_obs = 1000   # number of observations per draw

# run simulations and store ATE estimations
results_1, results_2, results_3 = pc.my_simulation(n_sim, n_obs)

# print performance measures and plot ATE estimations for all DGPs
for result in (results_1, results_2, results_3):
    pc.my_printout(data = result, true_ate = 3, title = result.name)
    pc.my_plot(data = result, true_ate = 3, title = result.name)

    


    
