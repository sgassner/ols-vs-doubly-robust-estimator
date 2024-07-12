"""
Monte Carlo Study: OLS vs. Doubly Robust Estimator

Author: Sandro Gassner

Date: 01.05.2022
"""

# import modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# DATA GENERATING PROCESS (DGP)
# --------------------------------------------------------------------------- #

def my_dgp(n):
    """
    Creates the following three draws:
    
    DGP 1: y = 4 + 3d + 5*X1 -2*X2 - X3 + u
    DGP 2: y = d * Y1 + (1-d) * Y0
    DGP 3: y = 2 + 3d + 3*X1 - 2*X2 + X3 + u

    Parameters
    ----------       
    n : TYPE: int
        DESCRIPTION: number of observations per DGP
        
    Returns
    -------
    dgp_1 : TYPE: pd.DataFrame
        DESCRIPTION: table with first DGP
    dgp_2 : TYPE: pd.DataFrame
        DESCRIPTION: table with second DGP
    dgp_3 : TYPE: pd.DataFrame
        DESCRIPTION: table with third DGP
    """
    
    # DGP 1
    x_12 = np.random.multivariate_normal([2, 3], [[2, 1], [1, 2]], n)
    x_3 = np.random.randint(0,2,n)
    u = np.random.normal(0,5,n)
    d = (x_12 @ np.array([5, -2]) + 0.5*x_3 > 0)*1
    y = 4 + 3*d + x_12 @ np.array([5, -2]) - x_3 + u 
    
    dgp_1 = pd.DataFrame({'outcome': y,
                          'treatment': d,
                          'covariate_1': x_12[:, 0],
                          'covariate_2': x_12[:, 1],
                          'covariate_3': x_3})
    
    # DGP 2
    x_1 = np.random.randint(0,2,n)
    x_2 = np.random.normal(0,2,n)
    x_3 = np.random.uniform(1,3,n)
    u = np.random.normal(0,1,n)
    d = (0.4 + 0.2*x_1 - 2*x_2 + x_3 > 0)*1
    y0 = 0 + u
    y1 = -2 + 2 * x_1 + x_2 + 2*x_3 + u
    y = d * y1 + (1-d) * y0 
    
    dgp_2 = pd.DataFrame({'outcome': y,
                          'treatment': d,
                          'covariate_1': x_1,
                          'covariate_2': x_2,
                          'covariate_3': x_3}) 
    
    # DGP 3
    x_1 = np.random.binomial(1,0.5,n)
    x_2 = np.random.randint(0,2,n)
    x_3 = np.random.normal(0,2,n) 
    u = np.random.normal(0, 2, n)
    d = (2*x_1 - x_2 - x_3 > 0)*1
    y = 2 + 3*d + 3*x_1 - 2*x_2 + x_3 + u
    
    dgp_3 = pd.DataFrame({'outcome': y,
                          'treatment': d,
                          'covariate_1': x_1,
                          'covariate_2': x_2,
                          'covariate_3': x_3}) 
    
    # return DGPs as data frames
    return dgp_1, dgp_2, dgp_3

# --------------------------------------------------------------------------- #
# ESTIMATORS
# --------------------------------------------------------------------------- #

# OLS function
def my_ols(data, outcome, exog, treat):
    """
    Estimates the ATE by OLS.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data frame that contains the variables of interest
    outcome : TYPE: string
        DESCRIPTION: outcome variable
    exog : TYPE: tuple
        DESCRIPTION: covariates
    treat : TYPE: string
        DESCRIPTION: treatment variable
    Returns
    -------
    ate : TYPE: float
        DESCRIPTION: Estimate of the average treatment effect by OLS.
    """    
    # subset data frame by treatment and covariates
    exog = data.loc[:, (treat, ) + exog]
    
    # add intercept (prepend a vector of ones to the covariate matrix)
    exog = pd.concat([pd.Series(np.ones(len(exog)), index = exog.index,
                                name = 'intercept'), exog], axis = 1)
    
    # compute (x'x)-1 by using the linear algebra from numpy
    x_inv = np.linalg.inv(np.dot(exog.T, exog))
    
    # estimate betas according to the OLS formula b=(x'x)-1(x'y)
    betas = np.dot(x_inv, np.dot(exog.T, data[outcome]))
    
    # get the estimated ATE of the treatment variable
    ate = betas[1]
    
    # return ATE
    return ate

# DRE function
def my_dre(data, outcome, exog, treat):
    """
    Estimates the ATE by Doubly Robust Estimator.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data frame that contains the variables of interest
    outcome : TYPE: string
        DESCRIPTION: outcome variable
    exog : TYPE: tuple
        DESCRIPTION: covariates
    treat : TYPE: string
        DESCRIPTION: treatment variable
    Returns
    -------
    ate : TYPE: float
        DESCRIPTION: Estimate of the average treatment effect by DRE. 
    """
    # convert exog to list
    exog = list(exog)
    
    # Step 1: estimate propensity score by logit model
    ps = LogisticRegression().fit(data[exog], 
                                  data[treat]).predict_proba(data[exog])[:, 1]
    
    # Step 2: estimate outcome equation by parametric model (OLS)
    mu0 = LinearRegression().fit(data.query(f"{treat}==0")[exog], 
                                 data.query(f"{treat}==0")[outcome]).predict(
                                 data[exog])                              
                                     
    mu1 = LinearRegression().fit(data.query(f"{treat}==1")[exog], 
                                 data.query(f"{treat}==1")[outcome]).predict(
                                 data[exog]) 
                                     
    # Step 3: calculate the estimated ATE with the formlua
    ate = np.mean(data[treat]*(data[outcome] - mu1)/ps + mu1) - \
          np.mean((1-data[treat])*(data[outcome] - mu0)/(1-ps) + mu0)
    
    # return ATE
    return ate

# --------------------------------------------------------------------------- #
# SIMULATION
# --------------------------------------------------------------------------- #

def my_simulation(n_sim, n_obs):
    """
    Runs multiple simulations of an ATE estimation by OLS and DRE on data 
    generated by the my_dgp() function.

    Parameters
    ----------
    n_sim : TYPE: int
        DESCRIPTION: number of simulations
    n_obs : TYPE: int
        DESCRIPTION: number of observations per simulation

    Returns
    -------
    results_1 : TYPE: pd.DataFrame
        DESCRIPTION: table with estimated ATEs for DGP 1
    results_2 : TYPE: pd.DataFrame
        DESCRIPTION: table with estimated ATEs for DGP 2
    results_3 : TYPE: pd.DataFrame
        DESCRIPTION: table with estimated ATEs for DGP 3
    """
    # define covariates for the different DGPs
    covariates_1 = ('covariate_1', 'covariate_2', 'covariate_3')
    covariates_2 = ('covariate_1', 'covariate_2', 'covariate_3')
    covariates_3 = ('covariate_1', 'covariate_3')
    
    # create data frames to store the results
    results_1 = pd.DataFrame(np.nan, index = range(0, n_sim), 
                             columns = ['OLS', 'DRE'])
    
    results_2 = pd.DataFrame(np.nan, index = range(0, n_sim), 
                             columns = ['OLS', 'DRE'])
    
    results_3 = pd.DataFrame(np.nan, index = range(0, n_sim), 
                             columns = ['OLS', 'DRE'])
    
    # name the data frames
    results_1.name = 'DGP_1'
    results_2.name = 'DGP_2'
    results_3.name = 'DGP_3'
    
    # run simulations
    for i in range(0, n_sim):
        
        # generate data for draw i
        dgp_1, dgp_2, dgp_3 = my_dgp(n_obs) 
        
        # compute and store estimates of OLS and DRE for draw i and dgp_1        
        results_1['OLS'][i] = my_ols(data = dgp_1, 
                                     outcome = 'outcome', 
                                     exog = covariates_1, 
                                     treat = 'treatment')
        
        results_1['DRE'][i] = my_dre(data = dgp_1, 
                                     outcome = 'outcome', 
                                     exog = covariates_1,
                                     treat = 'treatment')
        
        # compute and store estimates of OLS and DRE for draw i and dgp_2      
        results_2['OLS'][i] = my_ols(data = dgp_2, 
                                     outcome = 'outcome', 
                                     exog = covariates_2, 
                                     treat = 'treatment')
        
        results_2['DRE'][i] = my_dre(data = dgp_2, 
                                     outcome = 'outcome', 
                                     exog = covariates_2,
                                     treat = 'treatment')

        # compute and store estimates of OLS and DRE for draw i and dgp_3        
        results_3['OLS'][i] = my_ols(data = dgp_3, 
                                     outcome = 'outcome', 
                                     exog = covariates_3, 
                                     treat = 'treatment')
        
        results_3['DRE'][i] = my_dre(data = dgp_3, 
                                     outcome = 'outcome', 
                                     exog = covariates_3, 
                                     treat = 'treatment')
        
    # return data frames with results
    return results_1, results_2, results_3 

# --------------------------------------------------------------------------- #
# PRINTOUT
# --------------------------------------------------------------------------- #

def my_printout(data, true_ate, title):
    """
    Calculates the performance measures of the simulation.
    
    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: table with estimated ATEs of OLS and DRE
    truth : TYPE: int
        DESCRIPTION: value of the true ATE
    title : TYPE: str
        DESCRIPTION: title of the printout

    Returns
    -------
    None. Prints results table.

    """
    # create data frame to store results
    df_print = pd.DataFrame(np.nan, 
                            index = ['Bias', 'Variance', 'MSE'], 
                            columns = ['OLS', 'DRE'])
    
    # caculate bias of OLS and DRE estimates
    bias_ols = np.mean(data['OLS']) - true_ate
    bias_dre = np.mean(data['DRE']) - true_ate
    
    # calculate variance of OLS and DRE estimates
    variance_ols = np.mean((data['OLS'] - np.mean(data['OLS']))**2)
    variance_dre = np.mean((data['DRE'] - np.mean(data['DRE']))**2)
    
    # calculate MSE of OLS and DRE estimates
    mse_ols = bias_ols**2 + variance_ols
    mse_dre = bias_dre**2 + variance_dre
    
    # store results in data frame
    df_print.loc['Bias'] = pd.Series({'OLS':bias_ols, 
                                      'DRE':bias_dre})
    
    df_print.loc['Variance'] = pd.Series({'OLS':variance_ols, 
                                          'DRE':variance_dre})
    
    df_print.loc['MSE'] = pd.Series({'OLS':mse_ols, 
                                     'DRE':mse_dre})
    
    # print results table
    print(('Results: ' + str(title)), '-' * 80, round(df_print, 4), 
          '-' * 80, '\n', sep='\n')

# --------------------------------------------------------------------------- #
# PLOTS
# --------------------------------------------------------------------------- #

def my_plot(data, true_ate, title = 'Histogram'):
    """
    Creates a histogram for the estimated ATEs of the simulation.
    
    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: table with estimated ATEs of OLS and DRE
    truth : TYPE: int
        DESCRIPTION: value of true ATE
    title : TYPE: str
        DESCRIPTION: title of the plot, default = 'Histogram'

    Returns
    -------
    None. Shows histogram.
    """
    # get bin edges
    edges = np.histogram(np.hstack((data['OLS'], data['DRE'])), bins = 50)[1]
      
    # add histogram for OLS ATEs
    plt.hist(data['OLS'], bins = edges, color='blue', label="OLS", alpha=0.5)
        
    # add histogram for DRE ATEs
    plt.hist(data['DRE'], bins = edges, color='orange', label="DRE", alpha=0.5)
    
    # add vertival line with true ATE
    plt.axvline(x = true_ate, label = "True ATE", color='red')
    
    # customize plot
    plt.title(title)
    plt.xlabel('ATE Estimation')
    plt.ylabel('Frequency')
    plt.grid(axis = 'y', alpha = 0.75)
    plt.legend()
    
    # show plot
    plt.show()
