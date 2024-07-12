# Monte Carlo Study: OLS vs. Doubly Robust Estimator

## Overview

This project compares the performance of Ordinary Least Squares (OLS) and Doubly Robust Estimator (DRE) in estimating the Average Treatment Effect (ATE) using Monte Carlo simulations. The study is based on predefined data generating processes (DGPs) and aims to provide insights into the conditions under which each estimator performs better.

## Date

01.05.2022

## Project Structure

- `ols_vs_dre_functions.py`: Contains functions for data generation, estimation, simulation, and result visualization.
- `main.py`: The main script to run the simulations and generate results.

## Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sandrogassner/ols_vs_dre_monte_carlo_study.git
    cd ols_vs_dre_monte_carlo_study
    ```

2. Install the required packages:
    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```

## Usage

1. **Set the working directory**:
    Update the `PATH` variable in `main.py` to your current working directory.

    ```python
    PATH = 'YOUR_PATH'
    ```

2. **Run the simulations**:
    Execute the `main.py` script to run the simulations and generate the results.

    ```bash
    python main.py
    ```

3. **View the results**:
    The results of the simulations, including performance measures and plots, will be printed to the console and displayed as plots.

## Simulation Details

### Data Generating Processes (DGPs)

Three different DGPs are used in this study:

1. **DGP 1**:
    - Linear model with homogeneous treatment effect.
    - The true ATE is known to be 3.
    - Expected to perform better with OLS.

    ```python
    y = 4 + 3d + 5*X1 - 2*X2 - X3 + u
    ```

2. **DGP 2**:
    - Treatment effect is not homogeneous.
    - The true ATE is known to be 3.
    - Expected to perform better with DRE.

    ```python
    y = d * Y1 + (1-d) * Y0
    ```

3. **DGP 3**:
    - One identifying assumption is violated.
    - The true ATE is known to be 3.
    - Both OLS and DRE are expected to perform poorly.

    ```python
    y = 2 + 3d + 3*X1 - 2*X2 + X3 + u
    ```

### Estimation Methods

1. **Ordinary Least Squares (OLS)**:
    - Estimates parameters by minimizing the sum of squared residuals.
    - Suitable for linear models with homogeneous effects.

    ```math
    \text{ATE} = \alpha
    ```

    Where the model is specified as:

    ```math
    E[Y | D = d, X = x] = \mu(d, x) = d\alpha + x\beta
    ```

2. **Doubly Robust Estimator (DRE)**:
    - Combines propensity score weighting and regression adjustment.
    - Provides consistent estimates if either the propensity score model or the outcome model is correctly specified.

    Steps involved:

    - Estimate the propensity score $`\hat{p}(x_i)`$ by a logit or probit model.
    - Estimate the outcome equation \hat{\mu}(1, x_i) and \hat{\mu}(0, x_i) by parametric models.
    - Calculate the estimated ATE using the formula:

    ```math
    \hat{ATE}_{DR} = \frac{1}{N} \sum_{i=1}^{N} \left[ \hat{\mu}(1, x_i) - \hat{\mu}(0, x_i) + \frac{d_i(y_i - \hat{\mu}(1, x_i))}{\hat{p}(x_i)} - \frac{(1 - d_i)(y_i - \hat{\mu}(0, x_i))}{1 - \hat{p}(x_i)} \right]
    ```

### Performance Measures

- **Bias**: Difference between the average estimated ATE and the true ATE.
- **Variance**: Variability of the estimated ATEs.
- **Mean Squared Error (MSE)**: Sum of the variance and the squared bias, providing an overall measure of estimator performance.

## Results

The results are based on 1,000 simulations for each DGP with 1,000 observations per simulation. The performance of OLS and DRE is compared in terms of bias, variance, and MSE.

### Example Results

#### DGP 1

| Measure  | OLS   | DRE   |
|----------|-------|-------|
| Bias     | 0.007 | 0.018 |
| Variance | 0.289 | 0.837 |
| MSE      | 0.289 | 0.837 |

#### DGP 2

| Measure  | OLS   | DRE   |
|----------|-------|-------|
| Bias     | 1.743 | 0.005 |
| Variance | 0.022 | 0.031 |
| MSE      | 3.058 | 0.031 |

#### DGP 3

| Measure  | OLS   | DRE   |
|----------|-------|-------|
| Bias     | 0.878 | 1.030 |
| Variance | 0.048 | 0.054 |
| MSE      | 0.819 | 1.115 |

## Conclusion

This study demonstrates the varying performance of OLS and DRE under different data generating processes. OLS performs better in linear models with homogeneous effects, while DRE is more robust in cases with heterogeneous effects. Both estimators struggle when key assumptions are violated.
