>This is purely a theoretical problem/solution and should NOT be  used within the real world as a tool for diagnoses

#Overview
> Problem given by UoL 

- This problem uses a Python program to implement the fictitious ‘Asia’ network that models the following qualitative medical ‘Knowledge’:
    - Shortness-of-breath (dyspnoea) may be due to tuberculosis, 
    - Lung cancer
    - Bronchitis, 
    - None of them or more than one of them. 


- Within the problem it is said that:
    - A recent visit to Asia increases the chances of  tuberculosis,
    - Smoking is a risk factor for both lung cancer and bronchitis.
    - Results of a single chest X-ray does not discriminate between lung cancer and tuberculosis, as neither does the presence or absence of dyspnoea.
- The aim was to:
    - You will be helping clinicians to decide the probability of each disease given some evidence. The CPD values for this model, made of binary variables, are given in the Table


#Problem
- It must be:
    - A Bayesian network, with nodes/edges (showing directions of dependencies), CPDs, and given
some evidence, compute the posterior probabilities of the query variables. 
    - Network must use variable names:
        - asia,
        - smoke,
        - tub,
        - bron,
        - lung, 
        - either,
        - dysp,
        - xray 
    - Take evidence variables, computes, prints the exact (using VariableEliminiation
pgmpy class) and approximate posterior probabilities (using the provided GibbsSampling
class) for each of the query variables. 
    - The program should accept various input options
using the following switches:
        - --evidence A list of space separated variable:value pairs used as evidence
        - --query A list of space separated query variables
        - --exact Used to perform variable elimination
        - --gibbs Used to perform Gibbs sampling
        - -N Number of samples generated using Gibbs sampling
        - --ent Used to compute the cross-entropy between the exact and approximate
        distributions
  - Compute total cross-entropy between the approximated (q) and exact (p) posterior
probability distributions for all query variables.


#Usage
To use ensure all files are in the same directory
Use `qc_tester.py` to test application
It will be the format ` search( make_qc_problem(X,Y), ('A_star', zero_heuristic), Z, [])`
- Where X and Y are the board's dimensions
- And Z is the maximum search depth 