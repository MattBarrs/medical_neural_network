import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from GibbsSamplingWithEvidence import GibbsSampling

import numpy as np
import pandas as pd
import argparse
import re
import math as m

def printDashLine(x):
    i=0
    print('\n')
    while i<x:
        print('-----------------------------------------------------------')
        i=i+1


def initaliseCPDS():

    cpd_asia = TabularCPD(variable='asia', variable_card=2,
                            values=[[0.01],
                                    [0.99]])

    cpd_tub = TabularCPD(variable='tub', variable_card=2,
                            values=[[0.99,0.95],
                                    [0.01, 0.05] ],
                            evidence=['asia'], evidence_card=[2])

    cpd_smoke = TabularCPD(variable='smoke', variable_card=2,
                            values=[[0.5],
                                    [0.5]])

    cpd_bron = TabularCPD(variable='bron', variable_card=2,
                            values=[[0.7, 0.4],
                                    [0.3, 0.6]],
                            evidence=['smoke'], evidence_card=[2])

    cpd_lung = TabularCPD(variable='lung', variable_card=2,
                            values=[[0.99, 0.9],
                                    [0.01, 0.1]],
                            evidence=['smoke'], evidence_card=[2])

    cpd_either = TabularCPD(variable='either', variable_card=2,
                            values=[[0.999, 0.001, 0.001, 0.001],
                                    [0.001, 0.999, 0.999, 0.999]],
                            evidence=['tub', 'lung'],
                            evidence_card=[2, 2])

    cpd_dysp = TabularCPD(variable='dysp', variable_card=2,
                            values=[[0.9, 0.2, 0.3, 0.1],
                                    [0.1, 0.8, 0.7, 0.9]],
                            evidence=['either', 'bron'],
                            evidence_card=[2, 2])

    cpd_xray = TabularCPD(variable='xray', variable_card=2,
                            values=[[0.95, 0.02],
                                    [0.05, 0.98]],
                            evidence=['either'], evidence_card=[2])

    # Associating the parameters with the model structure.
    asia_model.add_cpds(cpd_asia, cpd_smoke, cpd_tub, cpd_bron, cpd_lung, cpd_either, cpd_dysp, cpd_xray)


def varElimFunction():

    varList = {}
    q = {}
    #Run variable elimination
    asia_infer = VariableElimination(asia_model)

    #Show VariableElimination for all given queries
    for qItem in query:
        q = asia_infer.query([qItem], evidence=evidence)
        print(q[qItem])

        #create a list of exact probability in a dictionary
        varList[qItem] = (q[qItem].values)
    return(varList)

def gibbsSampleingFunction():
    #create sampler,
    gibbs_sampler = GibbsSampling(asia_model)
    #create samples,
    samples = gibbs_sampler.sample(size=N, evidence = evidence )
    #fit to model
    asia_model.fit(samples)

    #prints cpd for each query variable
    for cpd in asia_model.get_cpds() :
        for qItem in query:

            if qItem == cpd.variable:
                print("CPD of {variable}".format(variable = cpd.variable))
                print(cpd)

                printDashLine(1)

    for column in query:
        trueCount[column] = 0

        for value in samples[column]:
            if(value == 1):
                trueCount[column] =  trueCount[column] + 1
        estimProb[column] =  trueCount[column]/float(N)

    printDashLine(1)
    return(estimProb)

def xEntroFunction(estimProb, xactProb):
    totalSum = 0

    #calculates entropy for each query Variable
    for qItem in query:
        xact = xactProb[qItem][1]
        #Adds very small number to avoid 0 error when using log
        estim =  estimProb[qItem]

        temp = (1-estim)*m.log10(1-xact)  + estim*m.log10(xact)
        print(qItem ," Estim:", estim, "Exac: ", xact, "Temp: ", temp)

        totalSum = totalSum + temp
    return -(totalSum)


print("******************************START******************************")

# This parses the inputs from test_asia.py (You don't have to modify this!)
parser = argparse.ArgumentParser(description='Asia Bayesian network')
parser.add_argument('--evidence', nargs='+', dest='eVars')
parser.add_argument('--query', nargs='+', dest='qVars')
parser.add_argument('-N', action='store', dest='N')
parser.add_argument('--exact', action="store_true", default=False)
parser.add_argument('--gibbs', action="store_true", default=False)
parser.add_argument('--ent', action="store_true", default=False)

args = parser.parse_args()
evidence = {}
trueCount = {}
estimProb = {}

if args.N is not None:
    N = int(args.N)

query = args.qVars
for item in args.eVars:
    evidence[re.split('=|:', item)[0]] = int(re.split('=|:',item)[1])

# Starting with defining the network structure
asia_model = BayesianModel([('asia', 'tub'),
                            ('smoke', 'lung'), ('smoke', 'bron'),
                            ('tub', 'either'), ('lung', 'either'),
                            ('bron', 'dysp'),('either', 'dysp'),
                            ('either', 'xray')])

#Intialises the CPDs for the baysian model
initaliseCPDS()

print("evidence: ", evidence)
print('query: ', query)
printDashLine(1)

if args.exact is True:
    print("Variable Elimination: True;")

    realProb = varElimFunction()
    printDashLine(1)


if args.gibbs is True:
    print("Gibbs Sampling: True;\nOutputting results...\n")

    estimProb = gibbsSampleingFunction()

    print("Estimated probabilities: ")
    print(estimProb)

    printDashLine(1)

if args.ent is True:
    print("cross-entropy: True;\nOutputting results...\n")

    xEntro = (xEntroFunction(estimProb, realProb))

    print("Cross-entropy calculated as:")
    print(xEntro)

    printDashLine(1)


# print("asia_model.get_independencies(): ", asia_model.get_independencies())
print("******************************END******************************")
