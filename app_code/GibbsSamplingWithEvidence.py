from collections import namedtuple
import itertools

import networkx as nx
import numpy as np

from pgmpy.factors import factor_product
#from pgmpy.inference import Inference
from pgmpy.models import BayesianModel, MarkovChain, MarkovModel
from pgmpy.utils.mathext import sample_discrete
from pgmpy.extern.six.moves import map, range
from pgmpy.sampling import _return_samples


State = namedtuple('State', ['var', 'state'])

class GibbsSampling(MarkovChain):
    """
    Class for performing Gibbs sampling.

    Parameters:
    -----------
    model: BayesianModel or MarkovModel
        Model from which variables are inherited and transition probabilites computed.

    Public Methods:
    ---------------
    set_start_state(state)
    sample(evidence, start_state, size)

    """
    def __init__(self, model=None):
        super(GibbsSampling, self).__init__()
        if isinstance(model, BayesianModel):
            self._get_kernel_from_bayesian_model(model)
        elif isinstance(model, MarkovModel):
            self._get_kernel_from_markov_model(model)

    def _get_kernel_from_bayesian_model(self, model):
        """
        Computes the Gibbs transition models from a Bayesian Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters:
        -----------
        model: BayesianModel
            The model from which probabilities will be computed.
        """
        self.variables = np.array(model.nodes())
        self.cardinalities = {var: model.get_cpds(var).variable_card for var in self.variables}

        for var in self.variables:
            other_vars = [v for v in self.variables if var != v]
            other_cards = [self.cardinalities[v] for v in other_vars]
            cpds = [cpd for cpd in model.cpds if var in cpd.scope()]
            prod_cpd = factor_product(*cpds)
            kernel = {}
            scope = set(prod_cpd.scope())
            for tup in itertools.product(*[range(card) for card in other_cards]):
                states = [State(v, s) for v, s in zip(other_vars, tup) if v in scope]
                prod_cpd_reduced = prod_cpd.reduce(states, inplace=False)
                kernel[tup] = prod_cpd_reduced.values / sum(prod_cpd_reduced.values)
            self.transition_models[var] = kernel

    def _get_kernel_from_markov_model(self, model):
        """
        Computes the Gibbs transition models from a Markov Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters:
        -----------
        model: MarkovModel
            The model from which probabilities will be computed.
        """
        self.variables = np.array(model.nodes())
        factors_dict = {var: [] for var in self.variables}
        for factor in model.get_factors():
            for var in factor.scope():
                factors_dict[var].append(factor)

        # Take factor product
        factors_dict = {var: factor_product(*factors) if len(factors) > 1 else factors[0]
                        for var, factors in factors_dict.items()}
        self.cardinalities = {var: factors_dict[var].get_cardinality([var])[var] for var in self.variables}

        for var in self.variables:
            other_vars = [v for v in self.variables if var != v]
            other_cards = [self.cardinalities[v] for v in other_vars]
            kernel = {}
            factor = factors_dict[var]
            scope = set(factor.scope())
            for tup in itertools.product(*[range(card) for card in other_cards]):
                states = [State(first_var, s) for first_var, s in zip(other_vars, tup) if first_var in scope]
                reduced_factor = factor.reduce(states, inplace=False)
                kernel[tup] = reduced_factor.values / sum(reduced_factor.values)
            self.transition_models[var] = kernel

    def sample(self, evidence=None, start_state=None, size=1, return_type="dataframe"):
        """
        Sample from the Markov Chain.

        Parameters:
        -----------
        start_state: dict or array-like iterable
            Representing the starting states of the variables. If None is passed, a random start_state is chosen.
        evidence: array-like iterable
            Representing states of the evidence variables
        size: int
            Number of samples to be generated.
        return_type: string (dataframe | recarray)
            Return type for samples, either of 'dataframe' or 'recarray'.
            Defaults to 'dataframe'

        Returns
        -------
        sampled: A pandas.DataFrame or a numpy.recarray object depending upon return_type argument
            the generated samples

        Examples:
        ---------
        >>> from pgmpy.models.BayesianModel import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from GibbsSamplingWithEvidence import GibbsSampling
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_d = TabularCPD('diff', 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD('intel', 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25,
        ...                0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...                ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> gibbs_sampler = GibbsSampling(student)
        >>> samples = gibbs_sampler.sample(size=5, evidence=[('grade',1)])
        >>> print(samples)
        diff  grade  intel
        0     1      1      1
        1     1      1      0
        2     0      1      0
        3     0      1      0
        4     1      1      0

        """
        if start_state is None and self.state is None:
            self.state = self.random_state()
        elif start_state is not None:
            self.set_start_state(start_state)

        #overwriting with evidence
        if evidence is not None:
            for j, (var, st) in enumerate(self.state):
                for key, value in evidence.items():
                #for k, (v_e, st_e) in enumerate(evidence):
                    if var == key:
                        #print(var, self.state[j])
                        self.state[j] = State(var, value)
                   

        types = [(var_name, 'int') for var_name in self.variables]
        sampled = np.zeros(size, dtype=types).view(np.recarray)
        sampled[0] = tuple([st for var, st in self.state])
        
        for i in range(size - 1):
            for j, (var, st) in enumerate(self.state):
                # check for evidence
                next_st = None
                if evidence is not None:
                    for v_e, st_e in evidence.items():
                        if var == v_e:
                            next_st = st_e
                
                if next_st is None:
                    other_st = tuple(st for v, st in self.state if var != v)
                    next_st = sample_discrete(list(range(self.cardinalities[var])),
                                              self.transition_models[var][other_st])[0]
                self.state[j] = State(var, next_st)
            sampled[i + 1] = tuple([st for var, st in self.state])

        return _return_samples(return_type, sampled)

