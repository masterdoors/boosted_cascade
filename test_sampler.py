'''
Created on Oct 1, 2023

@author: keen
'''
from formal_models.pcfg_length_sampling import LengthSampler
from formal_models.pcfg_tools import (
    remove_epsilon_rules, remove_unary_rules)

from grammars.unmarked_reversal import UnmarkedReversalGrammar
import random


grammar = UnmarkedReversalGrammar(2,40)
remove_epsilon_rules(grammar)
remove_unary_rules(grammar)

sampler = LengthSampler(grammar)
generator = random.Random()

print([list(sampler.sample(40, generator))
        for i in range(10)])
