import abc
import functools
import numpy as np
from unimatch.codes import paulitools as pt
from unimatch.codes import ErrorModel 
import random 

class SimpleErrorModel(ErrorModel):
    """
    Implements a simple IID error model that generates an error based on the number of qubits and the probability
    distribution.
    """

    @abc.abstractmethod
    def probability_distribution(self, probability):
        """
        Abstract meth.
        """

    def generate(self, code, probability, rng=None):
        """
        Generates probability distribution on the code.
        """
        rng = np.random.default_rng() if rng is None else rng
        n_qubits = code.n_k_d[0]
        error_pauli = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits,
            p=self.probability_distribution(probability)
        ))
        return pt.pauli_to_bsf(error_pauli)


class BitFlipErrorModel(SimpleErrorModel):
    """
    Implements a bit-flip error model.
    The probability distribution for a given error probability p is:
    * (1 - p): I (i.e. no error)
    * p: X
    * 0: Y
    * 0: Z
    """
    @functools.lru_cache()
    def probability_distribution(self, probability):
        p_x = probability
        p_y = p_z = 0
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z


#########################################################
# Mapped noise as a "bad" entropy model on red squares of the color code
# Need to incorporate this in the code. 

def BadEntropyErrorModel(my_code, p):

    """
    Create bad entropy noise on red plaquettes.
    Returns a numpy array of length number of qubits, where the necessary qubits are flipped. 
    """

    cells = my_code._red_plaquette_indices
    error = np.zeros(my_code.n_k_d[0])

    for cell in cells:
        sites = my_code.plaquette_sites[cell]
        b = np.random.choice([0, 1], p=[1-p, p]) # flip a pair with probability p
        if b == 1:
            pair = random.choices([np.array(sites[:2]), np.array(sites[::2]), np.array(sites[1:3])])
            error[tuple(pair)] = 1
        else:
            pass
    return error