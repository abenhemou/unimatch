"""This file contains useful functions borrowed from David Tuckett's qecsim package."""

import abc
import functools
import numpy as np

class StabilizerCode(metaclass=abc.ABCMeta):
    """
    Defines stabilizer code properties and methods.
    """

    @property
    @abc.abstractmethod
    def stabilizers(self):
        """
        Stabilizer generators as binary symplectic vector or matrix.
        """

    @property
    @abc.abstractmethod
    def logical_xs(self):
        """
        Logical X operators as binary symplectic vector or matrix.
        Notes:
        * Each row is a logical X operator.
        * The order of logical X operators matches that of logical Z operators given by :meth:`logical_zs`, with one for
          each logical qubit.
        """

    @property
    @abc.abstractmethod
    def logical_zs(self):
        """
        Logical Z operators as binary symplectic vector or matrix.
        Notes:
        * Each row is a logical Z operator.
        * The order of logical Z operators matches that of logical X operators given by :meth:`logical_xs`, with one for
          each logical qubit.
        """

    @property
    @functools.lru_cache()
    def logicals(self):
        """
        Logical operators as binary symplectic matrix.
        Notes:
        * Each row is a logical operator.
        * All logical X operators are stacked above all logical Z operators.

        """
        return np.vstack((self.logical_xs, self.logical_zs))

    @property
    @abc.abstractmethod
    def n_k_d(self):
        """
        Descriptor of the code parameters.  
        """


class ErrorModel(metaclass=abc.ABCMeta):
    """
    Defines error model properties and methods. 
    """

    def probability_distribution(self, probability):
        """
        Return the single-qubit probability distribution amongst Pauli I, X, Y and Z.
        """
        
    @abc.abstractmethod
    def generate(self, code, probability, rng=None):
        """
        Generate new error with probability p.
        """
