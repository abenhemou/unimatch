"""Note: TO DO - some of the methods here can be merged for conciseness and 
better code quality."""

import functools
import itertools
import numpy as np
from unimatch.codes.model import StabilizerCode 
from unimatch.codes.color import Color488Pauli

class Color488Code(StabilizerCode):
    """
    Implements a 4.8.8 color code.
    * Indices are in the format (row, column).
    For example, a size 4 square lattice with site indices in (parentheses) and plaquette indices in [brackets]:

        (0,0)---(0,2) ------------ (0,6)---(0,8)
          |   [1,1] |    [1,4]      |   [1,7] |
          |         |               |         |
        (2,0)---(1,1)              (1,4)---(1,5)
          |           \             /         |
          |   [4,1]     \         /  [4,7]    |
          |            (2,2)---(2,3)          |
          |             |   [4,4] |           |
          |             |         |           |
          |            (3,2)---(3,3)          |
          |             /         \           |
          |           /             \         |
        (6,0)---(4,1)              (4,4)---(4,5)
          |   [7,1] |               |   [7,7] |
          |         |    [7,4]      |         |
        (8,0)---(5,1) ------------ (5,4)---(5,5)

    """

    MIN_SIZE = 4

    def __init__(self, size):
        """
        Initialise new square color 4.8.8 code.
        """
        self._size = size


    @property
    @functools.lru_cache()
    def n_k_d(self):
        """Code parameters."""
        # e.g. size=7: k=1, d=7,  n=7+6+5+5+4+3+3+2+1+1=(d+1)d/2+((d-1)/2)^2=(3d^2+1)4
        return  2 * (self.size ** 2) - 4*self.size + 4, 1 ,self.size


    #@property
    @functools.lru_cache()
    def stabilizers(self, restricted=None):
        """Stabilizers of the 4.8.8 color code."""  
        return np.array( [self.new_pauli().plaquette('Z', i).to_bsf() for i in self._plaquette_indices if self.new_pauli().colour(i) != restricted] ) 
        # we only keep the Z stabilizers here for now
        #+ [self.new_pauli().plaquette('X', i).to_bsf() for i in self._plaquette_indices] )


    @property
    @functools.lru_cache()
    def logical_xs(self):
        """X logical operator basis."""
        return np.array([self.new_pauli().logical_x().to_bsf()])


    @property
    @functools.lru_cache()
    def logical_zs(self):
        """Z logical operator basis."""
        return np.array([self.new_pauli().logical_z().to_bsf()])


    @property
    def num_plaquettes(self):
        """Number of plaquettes in the code."""
        return len(self._plaquette_indices)
    
    # contained corners 
    @property
    def logical_crossings(self):

        return [(self._plaquette_indices.index(i), self.num_plaquettes) for i in self._plaquette_indices if i[1] == (self.bound - 1)] 
    
    # No corners 
    # @property
    # def logical_crossing(self):
    #     return [(self._plaquette_indices.index(i), self.num_plaquettes) for i in self._plaquette_indices if (i[1] == 1 and i[0] not in (1,self.bound-1)) ] 

    @property
    @functools.lru_cache()
    def logical_crossing(self):
        """
        Returns a list of edges of the matching graph which cross the logical operator 
        representative used in the decoding simulation.
        """
        if (self.bound / 2) % 2 == 1:
            c_i = int((self.bound / 2) - 3)
            c_j = int(self.bound / 2)
        elif (self.bound / 2) % 2 == 0:
            c_i = int((self.bound / 2) - 3)
            c_j = int(self.bound / 2)
        
        left_plaq = [self._plaquette_indices.index(i) for i in self._plaquette_indices if (i[1] == c_i)] #and i[0])]# not in (1,my_code.bound-1))] 
        right_plaq = [self._plaquette_indices.index(j) for j in self._plaquette_indices if (j[1] == c_j)] # and j[0])] # not in (1,my_code.bound-1))] 
        return list(zip(left_plaq, right_plaq))


    # @property 
    # @functools.lru_cache()
    # def red_logical_crossings(self):
    #     """
    #     .
    #     """
    #     pind = self._plaquette_position_restricted['red']
    #     p_list = [i for i in pind if i % (2*(self.size - 1)) == 1 or i % (self.size - 1) == 0]
    #     bulk_edges = [(j, p) for j in p_list for p in p_list if np.abs(p-j) in [self.size-2, self.size]]
    #     boundary_edges = [(1, self.num_plaquettes), (self.num_plaquettes, 1), (pind[-1], self.num_plaquettes), (self.num_plaquettes, pind[-1])]

    #     return bulk_edges  + boundary_edges
    
    # @property
    # def logical_crossing(self):
    
    #     if (self.bound / 2) % 2 == 1:
    #         r_i = int((self.bound / 2) - 3)
    #         r_j = int(self.bound / 2)
    #     elif (self.bound / 2) % 2 == 0:
    #         r_i = int((self.bound / 2) - 3)
    #         r_j = int(self.bound / 2)
        
    #     left_plaq = [self._plaquette_indices.index(i) for i in self._plaquette_indices if (i[0] == r_i)] #and i[0])] not in (1,my_code.bound-1))] 
    #     right_plaq = [self._plaquette_indices.index(j) for j in self._plaquette_indices if (j[0] == r_j)] # and j[0])] # not in (1,my_code.bound-1))] 
    #     return list(zip(left_plaq, right_plaq))

    @property
    def size(self):
        """
        Size of any side of the square lattice in terms of number of qubits.
        """
        return self._size

    @property
    def bound(self): 
        """
        Maximum value that an index coordinate can take.
        """
        return 3*(self.size - 1) - 1

  
    def is_plaquette(self, index, restricted=None):
        """
        Return True if the index specifies a plaquette, irrespective of lattice bounds,

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index specifies a plaquette
        :rtype: bool
        """
        r, c = index
        a = (r - 1) % 3
        b = (c - 1) % 3

        return (a,b) == (0,0) and self.new_pauli().colour((r,c)) != restricted

    @classmethod
    def is_site(self, index):
        """
        Return True if the index specifies a site on the code, irrespective of lattice bounds.
        """
        r, c = index
        if r % 6 in [0, 2] and c % 6 in [0, 2]:
            return True
        elif r % 6 in [3, 5] and c % 6 in [3, 5]:
            return True
        else:
            return False

    def is_in_bounds(self, index):
        """
        Return True if the index is within lattice bounds inclusive, irrespective of object type.
        """
        r, c = index 
        return 0 <= c <= self.bound and 0 <= r <= self.bound


    @property
    @functools.lru_cache()
    def _plaquette_indices(self):
        """
        Return a list of the plaquette indices on the lattice.
        """
        return [i for i in itertools.product(range(self.bound + 1), repeat=2)
                if self.is_in_bounds(i) and self.is_plaquette(i)] 
    
    @property
    @functools.lru_cache()
    def _site_indices(self):
        """
        Return a list of the site indices of the lattice.
        """
        return [i for i in itertools.product(range(self.bound + 1), repeat=2)
                if self.is_in_bounds(i) and self.is_site(i)] 

    @property
    @functools.lru_cache()
    def _plaquette_indices_restricted(self):
        """
        Return a list of the plaquette indices of the lattice.
        * Each index is in the format (row, column).
        * Indices are in order of increasing column and row.
        * Color is the RESTRICTED color
        """
        return {color: [i for i in itertools.product(range(self.bound + 1), repeat=2)
                        if self.is_in_bounds(i) and self.is_plaquette(i, color)] 
                for color in ['green', 'blue', 'red'] }

    @property
    @functools.lru_cache()
    def _plaquette_position_restricted(self):
        """
        Return a list of the plaquette indices of the restricted lattice, not as a tuple coordinate
        but as the order on the lattice.
        * Each index is in the format (row, column).
        * Indices are in order of increasing column and row.
        * Color is the RESTRICTED color
        """
        return {color: [self._plaquette_indices.index(i) for i in itertools.product(range(self.bound + 1), repeat=2)
                        if self.is_in_bounds(i) and self.is_plaquette(i, color)] 
                for color in ['green', 'blue', 'red'] }


    @property
    @functools.lru_cache(maxsize=None)
    def _red_plaquette_indices(self):
        """
        Return a list of the plaquette indices (not coordinates!) of the lattice.
        """
        return [self._plaquette_indices.index(i) for i in self._plaquette_indices if self.new_pauli().colour(i) == 'red']

    def site_list(self, index): # for each plaquette
        r, c = index 
        if self.new_pauli().colour(index) == 'red':
            sites = ((r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1))
        else:
            sites = ((r-2, c-1), (r-2, c+1), (r+2, c-1), (r+2, c+1), (r-1, c-2), (r-1, c+2), (r+1, c-2), (r+1, c+2))

        return [self.new_pauli()._flatten_site_index(i) for i in sites if self.is_in_bounds(i)]

    @property
    @functools.lru_cache(maxsize=None)
    def plaquette_sites(self):
       """
       Return a list of the sites on each plaquette.
       """
       return [self.site_list(i) for i in self._plaquette_indices]

    @property
    @functools.lru_cache(maxsize=None)
    def boundary_sites(self):
       """
       Return a list of the sites on the boundary of the code. 
       """
       return [self.new_pauli()._flatten_site_index(i) for i in self._site_indices if i[0] == 0 or i[1] == 0 or i[0] == self.bound or i[1] == self.bound]
 
    @property
    @functools.lru_cache(maxsize=None)
    def boundaries(self):
       """
       Returns separate lists of the sites on the boundary of the code. 
       """
       top = [self.new_pauli()._flatten_site_index(i) for i in self._site_indices if i[0] == 0]
       bottom = [self.new_pauli()._flatten_site_index(i) for i in self._site_indices if i[0] == self.bound]
       left = [self.new_pauli()._flatten_site_index(i) for i in self._site_indices if i[1] == 0]
       right = [self.new_pauli()._flatten_site_index(i) for i in self._site_indices if i[1] == self.bound]
       return [top, bottom, left, right]

    @property
    @functools.lru_cache(maxsize=None)
    def bulk_ids(self):
       """
       Return bulk ids by color in a dictionary. 
       """
       reds = []
       blues = []
       greens = []

       for id in range(3*self.n_k_d[0]-2*self.n_k_d[2]-1):
           if id in range(self.n_k_d[0]) and id not in self.boundary_sites:
               reds.append(id)
           elif id in range(self.n_k_d[0], 2*self.n_k_d[0]-2*self.n_k_d[2]):
               blues.append(id)
           elif id in range(2*self.n_k_d[0]-2*self.n_k_d[2], 3*self.n_k_d[0]-4*self.n_k_d[2]):
               greens.append(id)
       return [reds, blues, greens, 'red-blue-green']
 
    @property
    @functools.lru_cache()
    def red_dictionary(self):
        red_dict = {}
        for i in self._red_plaquette_indices :
            red_dict[i] = {"qubits":self.plaquette_sites[i]}
        return red_dict

    @property
    @functools.lru_cache()
    def corners(self):
        corners = [(0, 0), (0, self.bound), (self.bound, 0), (self.bound, self.bound)]
        return [self.new_pauli()._flatten_site_index(i) for i in corners]

    # def syndrome_to_plaquette_indices(self, syndrome):
    #     """
    #     Returns the indices of the plaquettes associated with the non-commuting zerszers identified by the syndrome.

    #     :param syndrome: Binary vector identifying commuting and non-commuting stabilizers by 0 and 1 respectively.
    #     :type syndrome: numpy.array (1d)
    #     :return: Two sets of plaquette indices (first set sfor X stabilizers, second for Z stabilizers).
    #     :rtype: set of 2-tuple of int, set of 2-tuple of int
    #     """
    #     x_syndrome, z_syndrome = np.hsplit(syndrome, 2)
    #     return (set(tuple(index) for index in np.array(self._plaquette_indices)[x_syndrome.nonzero()]),
    #             set(tuple(index) for index in np.array(self._plaquette_indices)[z_syndrome.nonzero()]))

    def syndrome_to_plaquette_indices(self, syndrome, restricted=None):
        """
        Returns the indices of the plaquettes associated with the non-commuting stabilizers identified by the syndrome.
        """
        if restricted is not None:
            indices = (set(tuple(index) for index in np.array(self._plaquette_indices_restricted[restricted])[syndrome.nonzero()]) )
        else:
            indices = (set(tuple(index) for index in np.array(self._plaquette_indices)[syndrome.nonzero()]) )
        return indices

    def full_syndrome_to_plaquette_indices(self, syndrome, restricted=None):
        """
        Returns the indices of the plaquettes associated with the non-commuting stabilizers identified by the syndrome. 

        """
        if restricted is not None:
            indices = (set(tuple(index) for index in np.array(self._plaquette_indices_restricted[restricted])) )
        else:
            indices = (set(tuple(index) for index in np.array(self._plaquette_indices)) )
        return indices


    @property
    @functools.lru_cache(maxsize=2 ** 15)
    def _site_label(self):
        """
        Labelling of sites based on their location (corner boundary of bulk). 
        This is used to assign weights to edges triggered by error mechanisms on these 
        sites.
        """

        labels = []
        rg_sides = list(set(self.boundaries[0] + self.boundaries[1]) - set(self.corners))
        rb_sides = list(set(self.boundaries[2] + self.boundaries[3]) - set(self.corners))

        for site in range(self.n_k_d[0]*3 - 4*self.size):
            
            if site in self.corners:
                labels.append('corner')

            elif site in rb_sides:
                labels.append('rb boundary')

            elif site in rg_sides:
                labels.append('rg boundary')

            elif site in self.bulk_ids[0]:
                labels.append('gb bulk')
            # else:
            #     labels.append('r bulk')
            elif site in self.bulk_ids[1]:
                labels.append('rg bulk')
    
            elif site in self.bulk_ids[2]:
                labels.append('rb bulk')
        return labels
    

    def __hash__(self):
        return hash(self._size)


    def new_pauli(self, bsf=None):
        """
        Constructor of color 4.8.8 Pauli for this code in 
        binary symplectic representation. 
        """
        return Color488Pauli(self, bsf)
