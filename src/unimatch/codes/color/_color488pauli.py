import numpy as np


class Color488Pauli:
    """
    Defines a Pauli operator on a color 4.8.8 lattice.
    (typically instantiated using unimatch.models.color.Color488Code.new_pauli (c.f. qecsim doc))
    """

    def __init__(self, code, bsf=None):
        """
        Initialise new color 4.8.8 Pauli.
        Note: Pauli representation in binary symplectic form here (keep 
            it this way for future generalisations to depolarising noise on
            use of more of the color code symmetries).
        """
        self._code = code
        self._from_bsf(bsf)

    def _from_bsf(self, bsf):
        # initialise lattices for X and Z operators from bsf
        n_qubits = self.code.n_k_d[0]
        if bsf is None:
            # initialise identity lattices for X and Z operators
            self._xs = np.zeros(n_qubits, dtype=int)
            self._zs = np.zeros(n_qubits, dtype=int)
        else:
            assert len(bsf) == 2 * n_qubits, 'BSF {} has incompatible length'.format(bsf)
            assert np.array_equal(bsf % 2, bsf), 'BSF {} is not in binary form'.format(bsf)
            # initialise lattices for X and Z operators from bsf
            self._xs, self._zs = np.hsplit(bsf, 2)  # split out Xs and Zs


    @property
    def site_dict(self):
        return dict(zip(self.code._site_indices, np.arange(self.code.n_k_d[0])))


    def _flatten_site_index(self, index):
        return self.site_dict[index]


    @property
    def code(self):
        """
        The color 4.8.8 code.
        """
        return self._code

    def copy(self):
        """
        Returns a copy of this Pauli that references the same code but is backed by a copy of the bsf.
        """
        return self.code.new_pauli(bsf=np.copy(self.to_bsf()))

    def operator(self, index):
        """
        Returns the operator on the site identified by the index.
        """
        # check valid in-bounds index
        if not (self.code.is_site(index) and self.code.is_in_bounds(self, index)):
            raise IndexError('{} is not an in-bounds site index for code of size {}.'.format(index, self.code.size))
        # extract binary x and z
        flat_index = self._flatten_site_index(index)
        x = self._xs[flat_index]
        z = self._zs[flat_index]
        # return Pauli
        if x == 1 and z == 1:
            return 'Y'
        if x == 1:
            return 'X'
        if z == 1:
            return 'Z'
        else:
            return 'I'

    def site(self, operator, *indices):
        """
        Apply the operator to site identified by the index.
        """
        for index in indices:
            # check valid index
            if not self.code.is_site(index):
                raise IndexError('{} is not a site index.'.format(index))
            # apply if index within lattice
            if self.code.is_in_bounds(index):
                # flip sites
                flat_index = self._flatten_site_index(index)
                if operator in ('X', 'Y'):
                    self._xs[flat_index] ^= 1
                if operator in ('Z', 'Y'): 
                    self._zs[flat_index] ^= 1
        return self


    #@staticmethod
    def colour(self, index):
        r, c = index
        if r % 6 == 4 and c % 6 == 1:
            return 'green'
        if r % 6 == 1 and c % 6 == 4:
            return 'blue'
        else:
            return 'red'


    def plaquette(self, operator, index):
        """
        Apply a plaquette operator at the given index.
        """
        r, c = index
        if not self.code.is_plaquette(index):
            raise IndexError('{} is not a plaquette index.'.format(index))
        
        if self.colour(index) == 'green' or self.colour(index) == 'blue' :
            self.site(operator, (r-2, c-1), (r-2, c+1), (r+2, c-1), (r+2, c+1), (r-1, c-2), (r-1, c+2), (r+1, c-2), (r+1, c+2))
        else:
            self.site(operator, (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1) )
        return self


    def plaquette_periodic(self, operator, index):
        """
        Apply a plaquette operator at the given index.
        """
        r, c = index
        m = self.code.bound - 2

        if not self.code.is_plaquette(index):
            raise IndexError('{} is not a plaquette index.'.format(index))
        
        if self.colour(index) == 'green' or self.colour(index) == 'blue' :
            self.site(operator, (r-2 % m, c-1 % m), (r-2 % m, c+1 % m), (r+2 % m, c-1 % m), (r+2 % m, c+1 % m), (r-1 % m, c-2 % m), (r-1 % m, c+2 % m), (r+1 % m, c-2 % m), (r+1 % m, c+2 % m))
        else:
            self.site(operator, (r-1 % m, c-1 % m), (r-1 % m, c+1 % m), (r+1 % m, c-1 % m), (r+1 % m, c+1 % m) )
        return self


    def logical_x(self):
        """
        Apply a logical X operator, i.e. column of X along leftmost sites.
        """
        for row in range(self.code.bound + 1):
            index = row, 0
            if self.code.is_site(index):
                self.site('X', index)
        return self


    def logical_z(self):
        """
        Apply a logical Z operator, i.e. row of Z along top sites.
        """
        for col in range(self.code.bound + 1):
            index = 0, col
            if self.code.is_site(index):
                self.site('Z', index)
        return self

    def __eq__(self, other):
        if type(other) is type(self):
            return np.array_equal(self._xs, other._xs) and np.array_equal(self._zs, other._zs)
        return NotImplemented

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self.code, self.to_bsf())

    def to_bsf(self):
        """
        Binary symplectic representation of Pauli.
        """
        return np.concatenate((self._xs, self._zs))
