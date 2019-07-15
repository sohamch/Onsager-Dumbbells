import numpy as np
from onsager import PowerExpansion as PE
from GFcalc import GFCrystalcalc
# import GFcalc
import itertools
from scipy.special import hyp1f1, gamma, expi, factorial

# from copy import deepcopy
# from numpy import linalg as LA
# from scipy.special import hyp1f1, gamma, expi #, gammainc
"""
GFcalc module for dumbbell interstitials
Inherits most aspects from the original GFcalc module written by Prof. Trinkle.
"""
T3D = PE.Taylor3D


class GF_dumbbells(GFCrystalcalc):
    """
    Class calculator for the Green function, designed to work with the Crystal class.

    Highly similar to vacany GF calculator, indexing of jumps is in (i,or) list rather than basis set (i).
    """

    def __init__(self, container, jumpnetwork, Nmax=4, kptwt=None):
        """
        Initializes our calculator with the appropriate topology / connectivity. Doesn't
        require, at this point, the site probabilities or transition rates to be known.

        :param container: Object containing all dumbbell state information.
        :param iorlist: flat list of (basis_index,orientation) tuples -> analog of basis[chem]
        :param symorlist: (basis_index,orientation) pairs grouped into symmetrically unique lists-> analog of sitelist
        :param jumpnetwork: list of unique transitions as lists of (i,j, dx, c1, c2)
                            "Note here i and j are indices into iorlist"
                            "Needs to be ensured that the jumpnetwork belongs to the container"
        :param Nmax: maximum range as estimator for kpt mesh generation
        :param kptwt: (optional) tuple of (kpts, wts) to short-circuit kpt mesh generation
        """
        # this is really just used by loadHDF5() to circumvent __init__
        if all(x is None for x in (container, jumpnetwork)): return
        if any(len(tup) != 2 for l in jumpnetwork for tup in l):
            raise TypeError("Need the indexed form of the jumpnetwork.")
        self.container = container
        self.crys = container.crys
        self.chem = container.chem
        self.iorlist = container.iorlist.copy()
        self.symorlist = container.symorlist.copy()
        self.N = len(self.iorlist)  # N - no. of dumbbell states
        self.Ndiff = self.networkcount(jumpnetwork, self.N)
        # Create invmap - which symmety-grouped (i,or) pair list in symorlist
        # does a given (i,or) pair in iorlist belong to
        self.invmap = container.invmap.copy()
        self.NG = len(self.crys.G)  # number of group operations
        # self.invmap = container.invmap
        self.grouparray, self.indexpair = self.BreakdownGroups()
        # modified BreakdownGroups using new indexmap for dumbbells
        bmagn = np.array([np.sqrt(np.dot(self.crys.reciplatt[:, i], self.crys.reciplatt[:, i]))
                          for i in range(3)])
        bmagn /= np.power(np.product(bmagn), 1 / 3)
        # make sure we have even meshes - same as vacancies
        # Don't need to change the k-mesh generation.
        self.kptgrid = np.array([2 * np.int(np.ceil(2 * Nmax * b)) for b in bmagn], dtype=int) \
            if kptwt is None else np.zeros(3, dtype=int)
        self.kpts, self.wts = self.crys.reducekptmesh(self.crys.fullkptmesh(self.kptgrid)) \
            if kptwt is None else deepcopy(kptwt)
        self.Nkpt = self.kpts.shape[0]
        # generate the Fourier transformation for each jump
        # also includes the multiplicity for the onsite terms (site expansion)
        # The latter is used to calculate escape rates
        self.FTjumps, self.SEjumps = self.FourierTransformJumps(jumpnetwork, self.N, self.kpts)
        # generate the Taylor expansion coefficients for each jump
        # generate the jumppairs
        self.jumppairs = tuple((self.invmap[jumplist[0][0][0]], self.invmap[jumplist[0][0][1]])
                               for jumplist in jumpnetwork)
        self.Taylorjumps = self.TaylorExpandJumps(jumpnetwork, self.N)

        self.D, self.eta = 0, 0

        Taylor = T3D()
        self.indlist_2 = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i + j + k > 2:
                        continue
                    self.indlist_2.append(Taylor.pow2ind[i, j, k])

    def BreakdownGroups(self):
        """
        indexing breakdown for each (i,j) pair.
        :return grouparray: array[NG][3][3] of the NG group operations
                            Only the cartesian rotation matrix
        :return indexpair: array[N][N][NG][2] of the index pair for each group operation
        """
        # Soham - change required in index mapping
        grouparray = np.zeros((self.NG, self.crys.dim, self.crys.dim))
        indexpair = np.zeros((self.N, self.N, self.NG, 2), dtype=int)
        for ng, g in enumerate(self.container.G):
            grouparray[ng, :, :] = g.cartrot[:, :]
            # first construct the indexmap of the group operations for dumbbells
            indexmap = g.indexmap[0]
            for i in range(self.N):
                for j in range(self.N):
                    indexpair[i, j, ng, 0], indexpair[i, j, ng, 1] = indexmap[i], indexmap[j]
        return grouparray, indexpair
