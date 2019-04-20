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

    # @staticmethod
    # def networkcount(jumpnetwork, N):
    #
    #     """
    #     Returns how many seperate networks of connected (via jumps) i,or states there are.
    #     Follows exactly the one for vacancies.
    #     Note - doesn't matter what c1, c2 are. If there is a jump (i, j, dx, c1, c2),
    #     then this means the two dumbbell states are connected.
    #     Return a count of how many separate connected networks there are
    #     that is, how many states are connected by jumps.
    #     say we have three states {0,1,2} in the iorlist.
    #     If 0 is connected to 1 and 1 to 2, then the connectivity is 1, because 0 and 2 are connected via 1
    #     even if not directly - see jupyter notebook in Practice Files folder."""
    #     jngraph = np.zeros((N, N), dtype=bool)
    #     for jlist in jumpnetwork:
    #         for (i, j, dx, c1, c2) in jlist:
    #             jngraph[i,j] = True
    #     connectivity = 0
    #     disconnected = {i for i in range(N)}#this is a set.
    #     while len(disconnected)>0:
    #         i = min(disconnected)
    #         cset = {i}
    #         disconnected.remove(i)
    #         while True:
    #             clen = len(cset)
    #             for n in cset.copy():
    #                 for m in disconnected.copy():
    #                     if jngraph[n,m]:
    #                         cset.add(m)
    #                         disconnected.remove(m)
    #             # check if we've stopped adding new members:
    #             if clen == len(cset): break
    #         connectivity += 1
    #
    #     return connectivity

    # def FourierTransformJumps(self, jumpnetwork, N, kpts):
    #     """
    #     Generate the Fourier transform coefficients for each jump - almost entirely same as vacancies
    #
    #     :param jumpnetwork: list of unique transitions, as lists of (i, j, dx, c1, c2)
    #                         i,j correspond to pair indices in iorlist
    #     :param N: number of sites
    #     :param kpts: array[Nkpt][3], in Cartesian (same coord. as dx) - the kpoints that are considered
    #     :return FTjumps: array[Njump][Nkpt][Nsite][Nsite] of FT of the jump network
    #             ->The Fourier transform of the repr. of each type of jump
    #             ->their values at the kpoint considered.
    #             ->their matrix elements corresponding to the initial and final sites.
    #             -> see equation(38) in the paper
    #     :return SEjumps: array[Nsite][Njump] multiplicity of jump on each site (?)
    #     """
    #     FTjumps = np.zeros((len(jumpnetwork), self.Nkpt, N, N), dtype=complex)
    #     SEjumps = np.zeros((N, len(jumpnetwork)), dtype=int)
    #     for J, jumplist in enumerate(jumpnetwork):
    #         for (i, j, dx, c1, c2) in jumplist:
    #             FTjumps[J, :, i, j] += np.exp(1.j * np.dot(kpts, dx)) #this is an array of exponentials
    #             SEjumps[i, J] += 1
    #             #How many jumps of each type come out of state j
    #             #in case of dumbbell -> point to the (i,or) index j
    #     return FTjumps, SEjumps

    # def TaylorExpandJumps(self, jumpnetwork, N):
    #     """
    #     Generate the Taylor expansion coefficients for each jump
    #
    #     :param jumpnetwork: list of unique transitions, as lists of (i, j, dx, c1, c2)
    #                         i,j correspond to pair indices in iorlist
    #     :param N: number of states
    #     :return T3Djumps: list of Taylor3D expansions of the jump network
    #     """
    #     Taylor = T3D
    #     Taylor()
    #     # Taylor expansion coefficient prefactors for exp(1j*x) : (1j)^n/n!
    #     pre = np.array([(1j) ** n / factorial(n, True) for n in range(Taylor.Lmax + 1)])
    #     #The prefactors for the jumps in the Taylor expansion of e^(-iq.dx)
    #     Taylorjumps = []
    #     for jumplist in jumpnetwork:
    #         # coefficients; we use tuples because we'll be successively adding to the coefficients in place
    #         c = [(n, n, np.zeros((Taylor.powlrange[n], N, N), dtype=complex)) for n in range(Taylor.Lmax + 1)]
    #         for (i, j, dx,c1,c2) in jumplist:
    #             pexp = Taylor.powexp(dx, normalize=False) #make the powerexpansion for the components of dx
    #             for n in range(Taylor.Lmax + 1):
    #                 (c[n][2])[:, i, j] += pre[n] * (Taylor.powercoeff[n] * pexp)[:Taylor.powlrange[n]]
    #                 #look at the symoblic python notebook for each of these
    #         Taylorjumps.append(Taylor(c))
    #     return Taylorjumps

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
