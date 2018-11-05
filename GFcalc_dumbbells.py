import numpy as np
from onsager import PowerExpansion as PE
import itertools
from copy import deepcopy
from numpy import linalg as LA
from scipy.special import hyp1f1, gamma, expi #, gammainc

class GFCrystalcalc(object):
    """
    Class calculator for the Green function, designed to work with the Crystal class.

    Many aspects similar to vacany GF calculator, indexing needs to change.
    """

    def __init__(self, crys, chem, iorlist, symorlist, jumpnetwork, Nmax=4, kptwt = None):
        """
        Initializes our calculator with the appropriate topology / connectivity. Doesn't
        require, at this point, the site probabilities or transition rates to be known.

        :param crys: Crystal object
        :param chem: index identifying the diffusing species
        :param iorlist: flat list of (basis_index,orientation) tuples -> analog of basis[chem] for dumbbells
        :param symorlist: (basis_index,orientation) pairs grouped into symmetrically unique lists
        :param jumpnetwork: list of unique transitions as lists of ((i,j), dx, c1, c2)
                            "Note here i and j are indices into iorlist"
        :param Nmax: maximum range as estimator for kpt mesh generation
        :param kptwt: (optional) tuple of (kpts, wts) to short-circuit kpt mesh generation
        """
        # this is really just used by loadHDF5() to circumvent __init__
        if all(x is None for x in (crys, chem, iorlist, symorlist, jumpnetwork)): return
        self.crys = crys
        self.chem = chem
        self.iorlist = iorlist
        self.symorlist = symorlist.copy()
        self.N = sum(len(w) for w in symorlist)#N - no. of dumbbell states
        self.Ndiff = self.networkcount(jumpnetwork, self.N)
        # self.invmap = np.zeros(self.N, dtype=int)
        # for ind, w in enumerate(symorlist):
        #     for i in range(len(w)):
        #         self.invmap[i] = ind
        # TODO: Figure out the inverse mapping into the flat list from the symmetric list
        # TODO: Cannot use tuples with np arrays as keys as they are unhashable
        self.NG = len(self.crys.G)  # number of group operations
        self.grouparray, self.indexpair = self.BreakdownGroups()
        # note: currently, we don't store jumpnetwork. If we want to rewrite the class
        # to allow a new kpoint mesh to be generated "on the fly", we'd need to store
        # a copy for regeneration
        # self.jumpnetwork = jumpnetwork
        # generate a kptmesh: now we try to make the mesh more "uniform" ??
        bmagn = np.array([np.sqrt(np.dot(crys.reciplatt[:, i], crys.reciplatt[:, i]))
                          for i in range(self.crys.dim)])
        bmagn /= np.power(np.product(bmagn), 1 / self.crys.dim)
        # make sure we have even meshes - same as vacancies
        # Don't need to change the k-mesh generation.
        self.kptgrid = np.array([2 * np.int(np.ceil(2 * Nmax * b)) for b in bmagn], dtype=int) \
            if kptwt is None else np.zeros(self.crys.dim, dtype=int)
        self.kpts, self.wts = crys.reducekptmesh(crys.fullkptmesh(self.kptgrid)) \
            if kptwt is None else deepcopy(kptwt)
        self.Nkpt = self.kpts.shape[0]
        # generate the Fourier transformation for each jump
        # also includes the multiplicity for the onsite terms (site expansion)
        self.FTjumps, self.SEjumps = self.FourierTransformJumps(jumpnetwork, self.N, self.kpts)
        # generate the Taylor expansion coefficients for each jump
        self.Taylorjumps = self.TaylorExpandJumps(jumpnetwork, self.N)
        # tuple of the Wyckoff site indices for each jump (needed to make symmrate)
        # self.jumppairs = tuple((self.invmap[jumplist[0][0][0]], self.invmap[jumplist[0][0][1]])
        #                        for jumplist in jumpnetwork)
        # TODO:jumppairs are used in SymmRates function - need to figure out inverse mapping
        self.D, self.eta = 0, 0  # we don't yet know the diffusivity

        def FourierTransformJumps(self, jumpnetwork, N, kpts):
        """
        Generate the Fourier transform coefficients for each jump - almost entirely same as vacancies

        :param jumpnetwork: list of unique transitions, as lists of ((i,j), dx, c1, c2)
                            i,j correspond to pair indices in iorlist
        :param N: number of sites
        :param kpts: array[Nkpt][3], in Cartesian (same coord. as dx) - the kpoints that are considered
        :return FTjumps: array[Njump][Nkpt][Nsite][Nsite] of FT of the jump network
                ->The Fourier transform of the repr. of each type of jump
                ->their values at the kpoint considered.
                ->their matrix elements corresponding to the initial and final sites.
                -> see equation(38) in the paper
        :return SEjumps: array[Nsite][Njump] multiplicity of jump on each site (?)
        """
        FTjumps = np.zeros((len(jumpnetwork), self.Nkpt, N, N), dtype=complex)
        SEjumps = np.zeros((N, len(jumpnetwork)), dtype=int)
        for J, jumplist in enumerate(jumpnetwork):
            for (i, j), dx, c1, c2 in jumplist:
                FTjumps[J, :, i, j] += np.exp(1.j * np.dot(kpts, dx)) #this is an array of exponentials
                SEjumps[i, J] += 1
                #How many jumps of each type come out of site j
                #in case of dumbbell -> point to the (i,or) index
        return FTjumps, SEjumps

    def TaylorExpandJumps(self, jumpnetwork, N):
        """
        Generate the Taylor expansion coefficients for each jump

        :param jumpnetwork: list of unique transitions, as lists of ((i,j), dx, c1, c2)
                            i,j correspond to pair indices in iorlist
        :param N: number of sites
        :return T3Djumps: list of Taylor3D expansions of the jump network
        """
        #Soham - any change required? -> see the Fourier equation in my case
        Taylor = T3D if self.crys.dim == 3 else T2D
        Taylor()  # need to do just to initialize the class; if already initialized, won't do anything
        # Taylor expansion coefficients for exp(1j*x) = (1j)^n/n!
        pre = np.array([(1j) ** n / factorial(n, True) for n in range(Taylor.Lmax + 1)])
        #The prefactors for the jumps in the Taylor expansion of e^(-iq.dx)
        Taylorjumps = []
        for jumplist in jumpnetwork:
            # coefficients; we use tuples because we'll be successively adding to the coefficients in place
            c = [(n, n, np.zeros((Taylor.powlrange[n], N, N), dtype=complex)) for n in range(Taylor.Lmax + 1)]
            for (i, j), dx in jumplist:
                pexp = Taylor.powexp(dx, normalize=False) #make the powerexpansion for the components of dx
                for n in range(Taylor.Lmax + 1):
                    (c[n][2])[:, i, j] += pre[n] * (Taylor.powercoeff[n] * pexp)[:Taylor.powlrange[n]]
                    #look at the symobilc python notebook for each of these
            Taylorjumps.append(Taylor(c))
        return Taylorjumps

        def BreakdownGroups(self):
        """
        Takes in a crystal, and a chemistry, and constructs the indexing breakdown for each
        (i,j) pair.
        :return grouparray: array[NG][3][3] of the NG group operations
        :return indexpair: array[N][N][NG][2] of the index pair for each group operation
        """
        #Soham - no changes required to this
        grouparray = np.zeros((self.NG, self.crys.dim, self.crys.dim))
        indexpair = np.zeros((self.N, self.N, self.NG, 2), dtype=int)
        for ng, g in enumerate(self.crys.G):
            grouparray[ng, :, :] = g.cartrot[:, :]
            indexmap = g.indexmap[self.chem]
            for i in range(self.N):
                for j in range(self.N):
                    indexpair[i, j, ng, 0], indexpair[i, j, ng, 1] = indexmap[i], indexmap[j]
        return grouparray, indexpair
