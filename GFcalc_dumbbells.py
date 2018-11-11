import numpy as np
from onsager import PowerExpansion as PE
from onsager.GFcalc import GFCrystalcalc
import itertools
from copy import deepcopy
from numpy import linalg as LA
from scipy.special import hyp1f1, gamma, expi #, gammainc

class GFcalc_dumbbells(GFCrystalcalc):
    """
    Class calculator for the Green function, designed to work with the Crystal class.

    Highly similar to vacany GF calculator, indexing of jumps is in (i,or) list rather than basis set (i).
    """

    def __init__(self, container, jumpnetwork, Nmax=4, kptwt = None):
        """
        Initializes our calculator with the appropriate topology / connectivity. Doesn't
        require, at this point, the site probabilities or transition rates to be known.

        :param crys: Crystal object
        :param chem: index identifying the diffusing species
        :param iorlist: flat list of (basis_index,orientation) tuples -> analog of basis[chem] for dumbbells
        :param symorlist: (basis_index,orientation) pairs grouped into symmetrically unique lists
        :param jumpnetwork: list of unique transitions as lists of ((i,j), dx, c1, c2)
                            "Note here i and j are indices into iorlist"
                            "Needs to be ensured that the jumpnetwork belongs to the container"
        :param Nmax: maximum range as estimator for kpt mesh generation
        :param kptwt: (optional) tuple of (kpts, wts) to short-circuit kpt mesh generation
        """
        # this is really just used by loadHDF5() to circumvent __init__
        if all(x is None for x in (container, jumpnetwork)): return
        if any (len(tup)!=5 for l in jumpnetwork for tup in l):
            raise TypeError("Enter the jumpnetwork of the form (i,j,dx,c1,c2)")
        self.crys = container.crys
        self.chem = container.chem
        self.iorlist = container.iorlist
        self.symorlist = container.symorlist.copy()
        self.N = len(iorlist)#N - no. of dumbbell states
        self.Ndiff = self.networkcount(jumpnetwork, self.N)
        #Create invmap - which symmety-grouped (i,or) pair list in symorlist
        # does a given (i,or) pair in iorlist belong to
        self.invmap = container.invmap #internalized in the container definition itself
        self.NG = len(self.crys.G)  # number of group operations
        self.grouparray, self.indexpair = self.BreakdownGroups()
        # note: currently, we don't store jumpnetwork. If we want to rewrite the class
        # to allow a new kpoint mesh to be generated "on the fly", we'd need to store
        # a copy for regeneration
        # self.jumpnetwork = jumpnetwork
        # generate a kptmesh: now we try to make the mesh more "uniform" ??
        bmagn = np.array([np.sqrt(np.dot(self.crys.reciplatt[:, i], self.crys.reciplatt[:, i]))
                          for i in range(3)])
        bmagn /= np.power(np.product(bmagn), 1 / 3)
        # make sure we have even meshes - same as vacancies
        # Don't need to change the k-mesh generation.
        self.kptgrid = np.array([2 * np.int(np.ceil(2 * Nmax * b)) for b in bmagn], dtype=int) \
            if kptwt is None else np.zeros(3, dtype=int)
        self.kpts, self.wts = crys.reducekptmesh(crys.fullkptmesh(self.kptgrid)) \
            if kptwt is None else deepcopy(kptwt)
        self.Nkpt = self.kpts.shape[0]
        # generate the Fourier transformation for each jump
        # also includes the multiplicity for the onsite terms (site expansion)
        self.FTjumps, self.SEjumps = self.FourierTransformJumps(jumpnetwork, self.N, self.kpts)
        # generate the Taylor expansion coefficients for each jump
        #generate the jumppairs
        self.jumppairs = tuple((self.invmap[jumplist[0][0]], self.invmap[jumplist[0][1]])
                               for jumplist in jumpnetwork)
        self.Taylorjumps = self.TaylorExpandJumps(jumpnetwork, self.N)

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
        Taylor = T3D
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
                    #look at the symoblic python notebook for each of these
            Taylorjumps.append(Taylor(c))
        return Taylorjumps

    def BreakdownGroups(self):
        """
        Same as that for the case of vacancies.
        Takes in a crystal, and a chemistry, and constructs the indexing breakdown for each
        (i,j) pair.
        :return grouparray: array[NG][3][3] of the NG group operations
        :return indexpair: array[N][N][NG][2] of the index pair for each group operation
        """
        #Soham - no changes required to this
        grouparray = np.zeros((self.NG, 3, 3))
        indexpair = np.zeros((self.N, self.N, self.NG, 2), dtype=int)
        for ng, g in enumerate(self.crys.G):
            grouparray[ng, :, :] = g.cartrot[:, :]
            indexmap = g.indexmap[self.chem]
            for i in range(self.N):
                for j in range(self.N):
                    indexpair[i, j, ng, 0], indexpair[i, j, ng, 1] = indexmap[i], indexmap[j]
        return grouparray, indexpair

    def SymmRates(self, pre, betaene, preT, betaeneT):
        """
        Constructs a set of symmetric jump rates for each type of unique jump in the jumplist
        Functions the same as the case for vacancies
        """
        #Note to self - refer to the Gfcalc module notes if you forget
        symmrates = np.array([pt*np.exp(0.5*betaene[sym0]+0.5*betaene[sym1]-beT)/np.sqrt(pre[sym0]*pre[sym1])
                                for (sym0,sym1),pt,beT in zip(self.jumppairs,preT,betaeneT)])
        return symmrates

    def SetRates(self, pre, betaene, preT, betaeneT, pmaxerror=1.e-8):
        self.symmrate = self.SymmRates(pre, betaene, preT, betaeneT)
        self.maxrate = self.symmrate.max()
        self.symmrate /= self.maxrate #make all rates relative to maxrate
