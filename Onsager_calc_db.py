import numpy as np
from numpy.core.multiarray import ndarray
import onsager.crystal as crystal
from onsager.crystalStars import zeroclean
from representations import *
from GFcalc_dumbbells import GF_dumbbells
import stars
import vector_stars
from functools import reduce
from scipy.linalg import pinv2
from onsager.OnsagerCalc import Interstitial, VacancyMediated
import itertools
import time


# Making stateprob, ratelist and symmratelist universal functions so that I can also use them later on in the case of
# #solutes.
def stateprob(pre, betaene, invmap):
    """Returns our (i,or) probabilities, normalized, as a vector.
       Straightforward extension from vacancy case.
    """
    # be careful to make sure that we don't under-/over-flow on beta*ene
    minbetaene = min(betaene)
    rho = np.array([pre[w] * np.exp(minbetaene - betaene[w]) for w in invmap])
    return rho / sum(rho)


# make a static method and reuse later for solute case?
def ratelist(jumpnetwork, pre, betaene, preT, betaeneT, invmap):
    """Returns a list of lists of rates, matched to jumpnetwork"""
    stateene = np.array([betaene[w] for w in invmap])
    statepre = np.array([pre[w] for w in invmap])
    return [[pT * np.exp(stateene[i] - beT) / statepre[i]
             for (i, j), dx in t]
            for t, pT, beT in zip(jumpnetwork, preT, betaeneT)]


def symmratelist(jumpnetwork, pre, betaene, preT, betaeneT, invmap):
    """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
    stateene = np.array([betaene[w] for w in invmap])
    statepre = np.array([pre[w] for w in invmap])
    return [[pT * np.exp(0.5 * stateene[i] + 0.5 * stateene[j] - beT) / np.sqrt(statepre[i] * statepre[j])
             for (i, j), dx in t]
            for t, pT, beT in zip(jumpnetwork, preT, betaeneT)]


class BareDumbbell(Interstitial):
    """
    class to compute Green's function for a bare interstitial dumbbell
    diffusing through as crystal.
    """

    def __init__(self, container, jumpnetwork, mixed=False):
        """
        param: container - container object for dumbbell states
        param: jumpnetwork - jumpnetwork (either omega_0 or omega_2)
        """
        self.container = container
        self.jumpnetwork = jumpnetwork
        self.N = sum([len(lst) for lst in container.symorlist])
        self.VB, self.VV = self.FullVectorBasis(mixed)
        self.NV = len(self.VB)

        # Need to check if this portion is still necessary
        self.omega_invertible = True
        if self.NV > 0:
            self.omega_invertible = any(np.allclose(g.cartrot, -np.eye(3)) for g in self.container.crys.G)

        # #What is this for though?
        # if self.omega_invertible:
        #     self.bias_solver = lambda omega,b : -la.solve(-omega,b,sym_pos=True)
        # else:
        #     # pseudoinverse required:
        #     self.bias_solver = lambda omega, b: np.dot(pinv2(omega), b)

        # self.sitegroupops = self.generateStateGroupOps()
        # self.jumpgroupops = self.generateJumpGroupOps()

    def FullVectorBasis(self, mixed):
        crys = self.container.crys
        chem = self.container.chem
        z = np.zeros(3, dtype=int)

        def make_Glist(tup):
            """
            Returns a set of gops that leave a state unchanged
            At the least there will be the identity
            """
            glist = set([])
            for g in self.container.crys.G:
                r1, (ch, i1) = crys.g_pos(g, z, (chem, tup[0]))
                onew = np.dot(g.cartrot, tup[1])
                onew -= onew.R
                if not mixed:
                    if np.allclose(onew + tup[1], z):
                        onew = -onew
                if tup[0] == i1 and np.allclose(r1, z) and np.allclose(onew, tup[1]):  # the state remains unchanged
                    glist.add(g - crys.g_pos(g, z, (chem, tup[0]))[0])
            return glist

        lis = []
        for statelistind, statelist in enumerate(self.container.symorlist):
            N = len(self.container.iorlist)
            glist = make_Glist(statelist[0])
            vbasis = reduce(crystal.CombineVectorBasis, [crystal.VectorBasis(*g.eigen()) for g in glist])
            for v in crys.vectlist(vbasis):
                v /= np.sqrt(len(statelist))
                vb = np.zeros((N, 3))
                for gind, g in enumerate(crys.G):
                    vb[self.container.indexmap[gind][self.container.indsymlist[statelistind][0]]] = self.g_direc(g, v)
                    # What if this changes the vector basis for the state itself?
                    # There are several groupos that leave a state unchanged.
                lis.append(vb)

        VV = np.zeros((3, 3, len(lis), len(lis)))
        for i, vb_i in enumerate(lis):
            for j, vb_j in enumerate(lis):
                VV[:, :, i, j] = np.dot(vb_i.T, vb_j)

        return np.array(lis), VV

    def generateStateGroupOps(self):
        """
        Returns a list of lists of groupOps that map the first element of each list in symorlist
        to the corresponding elements in the same list.
        """
        glist = []
        for lind, l in enumerate(self.container.symorlist):
            stind = self.container.indsymlist[lind][0]
            # tup0 = l[0]
            lis = []
            for ind, tup in enumerate(l):
                for gind, g in enumerate(self.container.crys.G):
                    if self.container.indexmap[g][ind] == stind:
                        lis.append(g)
            glist.append(lis)
        return glist

    def generateJumpGroupOps(self):
        """
        which group operations land the first jump of a jump list to the rest of the jumps in the same
        list.
        :return: glist - the list of the above-mentioned group operations
        """
        glist = []
        for jlist in self.jumpnetwork:
            tup = jlist[0]
            lis = []
            for j in jlist:
                for gind, g in self.container.crys.G:
                    if self.container.indexmap[gind][tup[0]] == j[0]:
                        if self.container.indexmap[gind][tup[1]] == j[1]:
                            if np.allclose(tup[2], self.container.crys.g_direc(g, j[2])):
                                lis.append(g)
            glist.append(lis)
        return glist

    # def stateprob(self, pre, betaene):
    #     """Returns our (i,or) probabilities, normalized, as a vector.
    #        Straightforward extension from vacancy case.
    #     """
    #     # be careful to make sure that we don't under-/over-flow on beta*ene
    #     minbetaene = min(betaene)
    #     rho = np.array([pre[w] * np.exp(minbetaene - betaene[w]) for w in self.container.invmap])
    #     return rho / sum(rho)
    #
    # #make a static method and reuse later for solute case?
    # def ratelist(self, pre, betaene, preT, betaeneT):
    #     """Returns a list of lists of rates, matched to jumpnetwork"""
    #     stateene = np.array([betaene[w] for w in self.container.invmap])
    #     statepre = np.array([pre[w] for w in self.container.invmap])
    #     return [[pT * np.exp(stateene[i] - beT) / statepre[i]
    #              for i, j, dx, c1, c2 in t]
    #             for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]
    #
    # def symmratelist(self, pre, betaene, preT, betaeneT):
    #     """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
    #     stateene = np.array([betaene[w] for w in self.container.invmap])
    #     statepre = np.array([pre[w] for w in self.container.invmap])
    #     return [[pT * np.exp(0.5 * stateene[i] + 0.5 * stateene[j] - beT) / np.sqrt(statepre[i] * statepre[j])
    #              for i, j, dx, c1, c2 in t]
    #             for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]

    def diffusivity(self, pre, betaene, preT, betaeneT):
        """
        Computes bare dumbbell diffusivity - works for mixed as well pure.
        """
        if len(pre) != len(self.container.symorlist):
            raise IndexError("length of prefactor {} doesn't match symorlist".format(pre))
        if len(betaene) != len(self.container.symorlist):
            raise IndexError("length of energies {} doesn't match symorlist".format(betaene))
        if len(preT) != len(self.jumpnetwork):
            raise IndexError("length of prefactor {} doesn't match jump network".format(preT))
        if len(betaeneT) != len(self.jumpnetwork):
            raise IndexError("length of energies {} doesn't match jump network".format(betaeneT))

        rho = stateprob(pre, betaene, self.container.invmap)
        sqrtrho = np.sqrt(rho)
        rates_lst = ratelist(self.jumpnetwork, pre, betaene, preT, betaeneT, self.container.invmap)
        symmrates_lst = symmratelist(self.jumpnetwork, pre, betaene, preT, betaeneT, self.container.invmap)
        omega_ij = np.zeros((self.N, self.N))
        domega_ij = np.zeros((self.N, self.N))
        bias_i = np.zeros((self.N, 3))
        dbias_i = np.zeros((self.N, 3))
        D0 = np.zeros((3, 3))
        # Dcorrection = np.zeros((3, 3))
        Db = np.zeros((3, 3))
        stateene = np.array([betaene[w] for w in self.container.invmap])
        # Boltmann averaged energies of all states
        Eave = np.dot(rho, stateene)

        for jlist, rates, symmrates, bET in zip(self.jumpnetwork, rates_lst, symmrates_lst, betaeneT):
            for ((i, j), dx), rate, symmrate in zip(jlist, rates, symmrates):
                omega_ij[i, j] += symmrate
                omega_ij[i, i] -= rate
                domega_ij[i, j] += symmrate * (bET - 0.5 * (stateene[i] + stateene[j]))
                bias_i[i] += sqrtrho[i] * rate * dx
                # dbias_i[i] += sqrtrho[i] * rate * dx * (bET - 0.5 * (stateene[i] + Eave))
                # for domega and dbias - read up section 2.2 in the paper.
                # These are to evaluate the derivative of D wrt to beta. Read later.
                D0 += 0.5 * np.outer(dx, dx) * rho[i] * rate
                Db += 0.5 * np.outer(dx, dx) * rho[i] * rate * (bET - Eave)
                # Db - derivative with respect to beta
        # gamma_i = np.zeros((self.N, 3))
        gamma_i = np.tensordot(pinv2(omega_ij), bias_i, axes=(1, 0))

        Dcorr = np.zeros((3, 3))
        for i in range(self.N):
            Dcorr += np.outer(bias_i[i], gamma_i[i])

        return D0 + Dcorr


class dumbbellMediated(VacancyMediated):
    """
    class to compute dumbbell mediated solute transport coefficients. We inherit the calculator
    for vacancies from Prof. Trinkle's code for vacancies with changes as and when required.

    Here, unlike vacancies, we must compute the Green's Function by Block inversion
    and Taylor expansion (as in the GFCalc module) for both bare pure (g0)
    and mixed(g2) dumbbells, since our Dyson equation requires so.
    Also, instead of working with crystal and chem, we work with the container objects.
    """

    def __init__(self, pdbcontainer, mdbcontainer, jnet0data, jnet2data, cutoff, solt_solv_cut, solv_solv_cut,
                 closestdistance, NGFmax=4, Nthermo=0):
        """

        :param pdbcontainer: The container object for pure dumbbells - instance of dbStates
        :param mdbcontainer: The container object for mixed dumbbell - instance of mStates

        :param jnet0data - (jnet0, jnet0_indexed) - the jumpnetworks for pure dumbbells
            jnet0 - jumps are of the form (state1, state2, c1 ,c2) - must be produced from states in pdbcontainer.
            jnet0_indexed - jumps are of the form ((i, j),d x) - indices must be matched to states in pdbcontainer.

        :param jnet2data - (jnet2, jnet2_indexed) - the jumpnetworks for mixed dumbbells
            jnet2 - jumps are of the form (state1, state2, c1 ,c2) - must be produced from states in mdbcontainer.
            jnet2_indexed - jumps are of the form ((i, j), dx) - indices must be matched to states in mdbcontainer.

        :param cutoff: The maximum jump distance to be considered while building the jump networks
        :param solt_solv_cut: The collision cutoff between solute and solvent atoms
        :param solv_solv_cut: The collision cutoff between solvent and solvent atoms
        :param closestdistance: The closest distance allowable to all other atoms in the crystal.
        :param NGFmax: Parameter controlling k-point density (cf - GFcalc.py from the vacancy version)
        :param Nthermo: The number of jump-nearest neighbor sites that are to be considered within the thermodynamic
        shell.
        """
        # All the required quantities will be extracted from the containers as we move along
        self.pdbcontainer = pdbcontainer
        self.mdbcontainer = mdbcontainer
        (self.jnet0, self.jnet0_indexed), (self.jnet2, self.jnet2_indexed) = jnet0data, jnet2data
        self.crys = pdbcontainer.crys  # we assume this is the same in both containers
        self.chem = pdbcontainer.chem

        # Create the solute invmap
        sitelist_solute = self.crys.sitelist(self.chem)
        self.invmap_solute = np.zeros(len(self.crys.basis[self.chem]),dtype=int)
        for wyckind,ls in enumerate(sitelist_solute):
            for site in ls:
                self.invmap_solute[site] = wyckind

        # self.jnet2_indexed = self.kinetic.starset.jnet2_indexed
        self.thermo = stars.StarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
                                    (self.jnet2, self.jnet2_indexed))
        self.kinetic = stars.StarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
                                     (self.jnet2, self.jnet2_indexed))

        # Note - even if empty, our starsets go out to atleast the NNstar - later we'll have to keep this in mind
        self.NNstar = stars.StarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
                                    (self.jnet2, self.jnet2_indexed), 2)
        self.vkinetic = vector_stars.vectorStars()

        # Make GF calculators.
        self.GFcalc_pure = GF_dumbbells(self.pdbcontainer, self.jnet0_indexed, Nmax=4, kptwt=None)
        self.GFcalc_mixed = GF_dumbbells(self.mdbcontainer, self.jnet2_indexed, Nmax=4, kptwt=None)

        # Generate the initialized crystal and vector stars and the jumpnetworks with the kinetic shell
        self.generate(Nthermo, cutoff, solt_solv_cut, solv_solv_cut, closestdistance)


    def generate_jnets(self, cutoff, solt_solv_cut, solv_solv_cut, closestdistance):
        """
        Note - for mixed dumbbells, indexing to the iorlist is the same as indexing to mixedstates, as the latter is
        just the former in the form of SdPair objects, all of which are origin states.
        """
        # first omega0 and omega2 - indexed to complexStates and mixed states
        # self.jnet2_indexed = self.vkinetic.starset.jnet2_indexed
        # self.omeg2types = self.vkinetic.starset.jnet2_types
        self.jtags2 = self.vkinetic.starset.jtags2
        # Next - omega1 - indexed to complexStates
        (self.jnet_1, self.jnet1_indexed, self.jtags1), self.om1types = self.vkinetic.starset.jumpnetwork_omega1()

        # next, omega3 and omega_4, indexed to pure and mixed states
        (self.symjumplist_omega43_all, self.symjumplist_omega43_all_indexed), (
            self.symjumplist_omega4, self.symjumplist_omega4_indexed, self.jtags4), \
        (self.symjumplist_omega3, self.symjumplist_omega3_indexed,
         self.jtags3) = self.vkinetic.starset.jumpnetwork_omega34(cutoff, solv_solv_cut, solt_solv_cut, closestdistance)

    def generate(self, Nthermo, cutoff, solt_solv_cut, solv_solv_cut, closestdistance):

        if Nthermo == getattr(self, "Nthermo", 0): return
        self.Nthermo = Nthermo
        self.thermo.generate(Nthermo)
        self.kinetic.generate(Nthermo + 1)
        # self.Nmixedstates = len(self.kinetic.mixedstates)
        # self.NcomplexStates = len(self.kinetic.complexStates)
        self.vkinetic.generate(self.kinetic)  # we generate the vector star out of the kinetic shell
        # Now generate the pure and mixed dumbbell Green functions expnsions - internalized within vkinetic.

        # Generate and indexing that takes from a star in the thermodynamic shell
        # to the corresponding star in the kinetic shell.
        self.thermo2kin = np.zeros(self.thermo.mixedstartindex, dtype=int)
        for th_ind, thstar in enumerate(self.thermo.stars[:self.thermo.mixedstartindex]):
            count = 0
            for k_ind, kstar in enumerate(self.vkinetic.starset.stars[:self.vkinetic.starset.mixedstartindex]):
                # check if the representative state of the thermo star is present in the kin star.
                if thstar[0] in set(kstar):
                    count += 1
                    self.thermo2kin[th_ind] = k_ind
            if count != 1:
                raise TypeError("thermodynamic and kinetic shells not consistent.")


        self.generate_jnets(cutoff, solt_solv_cut, solv_solv_cut, closestdistance)

        # Generate the GF expansions
        (self.GFstarset_pure, self.GFPureStarInd, self.GFexpansion_pure), \
        (self.GFstarset_mixed, self.GFMixedStarInd, self.GFexpansion_mixed) \
            = self.vkinetic.GFexpansion()

        # Generate the bias expansions
        self.biases = self.vkinetic.biasexpansion(self.jnet_1, self.jnet2, self.om1types, self.symjumplist_omega43_all)

        # generate the rate expansions
        self.rateExps = self.vkinetic.rateexpansion(self.jnet_1, self.om1types, self.symjumplist_omega43_all)

        #generate the outer products of the vector stars
        self.kinouter =  self.vkinetic.outer()
        # self.clearcache()

    def calc_eta(self, rate0list, rate2list):
        """
        Function to calculate the periodic eta vectors.
        rate0list, rate2list - the SYMMETRIZED rate lists for the bare and mixed dumbbell spaces.
        There is a slight misnomer - what we refer to as the eta vector here, is actually the gamma vector in the
        2017 phil. mag. paper (because we are using symmetrized rates).
        """

        # The non-local bias for the complex space has to be carried out based on the omega0 jumpnetwork,
        # not the omega1 jumpnetwork.This is because all the jumps that are allowed by omega0 out of a given dumbbell
        # state are not there in omega1. That is because omega1 considers only those states that are in the kinetic
        # shell. Not outside it.

        # First, we build up g0 and g2 - g2 will be required in makeGF as well. So, we make it an object attribute.
        omega0_nonloc = np.zeros((len(self.vkinetic.starset.bareStates), len(self.vkinetic.starset.bareStates)))
        # use the indexed omega2 to fill this up - need omega2 indexed to mixed subspace of starset
        for rate0, jlist in zip(rate0list, self.jnet0_indexed):
            for (i, j), dx in jlist:
                omega0_nonloc[i, j] += rate0[0]
                omega0_nonloc[i, i] -= rate0[0]

        self.g0 = pinv2(omega0_nonloc)

        omega2_nonloc = np.zeros((len(self.vkinetic.starset.mixedstates), len(self.vkinetic.starset.mixedstates)))

        for rate2, jlist in zip(rate2list, self.jnet2_indexed):
            for (i, j), dx in jlist:
                omega2_nonloc[i, j] += rate2[0]
                omega2_nonloc[i, i] -= rate2[0]

        self.g2 = pinv2(omega2_nonloc)

        self.biasBareExpansion = self.biases[-1]

        # get the biasBare and bias2 expansions. First check if non-local biases should be zero anyway (as is the case
        # with highly symmetric lattices - in that case vecpos_bare should be zero)
        if len(self.vkinetic.vecpos_bare) == 0:
            self.eta00_solvent = np.zeros((len(self.vkinetic.starset.complexStates), 3))
            self.eta00_solute = np.zeros((len(self.vkinetic.starset.complexStates), 3))
        # otherwise, we need to build the bare bias expansion
        else:
            # First we build up for just the bare starset

            # We first get the bias vector in the basis of the vector stars.
            # Since we are using symmetrized rates, we only need to consider them
            self.NlsolventBias_bare = np.zeros((len(self.vkinetic.starset.bareStates), 3))
            bias0SolventTotNonLoc = np.dot(self.biasBareExpansion,
                                           np.array([rate0list[i][0] for i in range(len(self.jnet0))]))

            # Then, we convert them to cartesian form for each state.
            for st in self.vkinetic.starset.bareStates:
                indlist = self.vkinetic.stateToVecStar_bare[st]
                if len(indlist) != 0:
                    self.NlsolventBias_bare[self.vkinetic.starset.bareindexdict[st][0]][:] = \
                        sum([bias0SolventTotNonLoc[tup[0]] * self.vkinetic.vecvec_bare[tup[0]][tup[1]] for tup in
                             indlist])

            # Then, we use g0 to get the eta0 vectors. The second 0 in eta00 indicates omega0 space.
            self.eta00_solvent_bare = np.tensordot(self.g0, self.NlsolventBias_bare, axes=(1, 0))
            self.eta00_solute_bare = np.zeros_like(self.eta00_solvent_bare)

            # Now match the non-local biases for complex states to the pure states
            self.eta00_solute = np.zeros((len(self.vkinetic.starset.complexStates), 3))
            self.eta00_solvent = np.zeros((len(self.vkinetic.starset.complexStates), 3))
            self.NlsolventBias0 = np.zeros((len(self.vkinetic.starset.complexStates), 3))

            for i in range(len(self.vkinetic.starset.complexStates)):
                db = self.vkinetic.starset.complexStates[i].db
                db = db - db.R
                for j in range(len(self.vkinetic.starset.bareStates)):
                    if db == self.vkinetic.starset.bareStates[j]:
                        self.eta00_solvent[i, :] = self.eta00_solvent_bare[j, :].copy()
                        self.NlsolventBias0[i, :] = self.NlsolventBias_bare[j, :].copy()
                        break
                        # No need to update anything for the solute in the complex space.
        # Now for omega2.
        # First, get the bias vectors in the basis of the vector stars.
        self.NlsoluteBias2 = np.zeros((len(self.vkinetic.starset.mixedstates), 3))
        self.NlsolventBias2 = np.zeros((len(self.vkinetic.starset.mixedstates), 3))
        bias2solute, bias2solvent = self.biases[2]
        bias2SoluteTotNonLoc = np.dot(bias2solute, np.array([rate2list[i][0] for i in range(len(self.jnet2_indexed))]))
        bias2SolventTotNonLoc = np.dot(bias2solvent,
                                       np.array([rate2list[i][0] for i in range(len(self.jnet2_indexed))]))

        # Now go state by state and get the cartesian form of the bias vectors for each species
        for st in self.vkinetic.starset.mixedstates:
            indlist = self.vkinetic.stateToVecStar_mixed[st]
            if len(indlist) != 0:
                # For the solute
                self.NlsoluteBias2[self.vkinetic.starset.mixedindexdict[st][0]][:] = \
                    sum([bias2SoluteTotNonLoc[tup[0] - self.vkinetic.Nvstars_pure] * \
                         self.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
                # For the solvent
                self.NlsolventBias2[self.vkinetic.starset.mixedindexdict[st][0]][:] = \
                    sum([bias2SolventTotNonLoc[tup[0] - self.vkinetic.Nvstars_pure] * \
                         self.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        # dot g2 with the cartesian bias vectors to get the eta0 for each state
        self.eta02_solvent = np.tensordot(self.g2, self.NlsolventBias2, axes=(1, 0))
        self.eta02_solute = np.tensordot(self.g2, self.NlsoluteBias2, axes=(1, 0))

        # So what do we have up until now?
        # We have constructed the Nstates x 3 eta0 vectors for pure and mixed states separately
        # But the jtags assume all the eta vectors are in the same list.
        # So, we need to concatenate the mixed eta vectors into the pure eta vectors.

        self.eta0total_solute = np.zeros(
            (len(self.vkinetic.starset.complexStates) + len(self.vkinetic.starset.mixedstates), 3))
        # noinspection PyAttributeOutsideInit
        self.eta0total_solvent = np.zeros(
            (len(self.vkinetic.starset.complexStates) + len(self.vkinetic.starset.mixedstates), 3))

        self.eta0total_solute[:len(self.vkinetic.starset.complexStates), :] = self.eta00_solute.copy()
        self.eta0total_solute[len(self.vkinetic.starset.complexStates):, :] = self.eta02_solute.copy()

        self.eta0total_solvent[:len(self.vkinetic.starset.complexStates), :] = self.eta00_solvent.copy()
        self.eta0total_solvent[len(self.vkinetic.starset.complexStates):, :] = self.eta02_solvent.copy()

    def bias_changes(self):
        """
        Function that allows us to construct new bias and bare expansions based on the eta vectors already calculated.

        We don't want to repeat the construction of the jumpnetwork based on the recalculated displacements after
        subtraction of the eta vectors (as in the variational principle).

        The steps are illustrated in the GM slides of Feb 25, 2019 - will include in the detailed documentation later on
        """
        # create updated bias expansions
        # Construct the projection of eta vectors
        self.delbias1expansion_solute = np.zeros_like(self.biases[1][0])
        self.delbias1expansion_solvent = np.zeros_like(self.biases[1][0])

        self.delbias4expansion_solute = np.zeros_like(self.biases[4][0])
        self.delbias4expansion_solvent = np.zeros_like(self.biases[4][0])
        for i in range(self.vkinetic.Nvstars_pure):
            # get the representative state(its index in complexStates) and vector
            v0 = self.vkinetic.vecvec[i][0]
            st0 = self.vkinetic.starset.complexIndexdict[self.vkinetic.vecpos[i][0]][
                0]  # Index of the state in the flat list
            # Form the projection of the eta vectors on v0
            eta_proj_solute = np.dot(self.eta0total_solute, v0)
            eta_proj_solvent = np.dot(self.eta0total_solvent, v0)
            # Now go through the omega1 jump network tags
            for jt, initindexdict in enumerate(self.jtags1):
                # see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                self.delbias1expansion_solute[i, jt] += len(self.vkinetic.vecpos[i]) * np.sum(
                    np.dot(initindexdict[st0], eta_proj_solute))
                self.delbias1expansion_solvent[i, jt] += len(self.vkinetic.vecpos[i]) * np.sum(
                    np.dot(initindexdict[st0], eta_proj_solvent))
            # Now let's build it for omega4
            for jt, initindexdict in enumerate(self.jtags4):
                # see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                self.delbias4expansion_solute[i, jt] += len(self.vkinetic.vecpos[i]) * np.sum(
                    np.dot(initindexdict[st0], eta_proj_solute))
                self.delbias4expansion_solvent[i, jt] += len(self.vkinetic.vecpos[i]) * np.sum(
                    np.dot(initindexdict[st0], eta_proj_solvent))

        self.delbias3expansion_solute = np.zeros_like(self.biases[3][0])
        self.delbias3expansion_solvent = np.zeros_like(self.biases[3][0])

        self.delbias2expansion_solute = np.zeros_like(self.biases[2][0])
        self.delbias2expansion_solvent = np.zeros_like(self.biases[2][0])

        for i in range(self.vkinetic.Nvstars - self.vkinetic.Nvstars_pure):
            # get the representative state(its index in mixedstates) and vector
            v0 = self.vkinetic.vecvec[i + self.vkinetic.Nvstars_pure][0]
            st0 = self.vkinetic.starset.mixedindexdict[self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure][0]][0]
            # Form the projection of the eta vectors on v0
            eta_proj_solute = np.dot(self.eta0total_solute, v0)
            eta_proj_solvent = np.dot(self.eta0total_solvent, v0)
            # Now go through the omega1 jump network tags
            for jt, initindexdict in enumerate(self.jtags2):
                # see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                self.delbias2expansion_solute[i, jt] += len(
                    self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * np.sum(
                    np.dot(initindexdict[st0], eta_proj_solute))
                self.delbias2expansion_solvent[i, jt] += len(
                    self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * np.sum(
                    np.dot(initindexdict[st0], eta_proj_solvent))
            # Now let's build it for omega4
            for jt, initindexdict in enumerate(self.jtags3):
                # see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                self.delbias3expansion_solute[i, jt] += len(
                    self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * np.sum(
                    np.dot(initindexdict[st0], eta_proj_solute))
                self.delbias3expansion_solvent[i, jt] += len(
                    self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * np.sum(
                    np.dot(initindexdict[st0], eta_proj_solvent))

    def update_bias_expansions(self, rate0list, rate2list):
        self.calc_eta(rate0list, rate2list)
        self.bias_changes()
        self.bias1_solute_new = zeroclean(self.biases[1][0] + self.delbias1expansion_solute)
        self.bias1_solvent_new = zeroclean(self.biases[1][1] + self.delbias1expansion_solvent)

        self.bias3_solute_new = zeroclean(self.biases[3][0] + self.delbias3expansion_solute)
        self.bias3_solvent_new = zeroclean(self.biases[3][1] + self.delbias3expansion_solvent)

        self.bias4_solute_new = zeroclean(self.biases[4][0] + self.delbias4expansion_solute)
        self.bias4_solvent_new = zeroclean(self.biases[4][1] + self.delbias4expansion_solvent)

        self.bias2_solute_new = zeroclean(self.biases[2][0] + self.delbias2expansion_solute)
        self.bias2_solvent_new = zeroclean(self.biases[2][1] + self.delbias2expansion_solvent)

    def bareExpansion(self, eta0_solute, eta0_solvent):
        """
        Returns the contributions to the terms of the uncorrelated diffusivity term,
        grouped separately for each type of jump. Intended to be called after displacements have been applied to the displacements.

        Params:
            jumpnetwork_omega* - indexed versions of the jumpnetworks with displacements for a given species. - jumps need to be of the form ((i,j),dx_species)
            jumptype - list that contains the omega_0 jump a given omega_1 jump list corresponds to. - these are the rates to be used to dot into b_0.

        In mixed dumbbell space, both solute and solvent will have uncorrelated contributions.
        The mixed dumbbell space is completely non-local.
        """
        # a = solute, b = solvent
        # eta0_solute, eta0_solvent = self.eta0total_solute, self.eta0total_solvent
        # Stores biases out of complex states, followed by mixed dumbbell states.
        jumpnetwork_omega1, jumptype, jumpnetwork_omega2, jumpnetwork_omega3, jumpnetwork_omega4 =\
            self.jnet1_indexed, self.om1types, self.jnet2_indexed, self.symjumplist_omega3_indexed,\
            self.symjumplist_omega4_indexed

        Ncomp = len(self.vkinetic.starset.complexStates)

        # D0expansion_aa = np.zeros((3, 3, len(self.jnet0)))
        # D0expansion_bb = np.zeros((3, 3, len(self.jnet0)))
        # D0expansion_ab = np.zeros((3, 3, len(sel

        # Since omega1 contains the total rate and not just the change, we don't need a separate D0 expansion.
        D1expansion_aa = np.zeros((3, 3, len(jumpnetwork_omega1)))
        D1expansion_bb = np.zeros((3, 3, len(jumpnetwork_omega1)))
        D1expansion_ab = np.zeros((3, 3, len(jumpnetwork_omega1)))

        D2expansion_aa = np.zeros((3, 3, len(jumpnetwork_omega2)))
        D2expansion_bb = np.zeros((3, 3, len(jumpnetwork_omega2)))
        D2expansion_ab = np.zeros((3, 3, len(jumpnetwork_omega2)))

        D3expansion_aa = np.zeros((3, 3, len(jumpnetwork_omega3)))
        D3expansion_bb = np.zeros((3, 3, len(jumpnetwork_omega3)))
        D3expansion_ab = np.zeros((3, 3, len(jumpnetwork_omega3)))

        D4expansion_aa = np.zeros((3, 3, len(jumpnetwork_omega4)))
        D4expansion_bb = np.zeros((3, 3, len(jumpnetwork_omega4)))
        D4expansion_ab = np.zeros((3, 3, len(jumpnetwork_omega4)))

        iorlist_pure = self.pdbcontainer.iorlist
        iorlist_mixed = self.mdbcontainer.iorlist
        # Need versions for solute and solvent - solute dusplacements are zero anyway
        for k, jt, jumplist in zip(itertools.count(), jumptype, jumpnetwork_omega1):
            d0 = np.sum(
                0.5 * np.outer(dx + eta0_solvent[i] - eta0_solvent[j], dx + eta0_solvent[i] - eta0_solvent[j]) for
                (i, j), dx in jumplist)
            # D0expansion_bb[:, :, jt] += d0
            D1expansion_bb[:, :, k] += d0
            # For solutes, don't need to do anything for omega1 and omega0 - solute does not move anyway
            # and therefore, their non-local eta corrections are also zero.

        for jt, jumplist in enumerate(jumpnetwork_omega2):
            # Build the expansions directly
            for (IS, FS), dx in jumplist:
                o1 = iorlist_mixed[self.vkinetic.starset.mixedstates[IS].db.iorind][1]
                o2 = iorlist_mixed[self.vkinetic.starset.mixedstates[FS].db.iorind][1]
                dx_solute = dx + o2/2. - o1/2. + eta0_solute[Ncomp + IS] - eta0_solute[Ncomp + FS]
                dx_solvent = dx - o2/2. + o1/2. + eta0_solvent[Ncomp + IS] - eta0_solvent[Ncomp + FS]
                D2expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D2expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                D2expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        for jt, jumplist in enumerate(jumpnetwork_omega3):
            for (IS, FS), dx in jumplist:
                o1 = iorlist_mixed[self.vkinetic.starset.mixedstates[IS].db.iorind][1]
                dx_solute = -o1/2. + eta0_solute[Ncomp + IS] - eta0_solute[FS]
                dx_solvent = dx + o1/ 2. + eta0_solvent[Ncomp + IS] - eta0_solvent[FS]
                D3expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D3expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                D3expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        for jt, jumplist in enumerate(jumpnetwork_omega4):
            for (IS, FS), dx in jumplist:
                o2 = iorlist_mixed[self.vkinetic.starset.mixedstates[FS].db.iorind][1]
                dx_solute = o2 / 2. + eta0_solute[IS] - eta0_solute[Ncomp + FS]
                dx_solvent = dx - o2 / 2. + eta0_solvent[IS] - eta0_solvent[Ncomp + FS]
                D4expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D4expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                D4expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        return (zeroclean(D1expansion_aa), zeroclean(D1expansion_bb), zeroclean(D1expansion_ab)),\
               (zeroclean(D2expansion_aa), zeroclean(D2expansion_bb), zeroclean(D2expansion_ab)),\
               (zeroclean(D3expansion_aa), zeroclean(D3expansion_bb), zeroclean(D3expansion_ab)),\
               (zeroclean(D4expansion_aa), zeroclean(D4expansion_bb), zeroclean(D4expansion_ab))

    # noinspection SpellCheckingInspection
    @staticmethod
    def preene2betafree(kT, predb0, enedb0, preS, eneS, preSdb, eneSdb, predb2, enedb2, preT0, eneT0, preT2, eneT2,
                        preT1, eneT1, preT43, eneT43):
        """
        Similar to the function for vacancy mediated OnsagerCalc. Takes in the energies and entropic pre-factors for
        the states and transition states and returns the corresponding free energies. The difference from the vacancy case
        is the consideration of more types of states ans transition states.

        Parameters:
            pre* - entropic pre-factors
            ene* - state/transition state energies.
        The pre-factors for pure dumbbells are matched to the symmorlist. For mixed dumbbells the mixedstarset and
        symmorlist are equivalent and the pre-factors are energies are matched to these.
        For solute-dumbbell complexes, the pre-factors and the energies are matched to the star set.

        Note - for the solute-dumbbell complexes, eneSdb and preSdb are the binding (excess) energies and pre
        factors respectively. We need to evaluate the total configuration energy separately.

        For all the transitions, the pre-factors and energies for transition states are matched to symmetry-unique jump types.

        Returns :
        bFdb0, bFdb2, bFS, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4
        the free energies for the states and transition states. Used in L_ij() and getsymmrates() to get the
        symmetrized transition rates.


        """
        beta = 1/kT
        bFdb0 = beta * enedb0 - np.log(predb0)
        bFdb2 = beta * enedb2 - np.log(predb2)
        bFS = beta * eneS - np.log(preS)
        bFSdb = beta * eneSdb - np.log(preSdb)

        bFT0 = beta * eneT0 - np.log(preT0)
        bFT1 = beta * eneT1 - np.log(preT1)
        bFT2 = beta * eneT2 - np.log(preT2)
        bFT3 = beta * eneT43 - np.log(preT43)
        bFT4 = beta * eneT43 - np.log(preT43)

        # Now, shift
        bFdb0_min = np.min(bFdb0)
        bFdb2_min = np.min(bFdb2)
        bFS_min = np.min(bFS)

        # bFdb0 -= bFdb0_min
        # bFdb2 -= bFdb2_min
        # bFS -= bFS_min
        # The unshifted values are required to be able to normalize the state probabilities.
        # See the L_ij function for details
        bFT0 -= bFdb0_min
        bFT2 -= bFdb2_min
        bFT3 -= bFdb2_min
        bFT1 -= (bFS_min + bFdb0_min)
        bFT4 -= (bFS_min + bFdb0_min)

        return bFdb0, bFdb2, bFS, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4

    def getsymmrates(self, bFdb0, bFdb2, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4):
        """
        :param bFdb0: beta * ene_db0 - ln(pre_db0) - relative to bFdb0min
        :param bFdb2: beta * ene_db2 - ln(pre_db2) - relative to bFdb2min
        :param bFSdb: beta * ene_Sdb - ln(pre_Sdb) - Total (not excess) - Relative to bFdb0min + bFSmin
        :param bFT0: beta * ene_T0 - ln(pre_T0) - relative to bFdb0min
        :param bFT1: beta * ene_T1 - ln(pre_T1) - relative to bFdb0min + bFSmin
        :param bFT2: beta * ene_T2 - ln(pre_T2) - relative to bFdb2min
        :param bFT3: beta * ene_T3 - ln(pre_T3) - relative to bFdb2min
        :param bFT4: beta * ene_T4 - ln(pre_T4) - relative to bFdb0min + bFSmin
        :return:
        """
        Nvstars_mixed = self.vkinetic.Nvstars - self.vkinetic.Nvstars_pure

        omega0 = np.zeros(len(self.jnet0))
        omega0escape = np.zeros((len(self.pdbcontainer.symorlist), len(self.jnet0)))

        omega2 = np.zeros(len(self.jnet2))
        omega2escape = np.zeros((len(self.mdbcontainer.symorlist), len(self.jnet2)))

        omega1 = np.zeros(len(self.jnet_1))
        omega1escape = np.zeros((self.vkinetic.Nvstars_pure, len(self.jnet_1)))

        omega3 = np.zeros(len(self.symjumplist_omega3))
        omega3escape = np.zeros((Nvstars_mixed, len(self.symjumplist_omega3)))

        omega4 = np.zeros(len(self.symjumplist_omega4))
        omega4escape = np.zeros((self.vkinetic.Nvstars_pure, len(self.symjumplist_omega4)))

        # build the omega0 lists
        for jt, jlist in enumerate(self.jnet0):

            # Get the bare dumbbells between which jumps are occurring
            st1 = jlist[0].state1 - jlist[0].state1.R
            st2 = jlist[0].state2 - jlist[0].state2.R

            # get the symorindex of the states - these serve analogous to Wyckoff sets
            w1 = self.vkinetic.starset.pdbcontainer.invmap[self.vkinetic.starset.pdbcontainer.db2ind(st1)]
            w2 = self.vkinetic.starset.pdbcontainer.invmap[self.vkinetic.starset.pdbcontainer.db2ind(st2)]

            omega0escape[w1, jt] = np.exp(-bFT0[jt] + bFdb0[w1])
            omega0escape[w2, jt] = np.exp(-bFT0[jt] + bFdb0[w2])
            omega0[jt] = np.sqrt(omega0escape[w1, jt] * omega0escape[w2, jt])

        # we need omega2 only for the uncorrelated contributions.
        for jt, jlist in enumerate(self.jnet2):
            st1 = jlist[0].state1.db
            st2 = jlist[0].state2.db

            w1 = self.vkinetic.starset.mdbcontainer.invmap[self.vkinetic.starset.mdbcontainer.db2ind(st1)]
            w2 = self.vkinetic.starset.mdbcontainer.invmap[self.vkinetic.starset.mdbcontainer.db2ind(st2)]

            omega2escape[w1, jt] = np.exp(-bFT2[jt] + bFdb2[w1])
            omega2escape[w2, jt] = np.exp(-bFT2[jt] + bFdb2[w2])
            omega2[jt] = np.sqrt(omega2escape[w1, jt] * omega2escape[w2, jt])

        # build the omega1 lists
        for jt, jlist in enumerate(self.jnet_1):

            st1 = jlist[0].state1
            st2 = jlist[0].state2

            if st1.is_zero(self.vkinetic.starset.pdbcontainer) or st2.is_zero(self.vkinetic.starset.pdbcontainer):
                continue

            # get the crystal stars of the representative jumps
            crStar1 = self.vkinetic.starset.complexIndexdict[st1][1]
            crStar2 = self.vkinetic.starset.complexIndexdict[st2][1]

            init2TS = np.exp(-bFT1[jt] + bFSdb[crStar1])
            fin2TS = np.exp(-bFT1[jt] + bFSdb[crStar2])

            omega1[jt] = np.sqrt(init2TS * fin2TS)

            # Get the vector stars where they are located
            v1list = self.vkinetic.stateToVecStar_pure[st1]
            v2list = self.vkinetic.stateToVecStar_pure[st2]

            for (v1, in_v1) in v1list:
                omega1escape[v1, jt] = init2TS

            for (v2, in_v2) in v2list:
                omega1escape[v2, jt] = fin2TS

        # Next, we need to build the lists for omega3 and omega4 lists
        for jt, jlist in enumerate(self.symjumplist_omega43_all):

            # The first state is a complex state, the second state is a mixed state.
            # This has been checked in test_crystal stars - look it up
            st1 = jlist[0].state1
            st2 = jlist[0].state2
            # If the solutes are not already at the origin, there is some error and it will show up
            # while getting the crystal stars.

            # get the crystal stars
            crStar1 = self.vkinetic.starset.complexIndexdict[st1][1]
            crStar2 = self.vkinetic.starset.mixedindexdict[st2][1] - self.vkinetic.starset.mixedstartindex
            # crStar2 is the same as the "Wyckoff" index for the mixed dumbbell state.

            init2TS = np.exp(-bFT4[jt] + bFSdb[crStar1])  # complex (bFSdb) to transition state
            fin2TS = np.exp(-bFT3[jt] + bFdb2[crStar2])  # mixed (bFdb2) to transition state.

            # symmetrized rates for omega3 and omega4 are equal
            omega4[jt] = np.sqrt(init2TS * fin2TS)
            omega3[jt] = omega4[jt]  # symmetry condition : = np.sqrt(fin2ts * init2Ts)

            # get the vector stars
            v1list = self.vkinetic.stateToVecStar_pure[st1]
            v2list = self.vkinetic.stateToVecStar_mixed[st2]

            for (v1, in_v1) in v1list:
                omega4escape[v1, jt] = init2TS

            for (v2, in_v2) in v2list:
                omega3escape[v2 - self.vkinetic.Nvstars_pure, jt] = fin2TS

        return (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),\
               (omega4, omega4escape)

    def makeGF(self, bFdb0, bFdb2, bFT0, bFT2, omegas):
        """
        Constructs the N_vs x N_vs GF matrix.
        """
        if not hasattr(self, 'g2'):
            raise AttributeError("g2 not found yet. Please run calc_eta first.")

        Nvstars_pure = self.vkinetic.Nvstars_pure

        (rate0expansion, rate0escape), (rate1expansion, rate1escape), (rate2expansion, rate2escape), \
        (rate3expansion, rate3escape), (rate4expansion, rate4escape) = self.rateExps

        # omega2 and omega2escape will not be needed here, but we still need them to calculate the uncorrelated part.
        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),\
        (omega4, omega4escape) = omegas

        GF20 = np.zeros((self.vkinetic.Nvstars, self.vkinetic.Nvstars))

        # left-upper part of GF20 = Nvstars_pure x Nvstars_pure g0 matrix
        # right-lower part of GF20 = Nvstars_mixed x Nvstars_mixed g2 matrix

        pre0, pre0T = np.ones_like(bFdb0), np.ones_like(bFT0)
        pre2, pre2T = np.ones_like(bFdb2), np.ones_like(bFT2)

        self.GFcalc_pure.SetRates(pre0, bFdb0, pre0T, bFT0)
        self.GFcalc_mixed.SetRates(pre2, bFdb2, pre2T, bFT2)

        GF0 = np.array([self.GFcalc_pure(tup[0][0], tup[0][1], tup[1]) for tup in
                        [star[0] for star in self.GFstarset_pure]])

        # noinspection PyTypeChecker
        GF2 = np.array([self.g2[tup[0][0], tup[0][1]] for tup in
                        [star[0] for star in self.GFstarset_mixed]])

        GF20[Nvstars_pure:, Nvstars_pure:] = np.dot(self.GFexpansion_mixed, GF2)
        GF20[:Nvstars_pure, :Nvstars_pure] = np.dot(self.GFexpansion_pure, GF0)

        # make delta omega
        delta_om = np.zeros((self.vkinetic.Nvstars, self.vkinetic.Nvstars))

        # off-diagonals
        delta_om[:Nvstars_pure, :Nvstars_pure] += np.dot(rate1expansion, omega1) - np.dot(rate0expansion, omega0)
        delta_om[Nvstars_pure:, :Nvstars_pure] += np.dot(rate3expansion, omega3)
        delta_om[:Nvstars_pure, Nvstars_pure:] += np.dot(rate4expansion, omega4)

        # escapes
        # omega1 and omega4 terms
        for i, starind in enumerate(self.vkinetic.vstar2star[:self.vkinetic.Nvstars_pure]):
            #######
            symindex = self.vkinetic.starset.star2symlist[starind]
            delta_om[i, i] += \
                np.dot(rate1escape[i, :], omega1escape[i, :])-\
                np.dot(rate0escape[i, :], omega0escape[symindex, :])+\
                np.dot(rate4escape[i, :], omega4escape[i, :])

        # omega3 terms
        for i in range(Nvstars_pure, self.vkinetic.Nvstars):
            delta_om[i, i] += np.dot(rate3escape[i - Nvstars_pure, :], omega3escape[i - Nvstars_pure, :])

        GF_total = np.dot(np.linalg.inv(np.eye(self.vkinetic.Nvstars) + np.dot(GF20, delta_om)), GF20)

        return GF_total, GF20, delta_om

    def L_ij(self, bFdb0, bFT0, bFdb2, bFT2, bFS, bFSdb, bFT1, bFT3, bFT4):

        """
        bFdb0[i] = beta*ene_pdb[i] - ln(pre_pdb[i]), i=1,2...,N_pdbcontainer.symorlist - pure dumbbell free energy
        bFdb2[i] = beta*ene_mdb[i] - ln(pre_mdb[i]), i=1,2...,N_mdbcontainer.symorlist - mixed dumbbell free energy
        bFS[i] = beta*ene_S[i] - _ln(pre_S[i]), i=1,2,..N_Wyckoff - site free energy for solute.
        THE ABOVE THREE VALUES ARE NOT SHIFTED RELATIVE TO THEIR RESPECTIVE MINIMUM VALUES.
        We need them to be unshifted to be able to normalize the state probabilities, which requires complex and
        mixed dumbbell energies to be with respect to the same reference. Shifting with their respective minimum values
        disturbs this.
        Wherever shifting is required, we'll do it there.

        bFSdb - beta*ene_Sdb[i] - ln(pre_Sdb[i]) [i=1,2...,mixedstartindex](binding)] excess free energy of interaction
        between a solute and a pure dumbbell in it's vicinity. This must be non-zero only for states within the
        thermodynamic shell. So the size is restricted to the number of thermodynamic crystal stars.

        Jump barrier free energies (See preene2betaene for details):
        bFT0[i] = beta*ene_TS[i] - ln(pre_TS[i]), i=1,2,...,N_omega0 - Shifted
        bFT2[i] = beta*ene_TS[i] - ln(pre_TS[i]), i=1,2,...,N_omega2 - Shited
        bFT1[i] = beta*eneT1[i] - len(preT1[i]) -> i = 1,2..,N_omega1 - Shifted
        bFT3[i] = beta*eneT3[i] - len(preT3[i]) -> i = 1,2..,N_omega3 - Shifted
        bFT4[i] = beta*eneT4[i] - len(preT4[i]) -> i = 1,2..,N_omega4 - Shifted
        # See the preene2betaene function to see what the shifts are.
        Return:
            L_aa, L_bb, L_ab - needs to be multiplied by c_db/KT
        """
        if not len(bFSdb) == self.thermo.mixedstartindex:
            raise TypeError("Interaction energies must be present for all and only all thermodynamic shell states.")

        # Get the minimum free energies of solutes, pure dumbbells and mixed dumbbells
        bFdb0_min = np.min(bFdb0)
        bFdb2_min = np.min(bFdb2)
        bFS_min = np.min(bFS)

        pre0, pre0T = np.ones_like(bFdb0), np.ones_like(bFT0)
        pre2, pre2T = np.ones_like(bFdb2), np.ones_like(bFT2)

        # Make the unsymmetrized rates for calculating eta0
        symrate0list = symmratelist(self.jnet0_indexed, pre0, bFdb0 - bFdb0_min, pre0T, bFT0,
                                 self.vkinetic.starset.pdbcontainer.invmap)

        symrate2list = symmratelist(self.jnet2_indexed, pre2, bFdb2 - bFdb2_min, pre2T, bFT2,
                                 self.vkinetic.starset.mdbcontainer.invmap)

        # Make the symmetrized rates for calculating GF, bias and gamma.
        # First, make bFSdb_total from individual solute and pure dumbbell free energies and the binding free energy,
        # i.e, bFdb0, bFS, bFSdb (binding), respectively.
        # For origin states, this should be in such a way so that omega_0 + del_omega = 0 -> this is taken care of in
        # getsymmrates function.

        bFSdb_total = np.zeros(self.vkinetic.starset.mixedstartindex)
        bFSdb_total_shift = np.zeros(self.vkinetic.starset.mixedstartindex)

        # first, just add up the solute and dumbbell energies. We will add in the corrections to the thermo shell states
        # later.
        # Also, we need to keep an unshifted version to be able to normalize the probabilities
        for starind, star in enumerate(self.vkinetic.starset.stars[:self.vkinetic.starset.mixedstartindex]):
            # For origin complex states, do nothing - leave them as zero.
            if star[0].is_zero(self.vkinetic.starset.pdbcontainer):
                continue
            symindex = self.vkinetic.starset.star2symlist[starind]
            # First, get the unshifted value
            bFSdb_total[starind] = bFdb0[symindex] + bFS[self.invmap_solute[star[0].i_s]]
            bFSdb_total_shift[starind] = bFSdb_total[starind] - (bFdb0_min + bFS_min)

        # Now add in the changes for the complexes inside the thermodynamic shell.
        for starind, star in enumerate(self.thermo.stars[:self.thermo.mixedstartindex]):
            # Get the symorlist index for the representative state of the star
            if star[0].is_zero(self.thermo.pdbcontainer):
                continue
            # keep the total energies zero for origin states.
            kinStarind = self.thermo2kin[starind]
            bFSdb_total[kinStarind] += bFSdb[starind]
            bFSdb_total_shift[kinStarind] += bFSdb[starind]
        # We incorporate a separate "shift" array so that even after shifting, the origin state energies remain
        # zero.
        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),\
        (omega4, omega4escape) = self.getsymmrates(bFdb0 - bFdb0_min, bFdb2 - bFdb2_min, bFSdb_total_shift, bFT0, bFT1,
                                                   bFT2, bFT3, bFT4)

        # Update the bias expansions
        self.update_bias_expansions(symrate0list, symrate2list)

        # Make the Greens function
        omegas = ((omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),
                  (omega4, omega4escape))

        GF_total, GF20, del_om = self.makeGF(bFdb0 - bFdb0_min, bFdb2 - bFdb2_min, bFT0, bFT2, omegas)

        # Once the GF is built, make the correlated part of the transport coefficient
        # First we make the projection of the bias vector
        self.biases_solute_vs = np.zeros(self.vkinetic.Nvstars)
        self.biases_solvent_vs = np.zeros(self.vkinetic.Nvstars)

        Nvstars_pure = self.vkinetic.Nvstars_pure
        # The values for the mixed dumbbells are stored first, and then the complexes
        # Among the complexes, the values for the origin states are stored first, then the other ones.

        # bias_..._new = the bias vector produced after updating with eta0 vectors.
        self.biases_solute_vs[:Nvstars_pure] = np.dot(self.bias4_solute_new, omega4)
        # For the solutes in complex configurations, the only local bias comes due to displacements during association.
        # complex-complex jumps leave the solute unchanged and hence do not contribute to solute bias.
        self.biases_solute_vs[Nvstars_pure:] = np.dot(self.bias3_solute_new, omega3)
        # remember that the omega2 bias is the non-local bias, and so has been subtracted out.
        # See line 350 in the test module to check that bias2_solute_new is all zeros.

        # omega1 has total rates. So, to get the non-local change in the rates, we must subtract out the corresponding
        # non-local rates.
        omega1_change = omega1 - np.array([omega0[jt] for jt in self.om1types])
        # This gives us only the change in the rates within the kinetic shell due to solute interactions.
        self.biases_solvent_vs[:Nvstars_pure] = np.dot(self.bias1_solvent_new, omega1_change) +\
                                            np.dot(self.bias4_solvent_new, omega4)
        # For solvents out of complex states, both omega1 and omega4 jumps contribute to the local bias.

        self.biases_solvent_vs[Nvstars_pure:] = np.dot(self.bias3_solvent_new, omega3)
        # In the mixed state space, the local bias comes due only to the omega3(dissociation) jumps.

        # Next, we create the gamma vector, projected onto the vector stars
        gamma_solute_vs = np.dot(GF_total, self.biases_solute_vs)
        gamma_solvent_vs = np.dot(GF_total, self.biases_solvent_vs)

        # Next we produce the outer product in the basis of the vector star vector state functions
        # a=solute, b=solvent
        L_c_aa = np.dot(np.dot(self.kinouter, gamma_solute_vs), self.biases_solute_vs)
        L_c_bb = np.dot(np.dot(self.kinouter, gamma_solvent_vs), self.biases_solvent_vs)
        L_c_ab = np.dot(np.dot(self.kinouter, gamma_solvent_vs), self.biases_solute_vs)

        # Next, we get to the bare or uncorrelated terms
        # First, we have to generate the probability arrays and multiply them with the ratelists. This will
        # Give the probability-square-root multiplied rates in the uncorrelated terms.

        # First, we work out the probabilities and their normalization.
        mixed_prob = np.zeros(len(self.vkinetic.starset.mixedstates))
        complex_prob = np.zeros(len(self.vkinetic.starset.complexStates))

        # get the complex boltzmann factors - unshifted
        # TODO Should we at least shift with respect to the minimum of the two (complex, mixed)
        # Otherwise, how do we think of preventing overflow?
        for starind, star in enumerate(self.vkinetic.starset.stars[:self.vkinetic.starset.mixedstartindex]):
            for state in star:
                complex_prob[self.vkinetic.starset.complexIndexdict[state][0]] = np.exp(-bFSdb_total[starind])

        # Form the mixed dumbbell boltzmann factors and the partition function
        part_func = 0.
        # First add in the mixed dumbbell contributions
        for siteind, wyckind in enumerate(self.vkinetic.starset.mdbcontainer.invmap):
            # don't need the site index but the wyckoff index corresponding to the site index.
            part_func += np.exp(-bFdb2[wyckind])
            mixed_prob[siteind] = np.exp(-bFdb2[wyckind])

        # Now add in the non-interactive complex contribution
        for dbsiteind, dbwyckind in enumerate(self.vkinetic.starset.pdbcontainer.invmap):
            for solsiteind, solwyckind in enumerate(self.invmap_solute):
                part_func += np.exp(-(bFdb0[dbwyckind] + bFS[solwyckind]))


        complex_prob *= 1./part_func
        mixed_prob *= 1./part_func

        stateprobs = (complex_prob, mixed_prob)

        # For the complex states, weed out the origin state probabilities
        for stateind, prob in enumerate(complex_prob):
            if self.vkinetic.starset.complexStates[stateind].is_zero(self.vkinetic.starset.pdbcontainer):
                complex_prob[stateind] = 0.

        # This ensure that summing over all complex + mixed states gives a probability of 1.
        # Note that this is why the bFdb0, bFS and bFdb2 values have to be entered unshifted.
        # The complex and mixed dumbbell energies need to be with respect to the same reference.

        # First, make the square root prob * rate lists to multiply with the rates

        # First, omega1
        # Is there a way to combine all of the next four loops?
        prob_om1 = np.zeros(len(self.jnet_1))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet1_indexed]):
            prob_om1[jt] = np.sqrt(complex_prob[IS]*complex_prob[FS])*omega1[jt]

        prob_om2 = np.zeros(len(self.jnet2))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet2_indexed]):
            prob_om2[jt] = np.sqrt(mixed_prob[IS]*mixed_prob[FS])*omega2[jt]

        prob_om4 = np.zeros(len(self.symjumplist_omega4))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.symjumplist_omega4_indexed]):
            prob_om4[jt] = np.sqrt(complex_prob[IS]*mixed_prob[FS])*omega4[jt]

        prob_om3 = np.zeros(len(self.symjumplist_omega3))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.symjumplist_omega3_indexed]):
            prob_om3[jt] = np.sqrt(mixed_prob[IS]*complex_prob[FS])*omega3[jt]

        probs = (prob_om1, prob_om2, prob_om4, prob_om3)
        start = time.time()
        # Generate the bare expansions
        (D1expansion_aa, D1expansion_bb, D1expansion_ab),\
        (D2expansion_aa, D2expansion_bb, D2expansion_ab),\
        (D3expansion_aa, D3expansion_bb, D3expansion_ab),\
        (D4expansion_aa, D4expansion_bb, D4expansion_ab) = self.bareExpansion(self.eta0total_solute,
                                                                              self.eta0total_solvent)

        print("time for uncorrelated term = {}".format(time.time()-start))
        L_uc_aa = np.dot(D1expansion_aa, prob_om1) + np.dot(D2expansion_aa, prob_om2) +\
                  np.dot(D3expansion_aa, prob_om3) + np.dot(D4expansion_aa, prob_om4)

        L_uc_bb = np.dot(D1expansion_bb, prob_om1) + np.dot(D2expansion_bb, prob_om2) + \
                  np.dot(D3expansion_bb, prob_om3) + np.dot(D4expansion_bb, prob_om4)

        L_uc_ab = np.dot(D1expansion_ab, prob_om1) + np.dot(D2expansion_ab, prob_om2) + \
                  np.dot(D3expansion_ab, prob_om3) + np.dot(D4expansion_ab, prob_om4)

        return (L_uc_aa,L_c_aa), (L_uc_bb,L_c_bb), (L_uc_ab,L_c_ab), GF_total, GF20, del_om, part_func, probs, omegas, stateprobs
