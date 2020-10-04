import numpy as np
from numpy.core.multiarray import ndarray
import onsager.crystal as crystal
from onsager.crystalStars import zeroclean
from representations import *
from GFcalc_dumbbells import GF_dumbbells
import stars
import vector_stars
from functools import reduce
from scipy.linalg import pinv
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
        D0 = np.zeros((3, 3))
        stateene = np.array([betaene[w] for w in self.container.invmap])

        for jlist, rates, symmrates, bET in zip(self.jumpnetwork, rates_lst, symmrates_lst, betaeneT):
            for ((i, j), dx), rate, symmrate in zip(jlist, rates, symmrates):
                omega_ij[i, j] += symmrate
                omega_ij[i, i] -= rate
                domega_ij[i, j] += symmrate * (bET - 0.5 * (stateene[i] + stateene[j]))
                bias_i[i] += sqrtrho[i] * rate * dx

                D0 += 0.5 * np.outer(dx, dx) * rho[i] * rate

        gamma_i = np.tensordot(pinv(omega_ij), bias_i, axes=(1, 0))
        Dcorr = np.zeros((3, 3))
        for i in range(self.N):
            Dcorr += np.outer(bias_i[i], gamma_i[i])

        return D0, Dcorr, omega_ij, pinv(omega_ij)


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
                 closestdistance, NGFmax=4, Nthermo=0, omega43_indices=None):
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
        :param self.omega43_indices - list of indices of omega43 jumps to keep.
        """
        # All the required quantities will be extracted from the containers as we move along
        self.pdbcontainer = pdbcontainer
        self.mdbcontainer = mdbcontainer
        (self.jnet0, self.jnet0_indexed), (self.jnet2, self.jnet2_indexed) = jnet0data, jnet2data
        self.omega43_indices = omega43_indices
        self.crys = pdbcontainer.crys  # we assume this is the same in both containers
        self.chem = pdbcontainer.chem

        # Create the solute invmap
        sitelist_solute = self.crys.sitelist(self.chem)
        self.invmap_solute = np.zeros(len(self.crys.basis[self.chem]), dtype=int)
        for wyckind, ls in enumerate(sitelist_solute):
            for site in ls:
                self.invmap_solute[site] = wyckind

        # self.jnet2_indexed = self.kinetic.starset.jnet2_indexed
        print("initializing thermo")
        self.thermo = stars.StarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
                                    (self.jnet2, self.jnet2_indexed))

        print("initializing kin")
        self.kinetic = stars.StarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
                                     (self.jnet2, self.jnet2_indexed))

        print("initializing NN")
        start = time.time()
        # Note - even if empty, our starsets go out to atleast the NNstar - later we'll have to keep this in mind
        self.NNstar = stars.StarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
                                    (self.jnet2, self.jnet2_indexed), 2)
        print("2NN Shell initialization time: {}\n".format(time.time() - start))
        self.vkinetic = vector_stars.vectorStars()

        # Make GF calculators.
        self.GFcalc_pure = GF_dumbbells(self.pdbcontainer, self.jnet0_indexed, Nmax=NGFmax, kptwt=None)
        # self.GFcalc_mixed = GF_dumbbells(self.mdbcontainer, self.jnet2_indexed, Nmax=4, kptwt=None)

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
        (self.jnet1, self.jnet1_indexed, self.jtags1), self.om1types = self.vkinetic.starset.jumpnetwork_omega1()

        # next, omega3 and omega_4, indexed to pure and mixed states
        # If data already provided, use those
        (self.jnet43, self.jnet43_indexed), (self.jnet4, self.jnet4_indexed, self.jtags4), \
        (self.jnet3, self.jnet3_indexed, self.jtags3) = self.vkinetic.starset.jumpnetwork_omega34(cutoff, solv_solv_cut,
                                                                                                  solt_solv_cut, closestdistance)

    def regenerate43(self, indices):
        """
        This will be used to extract a subset of omega43 jumps of interest
        :param indices: indices - indices of jump lists to keep
        """
        self.jnet43 = [self.jnet43[i] for i in indices]
        self.jnet43_indexed = [self.jnet43_indexed[i] for i in indices]

        self.jnet4 = [self.jnet4[i] for i in indices]
        self.jnet4_indexed = [self.jnet4_indexed[i] for i in indices]
        self.jtags4 = [self.jtags4[i] for i in indices]

        self.jnet3 = [self.jnet3[i] for i in indices]
        self.jnet3_indexed = [self.jnet3_indexed[i] for i in indices]
        self.jtags3 = [self.jtags3[i] for i in indices]

        self.rateExps = self.vkinetic.rateexpansion(self.jnet1, self.om1types, self.jnet43)

        # # Generate the bias expansions
        self.biases = self.vkinetic.biasexpansion(self.jnet1, self.jnet2, self.om1types, self.jnet43)

    def generate(self, Nthermo, cutoff, solt_solv_cut, solv_solv_cut, closestdistance):

        if Nthermo == getattr(self, "Nthermo", 0): return
        self.Nthermo = Nthermo
        print("generating thermodynamic shell")
        start = time.time()
        self.thermo.generate(Nthermo)
        print("thermodynamic shell generated: {}".format(time.time() - start))
        print("Total number of states in Thermodynamic Shell - {}, {}".format(len(self.thermo.complexStates),
                                                                              len(self.thermo.mixedstates)))
        print("generating kinetic shell")
        start = time.time()
        self.kinetic.generate(Nthermo + 1)
        print("Kinetic shell generated: {}".format(time.time() - start))
        print("Total number of states in Kinetic Shell - {}, {}".format(len(self.kinetic.complexStates),
                                                                        len(self.kinetic.mixedstates)))
        # self.Nmixedstates = len(self.kinetic.mixedstates)
        # self.NcomplexStates = len(self.kinetic.complexStates)
        print("generating kinetic shell vector starset")
        start = time.time()
        self.vkinetic.generate(self.kinetic)  # we generate the vector star out of the kinetic shell
        print("Kinetic shell vector starset generated: {}".format(time.time()-start))
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
        print("Generating Jump networks")
        start = time.time()
        self.generate_jnets(cutoff, solt_solv_cut, solv_solv_cut, closestdistance)
        print("Jump networks generated: {}".format(time.time() - start))

        # Generate the GF expansions
        start = time.time()
        (self.GFstarset_pure, self.GFPureStarInd, self.GFexpansion_pure), \
        (self.GFstarset_mixed, self.GFMixedStarInd, self.GFexpansion_mixed) \
            = self.vkinetic.GFexpansion()
        print("built GFstarsets: {}".format(time.time() - start))

        # generate the rate expansions
        start = time.time()
        self.rateExps = self.vkinetic.rateexpansion(self.jnet1, self.om1types, self.jnet43)
        print("built rate expansions: {}".format(time.time() - start))

        # # Generate the bias expansions
        start = time.time()
        self.biases = self.vkinetic.biasexpansion(self.jnet1, self.jnet2, self.om1types, self.jnet43)
        print("built bias expansions: {}".format(time.time() - start))
        #
        # # generate the outer products of the vector stars
        start = time.time()
        self.kinouter = self.vkinetic.outer()
        print("built outer product tensor:{}".format(time.time() - start))
        # self.clearcache()

    def calc_eta(self, rate0list, omega0escape, rate2list, omega2escape, eta2shift=True):
        """
        Function to calculate the periodic eta vectors.
        rate0list, rate2list - the NON-SYMMETRIZED rate lists for the bare and mixed dumbbell spaces.
        We are calulcating the eta vectors, not the gamma vectors.
        """

        # The non-local bias for the complex space has to be carried out based on the omega0 jumpnetwork,
        # not the omega1 jumpnetwork.This is because all the jumps that are allowed by omega0 out of a given dumbbell
        # state are not there in omega1. That is because omega1 considers only those states that are in the kinetic
        # shell. Not outside it.

        # First, we build up G0
        W0 = np.zeros((len(self.vkinetic.starset.bareStates), len(self.vkinetic.starset.bareStates)))
        # use the indexed omega2 to fill this up - need omega2 indexed to mixed subspace of starset
        for jt, jlist in enumerate(self.jnet0_indexed):
            for jnum, ((i, j), dx) in enumerate(jlist):
                W0[i, j] += rate0list[jt][jnum]  # The unsymmetrized rate for that jump.
                W0[i, i] -= rate0list[jt][jnum]  # Add the same to the diagonal
        # Here, G0 = sum(x_s')G0(x_s') - and we have [sum(x_s')G0(x_s')][sum(x_s')W0(x_s')] = identity
        # The equation can be derived from the Fourier space inverse relations at q=0 for their symmetrized versions.
        self.G0 = pinv(W0)

        W2 = np.zeros((len(self.kinetic.mixedstates),
                       len(self.kinetic.mixedstates)))
        # use the indexed omega2 to fill this up - need omega2 indexed to mixed subspace of starset
        for jt, jlist in enumerate(self.jnet2_indexed):
            for jnum, ((i, j), dx) in enumerate(jlist):
                W2[i, j] += rate2list[jt][jnum]  # The unsymmetrized rate for that jump.
                W2[i, i] -= rate2list[jt][jnum]  # Add the same to the diagonal

        self.G2 = pinv(W2)
        self.W2 = W2

        self.biasBareExpansion = self.biases[-1]

        # First check if non-local biases should be zero anyway (as is the case
        # with highly symmetric lattices - in that case vecpos_bare should be zero sized)
        if len(self.vkinetic.vecpos_bare) == 0:
            self.eta00_solvent = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))
            self.eta00_solute = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))

        # otherwise, we need to build the bare bias expansion
        else:
            # First we build up for just the bare starset

            # We first get the bias vector in the basis of the vector stars.
            # Since we are using symmetrized rates, we only need to consider them
            self.NlsolventVel_bare = np.zeros((len(self.vkinetic.starset.bareStates), self.crys.dim))

            # We evaluate the velocity vectors in the basis of vector wyckoff sets.
            # Need omega0_escape.
            velocity0SolventTotNonLoc = np.array([np.dot(self.biasBareExpansion[i, :],
                                                         omega0escape[self.vkinetic.vwycktowyck_bare[i], :])
                                                  for i in range(len(self.vkinetic.vecpos_bare))])

            # Then, we convert them to cartesian form for each state.
            for st in self.vkinetic.starset.bareStates:
                indlist = self.vkinetic.stateToVecStar_bare[st]
                if len(indlist) != 0:
                    self.NlsolventVel_bare[self.vkinetic.starset.bareindexdict[st][0]][:] = \
                        sum([velocity0SolventTotNonLoc[tup[0]] * self.vkinetic.vecvec_bare[tup[0]][tup[1]] for tup in
                             indlist])

            # Then, we use G0 to get the eta0 vectors. The second 0 in eta00 indicates omega0 space.
            self.eta00_solvent_bare = np.tensordot(self.G0, self.NlsolventVel_bare, axes=(1, 0))
            self.eta00_solute_bare = np.zeros_like(self.eta00_solvent_bare)

            # Now match the non-local biases for complex states to the pure states
            self.eta00_solvent = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))
            self.eta00_solute = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))
            self.NlsolventBias0 = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))

            for i, state in enumerate(self.vkinetic.starset.complexStates):
                dbstate_ind = state.db.iorind
                self.eta00_solvent[i, :] = self.eta00_solvent_bare[dbstate_ind, :].copy()
                self.NlsolventBias0[i, :] = self.NlsolventVel_bare[dbstate_ind, :].copy()

        # For the mixed dumbbell space, translational symmetry tells us that we only need to consider the dumbbells
        # in the first unit cell only. So, we are already considering the bias out of every state we need to consider.

        if eta2shift:

            bias2exp_solute, bias2exp_solvent = self.biases[2]
            self.NlsoluteVel_mixed = np.zeros((len(self.vkinetic.starset.mixedstates), self.crys.dim))
            Nvstars_mixed = self.vkinetic.Nvstars - self.vkinetic.Nvstars_pure
            Nvstars_pure = self.vkinetic.Nvstars_pure

            mstart = self.kinetic.mixedstartindex

            # We evaluate the velocity vectors in the basis of vector wyckoff sets.
            # Need omega2_escape.

            velocity2SolventTotNonLoc = np.array([np.dot(bias2exp_solvent[i - Nvstars_pure, :],
                                                         omega2escape[self.vkinetic.vstar2star[i] - mstart, :])
                                                  for i in range(Nvstars_pure, self.vkinetic.Nvstars)])

            velocity2SoluteTotNonLoc = np.array([np.dot(bias2exp_solute[i - Nvstars_pure, :],
                                                        omega2escape[self.vkinetic.vstar2star[i] - mstart, :])
                                                 for i in range(Nvstars_pure, self.vkinetic.Nvstars)])

            self.NlsolventVel_mixed = np.zeros((len(self.kinetic.mixedstates), self.crys.dim))
            self.NlsoluteVel_mixed = np.zeros((len(self.kinetic.mixedstates), self.crys.dim))

            # Then, we convert them to cartesian form for each state.
            for st in self.vkinetic.starset.mixedstates:
                indlist = self.vkinetic.stateToVecStar_mixed[st]
                if len(indlist) != 0:
                    self.NlsolventVel_mixed[self.vkinetic.starset.mixedindexdict[st][0]][:] = \
                        sum([velocity2SolventTotNonLoc[tup[0] - Nvstars_pure] * self.vkinetic.vecvec[tup[0]][tup[1]] for tup
                             in
                             indlist])
                    self.NlsoluteVel_mixed[self.vkinetic.starset.mixedindexdict[st][0]][:] = \
                        sum([velocity2SoluteTotNonLoc[tup[0] - Nvstars_pure] * self.vkinetic.vecvec[tup[0]][tup[1]] for tup
                             in
                             indlist])

            # Then, we use G2 to get the eta2 vectors. The second 2 in eta02 indicates omega2 space.
            self.eta02_solvent = np.tensordot(self.G2, self.NlsolventVel_mixed, axes=(1, 0))
            self.eta02_solute = np.tensordot(self.G2, self.NlsoluteVel_mixed, axes=(1, 0))

        else:
            self.eta02_solvent = np.zeros((len(self.kinetic.mixedstates), self.crys.dim))
            self.eta02_solute = np.zeros((len(self.kinetic.mixedstates), self.crys.dim))

        # So what do we have up until now?
        # We have constructed the Nstates x 3 eta0 vectors for complex states
        # We need to produce a total eta vector list.

        # Nothing called solute eta vector in bare dumbbell jumps.
        self.eta0total_solute = np.zeros((len(self.vkinetic.starset.complexStates) +
                                          len(self.vkinetic.starset.mixedstates), self.crys.dim))
        # noinspection PyAttributeOutsideInit
        self.eta0total_solvent = np.zeros((len(self.vkinetic.starset.complexStates) +
                                           len(self.vkinetic.starset.mixedstates), self.crys.dim))

        # Just copy the portion for the complex states, leave mixed dumbbell state space as zeros.
        self.eta0total_solvent[:len(self.vkinetic.starset.complexStates), :] = self.eta00_solvent.copy()
        self.eta0total_solvent[len(self.vkinetic.starset.complexStates):, :] = self.eta02_solvent.copy()
        self.eta0total_solute[len(self.vkinetic.starset.complexStates):, :] = self.eta02_solute.copy()

    def bias_changes(self, eta2shift=True):
        """
        Function that allows us to construct new bias and bare expansions based on the eta vectors already calculated.

        We don't want to repeat the construction of the jumpnetwork based on the recalculated displacements after
        subtraction of the eta vectors (as in the variational principle).

        The steps are illustrated in the GM slides of Feb 25, 2019 - will include in the detailed documentation later on
        """
        # create updates to the bias expansions
        # Construct the projection of eta vectors
        self.delbias1expansion_solute = np.zeros_like(self.biases[1][0])
        self.delbias1expansion_solvent = np.zeros_like(self.biases[1][1])

        self.delbias4expansion_solute = np.zeros_like(self.biases[4][0])
        self.delbias4expansion_solvent = np.zeros_like(self.biases[4][1])

        self.delbias3expansion_solute = np.zeros_like(self.biases[3][0])
        self.delbias3expansion_solvent = np.zeros_like(self.biases[3][0])

        self.delbias2expansion_solute = np.zeros_like(self.biases[2][0])
        self.delbias2expansion_solvent = np.zeros_like(self.biases[2][0])

        if eta2shift:
            for i in range(self.vkinetic.Nvstars_pure):
                # get the representative state(its index in complexStates) and vector
                v0 = self.vkinetic.vecvec[i][0]
                st0 = self.vkinetic.starset.complexIndexdict[self.vkinetic.vecpos[i][0]][0]
                # Index of the state in the flat list
                eta_proj_solute = np.dot(self.eta0total_solute, v0)
                eta_proj_solvent = np.dot(self.eta0total_solvent, v0)
                # Now go through the omega1 jump network tags
                for jt, initindexdict in enumerate(self.jtags1):
                    # see if there's an array corresponding to the initial state
                    if not st0 in initindexdict:
                        # if the representative state does not occur as an initial state in any of the jumps, continue.
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

            # again, no changes occur to the bias vectors out of mixed dumbbell space, so do nothing.

            for i in range(self.vkinetic.Nvstars - self.vkinetic.Nvstars_pure):
                # get the representative state(its index in mixedstates) and vector
                v0 = self.vkinetic.vecvec[i + self.vkinetic.Nvstars_pure][0]
                st0 = self.vkinetic.starset.mixedindexdict[self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure][0]][0]
                # Form the projection of the eta vectors on v0
                eta_proj_solute = np.dot(self.eta0total_solute, v0)
                eta_proj_solvent = np.dot(self.eta0total_solvent, v0)

                # Now go through the omega2 jump network tags
                for jt, initindexdict in enumerate(self.jtags2):
                    # see if there's an array corresponding to the initial state
                    if not st0 in initindexdict:
                        continue
                    self.delbias2expansion_solute[i, jt] += len(self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * \
                                                            np.sum(np.dot(initindexdict[st0], eta_proj_solute))

                    self.delbias2expansion_solvent[i, jt] += len(self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * \
                                                             np.sum(np.dot(initindexdict[st0], eta_proj_solvent))

                # However, need to update for omega3 because the solvent shift vector in the complex space is not zero.
                # Now let's build the change expansion for omega3

                for jt, initindexdict in enumerate(self.jtags3):
                    # see if there's an array corresponding to the initial state
                    if not st0 in initindexdict:
                        continue
                    self.delbias3expansion_solute[i, jt] += len(self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * \
                                                            np.sum(np.dot(initindexdict[st0], eta_proj_solute))
                    self.delbias3expansion_solvent[i, jt] += len(self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * \
                                                             np.sum(np.dot(initindexdict[st0], eta_proj_solvent))

    def update_bias_expansions(self, rate0list, omega0escape, rate2list, omega2escape, eta2shift=True):
        self.calc_eta(rate0list, omega0escape, rate2list, omega2escape, eta2shift=eta2shift)
        self.bias_changes(eta2shift=eta2shift)
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

        Params: The eta vectors in each state.


        In mixed dumbbell space, both solute and solvent will have uncorrelated contributions.
        The mixed dumbbell space is completely non-local.
        """
        # a = solute, b = solvent
        # eta0_solute, eta0_solvent = self.eta0total_solute, self.eta0total_solvent
        # Stores biases out of complex states, followed by mixed dumbbell states.
        jumpnetwork_omega1, jumptype, jumpnetwork_omega2, jumpnetwork_omega3, jumpnetwork_omega4 = \
            self.jnet1_indexed, self.om1types, self.jnet2_indexed, self.jnet3_indexed, \
            self.jnet4_indexed

        Ncomp = len(self.vkinetic.starset.complexStates)

        # We need the D0expansion to evaluate the modified non-local contribution
        # outside the kinetic shell.

        dim = self.crys.dim

        D0expansion_bb = np.zeros((dim, dim, len(self.jnet0)))

        # Omega1 contains the total rate and not just the change.
        D1expansion_aa = np.zeros((dim, dim, len(jumpnetwork_omega1)))
        D1expansion_bb = np.zeros((dim, dim, len(jumpnetwork_omega1)))
        D1expansion_ab = np.zeros((dim, dim, len(jumpnetwork_omega1)))

        D2expansion_aa = np.zeros((dim, dim, len(jumpnetwork_omega2)))
        D2expansion_bb = np.zeros((dim, dim, len(jumpnetwork_omega2)))
        D2expansion_ab = np.zeros((dim, dim, len(jumpnetwork_omega2)))

        D3expansion_aa = np.zeros((dim, dim, len(jumpnetwork_omega3)))
        D3expansion_bb = np.zeros((dim, dim, len(jumpnetwork_omega3)))
        D3expansion_ab = np.zeros((dim, dim, len(jumpnetwork_omega3)))

        D4expansion_aa = np.zeros((dim, dim, len(jumpnetwork_omega4)))
        D4expansion_bb = np.zeros((dim, dim, len(jumpnetwork_omega4)))
        D4expansion_ab = np.zeros((dim, dim, len(jumpnetwork_omega4)))

        # iorlist_pure = self.pdbcontainer.iorlist
        # iorlist_mixed = self.mdbcontainer.iorlist
        # Need versions for solute and solvent - solute dusplacements are zero anyway
        for k, jt, jumplist in zip(itertools.count(), jumptype, jumpnetwork_omega1):
            d0 = np.sum(
                0.5 * np.outer(dx + eta0_solvent[i] - eta0_solvent[j], dx + eta0_solvent[i] - eta0_solvent[j]) for
                (i, j), dx in jumplist)
            D0expansion_bb[:, :, jt] += d0
            D1expansion_bb[:, :, k] += d0
            # For solutes, don't need to do anything for omega1 and omega0 - solute does not move anyway
            # and therefore, their non-local eta corrections are also zero.

        for jt, jumplist in enumerate(jumpnetwork_omega2):
            # Build the expansions directly
            for (IS, FS), dx in jumplist:
                # o1 = iorlist_mixed[self.vkinetic.starset.mixedstates[IS].db.iorind][1]
                # o2 = iorlist_mixed[self.vkinetic.starset.mixedstates[FS].db.iorind][1]
                dx_solute = dx + eta0_solute[Ncomp + IS] - eta0_solute[Ncomp + FS]  # + o2 / 2. - o1 / 2.
                dx_solvent = dx + eta0_solvent[Ncomp + IS] - eta0_solvent[Ncomp + FS]  # - o2 / 2. + o1 / 2.
                D2expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D2expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                D2expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        for jt, jumplist in enumerate(jumpnetwork_omega3):
            for (IS, FS), dx in jumplist:
                # o1 = iorlist_mixed[self.vkinetic.starset.mixedstates[IS].db.iorind][1]
                dx_solute = eta0_solute[Ncomp + IS] - eta0_solute[FS]  # -o1 / 2.
                dx_solvent = dx + eta0_solvent[Ncomp + IS] - eta0_solvent[FS]  # + o1 / 2.
                D3expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D3expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                D3expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        for jt, jumplist in enumerate(jumpnetwork_omega4):
            for (IS, FS), dx in jumplist:
                # o2 = iorlist_mixed[self.vkinetic.starset.mixedstates[FS].db.iorind][1]
                dx_solute = eta0_solute[IS] - eta0_solute[Ncomp + FS]  # o2 / 2. +
                dx_solvent = dx + eta0_solvent[IS] - eta0_solvent[Ncomp + FS]  # - o2 / 2.
                D4expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D4expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                D4expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        return zeroclean(D0expansion_bb), \
               (zeroclean(D1expansion_aa), zeroclean(D1expansion_bb), zeroclean(D1expansion_ab)), \
               (zeroclean(D2expansion_aa), zeroclean(D2expansion_bb), zeroclean(D2expansion_ab)), \
               (zeroclean(D3expansion_aa), zeroclean(D3expansion_bb), zeroclean(D3expansion_ab)), \
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
        beta = 1. / kT
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

        omega1 = np.zeros(len(self.jnet1))
        omega1escape = np.zeros((self.vkinetic.Nvstars_pure, len(self.jnet1)))

        omega3 = np.zeros(len(self.jnet3))
        omega3escape = np.zeros((Nvstars_mixed, len(self.jnet3)))

        omega4 = np.zeros(len(self.jnet4))
        omega4escape = np.zeros((self.vkinetic.Nvstars_pure, len(self.jnet4)))

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
        for jt, jlist in enumerate(self.jnet1):

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
        for jt, jlist in enumerate(self.jnet43):

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

        return (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape), \
               (omega4, omega4escape)

    def makeGF(self, bFdb0, bFT0, omegas, mixed_prob):
        """
        Constructs the N_vs x N_vs GF matrix.
        """
        if not hasattr(self, 'G2'):
            raise AttributeError("G2 not found yet. Please run calc_eta first.")

        Nvstars_pure = self.vkinetic.Nvstars_pure

        (rate0expansion, rate0escape), (rate1expansion, rate1escape), (rate2expansion, rate2escape), \
        (rate3expansion, rate3escape), (rate4expansion, rate4escape) = self.rateExps

        # omega2 and omega2escape will not be needed here, but we still need them to calculate the uncorrelated part.
        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape), \
        (omega4, omega4escape) = omegas

        GF02 = np.zeros((self.vkinetic.Nvstars, self.vkinetic.Nvstars))

        # left-upper part of GF02 = Nvstars_pure x Nvstars_pure g0 matrix
        # right-lower part of GF02 = Nvstars_mixed x Nvstars_mixed g2 matrix

        pre0, pre0T = np.ones_like(bFdb0), np.ones_like(bFT0)

        # Make g2 by symmetrizing G2

        # First, form the probability matrix
        P0mixedSqrt = np.diag(np.sqrt(mixed_prob))
        P0mixedSqrt_inv = np.diag(1. / np.sqrt(mixed_prob))

        self.g2 = np.dot(np.dot(P0mixedSqrt, self.G2), P0mixedSqrt_inv)

        self.GFcalc_pure.SetRates(pre0, bFdb0, pre0T, bFT0)

        GF0 = np.array([self.GFcalc_pure(tup[0][0], tup[0][1], tup[1]) for tup in
                        [star[0] for star in self.GFstarset_pure]])

        GF2 = np.array([self.g2[tup[0][0], tup[0][1]] for tup in
                        [star[0] for star in self.GFstarset_mixed]])

        GF02[Nvstars_pure:, Nvstars_pure:] = np.dot(self.GFexpansion_mixed, GF2)
        GF02[:Nvstars_pure, :Nvstars_pure] = np.dot(self.GFexpansion_pure, GF0)

        # make delta omega
        delta_om = np.zeros((self.vkinetic.Nvstars, self.vkinetic.Nvstars))

        # off-diagonals
        delta_om[:Nvstars_pure, :Nvstars_pure] += np.dot(rate1expansion, omega1) - np.dot(rate0expansion, omega0)
        delta_om[Nvstars_pure:, :Nvstars_pure] += np.dot(rate3expansion, omega3)
        delta_om[:Nvstars_pure, Nvstars_pure:] += np.dot(rate4expansion, omega4)

        # escapes
        # omega1 and omega4 terms
        for i, starind in enumerate(self.vkinetic.vstar2star[:Nvstars_pure]):
            #######
            symindex = self.vkinetic.starset.star2symlist[starind]
            delta_om[i, i] += \
                np.dot(rate1escape[i, :], omega1escape[i, :]) - \
                np.dot(rate0escape[i, :], omega0escape[symindex, :]) + \
                np.dot(rate4escape[i, :], omega4escape[i, :])

        # omega3 terms
        for i in range(Nvstars_pure, self.vkinetic.Nvstars):
            delta_om[i, i] += np.dot(rate3escape[i - Nvstars_pure, :], omega3escape[i - Nvstars_pure, :])

        GF_total = np.dot(np.linalg.inv(np.eye(self.vkinetic.Nvstars) + np.dot(GF02, delta_om)), GF02)

        return zeroclean(GF_total), GF02, delta_om

    def L_ij(self, bFdb0, bFT0, bFdb2, bFT2, bFS, bFSdb, bFT1, bFT3, bFT4, eta2shift=False):

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
            L_aa, L_bb, L_ab - needs to be multiplied by Cs*C_db/KT
            Note - L_bb contains local jumps and contribution from mixed dumbbell space.
            L0bb - contains non-local contribution to solvent diffusion. Needs to be multiplied by C_db/KT.
            Note the net solvent transport coefficient is (C_db*L0bb/kT + Cs*C_db*L_bb/kT)
        """
        if not len(bFSdb) == self.thermo.mixedstartindex:
            raise TypeError("Interaction energies must be present for all and only all thermodynamic shell states.")
        for en in bFSdb[self.thermo.mixedstartindex + 2:]:
            if not en == bFSdb[self.thermo.mixedstartindex + 1]:
                raise ValueError("States in kinetic shell have difference reference interaction energy")

        # 1. Get the minimum free energies of solutes, pure dumbbells and mixed dumbbells
        bFdb0_min = np.min(bFdb0)
        bFdb2_min = np.min(bFdb2)
        bFS_min = np.min(bFS)

        # 2. Make the unsymmetrized rates for calculating eta0
        # The energies of bare dumbbells, solutes and mixed dumbbells are not shifted with their minimum values
        # pass them in after shifting them.

        pre0, pre0T = np.ones_like(bFdb0), np.ones_like(bFT0)
        pre2, pre2T = np.ones_like(bFdb2), np.ones_like(bFT2)

        rate0list = ratelist(self.jnet0_indexed, pre0, bFdb0 - bFdb0_min, pre0T, bFT0,
                             self.vkinetic.starset.pdbcontainer.invmap)

        rate2list = ratelist(self.jnet2_indexed, pre2, bFdb2 - bFdb2_min, pre2T, bFT2,
                             self.vkinetic.starset.mdbcontainer.invmap)

        # 3. Make the symmetrized rates and escape rates for calculating eta0, GF, bias and gamma.
        # 3a. First, make bFSdb_total from individual solute and pure dumbbell and the binding free energies,
        # i.e, bFdb0, bFS, bFSdb (binding), respectively.
        # For origin states, this should be in such a way so that omega_0 + del_omega = 0 -> this is taken care of in
        # getsymmrates function.
        # Also, we need to keep a shifted version, to calculate rates.

        bFSdb_total = np.zeros(self.vkinetic.starset.mixedstartindex)
        bFSdb_total_shift = np.zeros(self.vkinetic.starset.mixedstartindex)

        # first, just add up the solute and dumbbell energies.
        # Now adding changes to states to both within and outside the thermodynamics shell. This is because on
        # changing the energy reference, the "interaction energy" might not be zero in the kinetic shell.
        # The kinetic shell is defined as that outside which the omega1 rates are the same as the omega0 rates.
        # THAT is the definition that needs to be satisfied.
        for starind, star in enumerate(self.vkinetic.starset.stars[:self.vkinetic.starset.mixedstartindex]):
            # For origin complex states, do nothing - leave them as zero.
            if star[0].is_zero(self.vkinetic.starset.pdbcontainer):
                continue
            symindex = self.vkinetic.starset.star2symlist[starind]
            # First, get the unshifted value
            bFSdb_total[starind] = bFdb0[symindex] + bFS[self.invmap_solute[star[0].i_s]]
            bFSdb_total_shift[starind] = bFSdb_total[starind] - (bFdb0_min + bFS_min)

        # Now add in the changes for the complexes inside the thermodynamic shell.
        # Note that we are still not making any changes to the origin states.
        # We always keep them as zero.
        for starind, star in enumerate(self.thermo.stars[:self.thermo.mixedstartindex]):
            # Get the symorlist index for the representative state of the star
            if star[0].is_zero(self.thermo.pdbcontainer):
                continue
            # keep the total energies zero for origin states.
            kinStarind = self.thermo2kin[starind]  # Get the index of the thermo star in the kinetic starset
            bFSdb_total[kinStarind] += bFSdb[starind]  # add in the interaction energy to the appropriate index
            bFSdb_total_shift[kinStarind] += bFSdb[starind]

        # 3b. Get the rates and escapes
        # We incorporate a separate "shift" array so that even after shifting, the origin state energies remain
        # zero.
        betaFs = [bFdb0, bFdb2, bFS, bFSdb, bFSdb_total, bFSdb_total_shift, bFT0, bFT1, bFT2, bFT3, bFT4]
        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape), \
        (omega4, omega4escape) = self.getsymmrates(bFdb0 - bFdb0_min, bFdb2 - bFdb2_min, bFSdb_total_shift, bFT0, bFT1,
                                                   bFT2, bFT3, bFT4)

        # 3b.1 - Put them in a tuple to use in makeGF later on - maybe simplify this process later on.
        omegas = ((omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),
                  (omega4, omega4escape))

        # 4. Update the bias expansions
        self.update_bias_expansions(rate0list, omega0escape, rate2list, omega2escape, eta2shift=eta2shift)

        # 5. Work out the probabilities and the normalization - will be needed to produce g2 from G2 (created in bias
        # updates)
        mixed_prob = np.zeros(len(self.vkinetic.starset.mixedstates))
        complex_prob = np.zeros(len(self.vkinetic.starset.complexStates))

        # 5a. get the complex boltzmann factors - unshifted
        # TODO Should we at least shift with respect to the minimum of the two (complex, mixed)
        # Otherwise, how do we think of preventing overflow in case it occurs?
        for starind, star in enumerate(self.vkinetic.starset.stars[:self.vkinetic.starset.mixedstartindex]):
            for state in star:
                if not (self.vkinetic.starset.complexIndexdict[state][1] == starind):
                    raise ValueError("check complexIndexdict")
                # For states outside the thermodynamics shell, there is no interaction and the probabilities are
                # just the product solute and dumbbell probabilities.
                complex_prob[self.vkinetic.starset.complexIndexdict[state][0]] = np.exp(-bFSdb_total[starind])

        # 5b. get the mixed dumbbell boltzmann factors.
        for siteind, wyckind in enumerate(self.vkinetic.starset.mdbcontainer.invmap):
            # don't need the site index but the wyckoff index corresponding to the site index.
            # The energies are not shifted with respect to the minimum
            mixed_prob[siteind] = np.exp(-bFdb2[wyckind])

        # 5c. Form the partition function
        # get the "reference energy" for non-interacting complexes. This is just the value of bFSdb (interaction)
        # for any state in the kinetic shell
        # del_en = bFSdb[self.thermo.mixedstartindex + 1]
        part_func = 0.
        # Now add in the non-interactive complex contribution to the partition function
        for dbsiteind, dbwyckind in enumerate(self.vkinetic.starset.pdbcontainer.invmap):
            for solsiteind, solwyckind in enumerate(self.invmap_solute):
                part_func += np.exp(-(bFdb0[dbwyckind] + bFS[solwyckind]))

        # 5d. Normalize - division by the partition function ensures effects of shifting go away.
        complex_prob *= 1. / part_func
        mixed_prob *= 1. / part_func

        # 6. Get the symmetrized Green's function in the basis of the vector stars and the non-local contribution
        # to solvent (Fe dumbbell) diffusivity.
        # arguments for makeGF - bFdb0 (shifted), bFT0(shifted), omegas, mixed_prob
        # Note about mixed prob: g2_ij = p_mixed(i)^0.5 * G2_ij * p_mixed(j)^-0.5
        # So, at the end of the end the day, it only depends on boltzmann factors of the mixed states.
        # All other factors cancel out (including partition function).
        GF_total, GF02, del_om = self.makeGF(bFdb0 - bFdb0_min, bFT0, omegas, mixed_prob)
        L0bb = self.GFcalc_pure.Diffusivity()
        # 7. Once the GF is built, make the correlated part of the transport coefficient
        # 7a. First we make the projection of the bias vector
        self.biases_solute_vs = np.zeros(self.vkinetic.Nvstars)
        self.biases_solvent_vs = np.zeros(self.vkinetic.Nvstars)

        Nvstars_pure = self.vkinetic.Nvstars_pure
        Nvstars = self.vkinetic.Nvstars

        # 7b. We need the square roots of the probabilities of the representative state of each vector star.
        prob_sqrt_complex_vs = np.array([np.sqrt(complex_prob[self.kinetic.complexIndexdict[vp[0]][0]])
                                         for vp in self.vkinetic.vecpos[:Nvstars_pure]])
        prob_sqrt_mixed_vs = np.array([np.sqrt(mixed_prob[self.kinetic.mixedindexdict[vp[0]][0]])
                                       for vp in self.vkinetic.vecpos[Nvstars_pure:]])

        # bias_..._new = the bias vector produced after updating with eta0 vectors.
        # 7c. For the solutes in complex configurations, the only local bias comes due to displacements during
        # association.
        # complex-complex jumps leave the solute unchanged and hence do not contribute to solute bias.
        self.biases_solute_vs[:Nvstars_pure] = np.array([np.dot(self.bias4_solute_new[i, :], omega4escape[i, :]) *
                                                         prob_sqrt_complex_vs[i] for i in range(Nvstars_pure)])

        # 7d. Next, we work out the updated solute bias in the mixed space.
        # remember that the omega2 bias is the non-local bias, and so has been subtracted out.
        # See test_bias_updates function to check that bias2_solute_new is all zeros.
        self.biases_solute_vs[Nvstars_pure:] = np.array([np.dot(self.bias3_solute_new[i - Nvstars_pure, :],
                                                                omega3escape[i - Nvstars_pure, :]) *
                                                         prob_sqrt_mixed_vs[i - Nvstars_pure]
                                                         for i in range(Nvstars_pure, self.vkinetic.Nvstars)])

        # omega1 has total rates. So, to get the non-local change in the rates, we must subtract out the corresponding
        # non-local rates.
        # This gives us only the change in the rates within the kinetic shell due to solute interactions.
        # The effect of the non-local rates has been cancelled out by subtracting off the eta vectors.
        # For solvents out of complex states, both omega1 and omega4 jumps contribute to the local bias.

        self.del_W1 = np.zeros_like(omega1escape)
        for i in range(Nvstars_pure):
            for jt in range(len(self.jnet1)):
                self.del_W1[i, jt] = omega1escape[i, jt] - \
                                     omega0escape[
                                         self.kinetic.star2symlist[self.vkinetic.vstar2star[i]], self.om1types[jt]]

        self.biases_solvent_vs[:Nvstars_pure] = np.array([(np.dot(self.bias1_solvent_new[i, :], self.del_W1[i, :]) +
                                                           np.dot(self.bias4_solvent_new[i, :], omega4escape[i, :])) *
                                                          prob_sqrt_complex_vs[i] for i in range(Nvstars_pure)])

        self.biases_solvent_vs[Nvstars_pure:] = np.array([np.dot(self.bias3_solvent_new[i - Nvstars_pure, :],
                                                                 omega3escape[i - Nvstars_pure, :]) *
                                                          prob_sqrt_mixed_vs[i - Nvstars_pure]
                                                          for i in range(Nvstars_pure, self.vkinetic.Nvstars)])
        # In the mixed state space, the local bias comes due only to the omega3(dissociation) jumps.
        if not eta2shift:
            # if eta2shift is false, then the bias2_new tensors won't be all zeros
            for i in range(Nvstars_pure, Nvstars):
                st0 = self.vkinetic.vecpos[i][0]
                dbwyck2 = self.mdbcontainer.invmap[st0.db.iorind]

                self.biases_solute_vs[i] += np.dot(self.bias2_solute_new[i - Nvstars_pure, :], omega2escape[dbwyck2, :]) * \
                                     prob_sqrt_mixed_vs[i - Nvstars_pure]

                self.biases_solvent_vs[i] += np.dot(self.bias2_solvent_new[i - Nvstars_pure, :], omega2escape[dbwyck2, :]) * \
                                      prob_sqrt_mixed_vs[i - Nvstars_pure]

        # Next, we create the gamma vector, projected onto the vector stars
        self.gamma_solute_vs = np.dot(GF_total, self.biases_solute_vs)
        self.gamma_solvent_vs = np.dot(GF_total, self.biases_solvent_vs)

        # Next we produce the outer product in the basis of the vector star vector state functions
        # a=solute, b=solvent
        L_c_aa = np.dot(np.dot(self.kinouter, self.gamma_solute_vs), self.biases_solute_vs)
        L_c_bb = np.dot(np.dot(self.kinouter, self.gamma_solvent_vs), self.biases_solvent_vs)
        L_c_ab = np.dot(np.dot(self.kinouter, self.gamma_solvent_vs), self.biases_solute_vs)

        # Next, we get to the bare or uncorrelated terms
        # First, we have to generate the probability arrays and multiply them with the ratelists. This will
        # Give the probability-square-root multiplied rates in the uncorrelated terms.
        # For the complex states, weed out the origin state probabilities
        for stateind, prob in enumerate(complex_prob):
            if self.vkinetic.starset.complexStates[stateind].is_zero(self.vkinetic.starset.pdbcontainer):
                complex_prob[stateind] = 0.

        pr_states = (complex_prob, mixed_prob)  # For testing
        # Next, we need the bare dumbbell probabilities for the non-local part of the solvent-solvent transport
        # coefficients
        bareprobs = stateprob(pre0, bFdb0 - bFdb0_min, self.pdbcontainer.invmap)
        # This ensured that summing over all complex + mixed states gives a probability of 1.
        # Note that this is why the bFdb0, bFS and bFdb2 values have to be entered unshifted.
        # The complex and mixed dumbbell energies need to be with respect to the same reference.

        # First, make the square root prob * rate lists to multiply with the rates
        # TODO Is there a way to combine all of the next four loops?

        prob_om0 = np.zeros(len(self.jnet0))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet0_indexed]):
            prob_om0[jt] = np.sqrt(bareprobs[IS] * bareprobs[FS]) * omega0[jt]

        prob_om1 = np.zeros(len(self.jnet1))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet1_indexed]):
            prob_om1[jt] = np.sqrt(complex_prob[IS] * complex_prob[FS]) * omega1[jt]

        prob_om2 = np.zeros(len(self.jnet2))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet2_indexed]):
            prob_om2[jt] = np.sqrt(mixed_prob[IS] * mixed_prob[FS]) * omega2[jt]

        prob_om4 = np.zeros(len(self.jnet4))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet4_indexed]):
            prob_om4[jt] = np.sqrt(complex_prob[IS] * mixed_prob[FS]) * omega4[jt]

        prob_om3 = np.zeros(len(self.jnet3))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet3_indexed]):
            prob_om3[jt] = np.sqrt(mixed_prob[IS] * complex_prob[FS]) * omega3[jt]

        probs = (prob_om1, prob_om2, prob_om4, prob_om3)

        start = time.time()
        # Generate the bare expansions with modified displacements
        D0expansion_bb, (D1expansion_aa, D1expansion_bb, D1expansion_ab), \
        (D2expansion_aa, D2expansion_bb, D2expansion_ab), \
        (D3expansion_aa, D3expansion_bb, D3expansion_ab), \
        (D4expansion_aa, D4expansion_bb, D4expansion_ab) = self.bareExpansion(self.eta0total_solute,
                                                                              self.eta0total_solvent)

        L_uc_aa = np.dot(D1expansion_aa, prob_om1) + np.dot(D2expansion_aa, prob_om2) + \
                  np.dot(D3expansion_aa, prob_om3) + np.dot(D4expansion_aa, prob_om4)

        L_uc_bb = np.dot(D1expansion_bb, prob_om1) - np.dot(D0expansion_bb, prob_om0) + \
                  np.dot(D2expansion_bb, prob_om2) + np.dot(D3expansion_bb, prob_om3) + np.dot(D4expansion_bb, prob_om4)

        L_uc_ab = np.dot(D1expansion_ab, prob_om1) + np.dot(D2expansion_ab, prob_om2) + \
                  np.dot(D3expansion_ab, prob_om3) + np.dot(D4expansion_ab, prob_om4)

        return L0bb, (L_uc_aa, L_c_aa), (L_uc_bb, L_c_bb), (L_uc_ab, L_c_ab), GF_total, GF02, betaFs, del_om, \
               part_func, probs, omegas, pr_states, D0expansion_bb
