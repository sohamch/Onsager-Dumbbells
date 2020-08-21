from typing import Any, Union

import numpy as np
import onsager.crystal as crystal
from onsager.crystalStars import zeroclean, VectorStarSet, PairState
from states import *
from representations import *
from collections import defaultdict
from stars import *
from functools import reduce
import itertools
import time

class vectorStars(VectorStarSet):
    """
    Stores the vector stars corresponding to a given starset of dumbbell states
    """

    def __init__(self, starset=None):
        """
        Initiates a vector-star generator; work with a given star.

        :param starset: StarSet, from which we pull nearly all of the info that we need
        """
        # vecpos: list of "positions" (state indices) for each vector star (list of lists)
        # vecvec: list of vectors for each vector star (list of lists of vectors)
        # Nvstars: number of vector stars

        self.starset = None
        self.Nvstars = 0
        if starset is not None:
            if starset.Nshells > 0:
                self.generate(starset)

    def generate(self, starset):
        """
        Follows almost the same as that for solute-vacancy case. Only generalized to keep the state
        under consideration unchanged.
        """
        self.starset = None
        if starset.Nshells == 0: return
        if starset == self.starset: return
        self.starset = starset
        self.vecpos = []
        self.vecpos_indexed = []
        self.vecvec = []
        self.Nvstars_spec = 0
        # first do it for the complexes
        for star, indstar in zip(starset.stars[:starset.mixedstartindex],
                                 starset.starindexed[:starset.mixedstartindex]):
            pair0 = star[0]
            glist = []
            # Find group operations that leave state unchanged
            for gdumb in starset.pdbcontainer.G:
                pairnew = pair0.gop(starset.pdbcontainer, gdumb)[0]
                pairnew = pairnew - pairnew.R_s
                if pairnew == pair0:
                    glist.append(starset.pdbcontainer.G_crys[gdumb])  # Although appending gdumb itself also works
            # Find the intersected vector basis for these group operations
            vb = reduce(crystal.CombineVectorBasis, [crystal.VectorBasis(*g.eigen()) for g in glist])
            # Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)
            scale = 1. / np.sqrt(len(star))
            vlist = [v * scale for v in vlist]  # see equation 80 in the paper - (there is a typo, this is correct).
            Nvect = len(vlist)
            if Nvect > 0:
                if pair0.is_zero(self.starset.pdbcontainer):
                    self.Nvstars_spec += Nvect
                for v in vlist:
                    self.vecpos.append(star)
                    self.vecpos_indexed.append(indstar)
                    # implement a copy function like in case of vacancies
                    veclist = []
                    for pairI in star:
                        for gdumb in starset.pdbcontainer.G:
                            pairnew = pair0.gop(starset.pdbcontainer, gdumb)[0]
                            pairnew = pairnew - pairnew.R_s  # translate solute back to origin
                            # This is because the vectors associated with a state are translationally invariant.
                            # Wherever the solute is, if the relative position of the solute and the solvent is the
                            # same, the vector remains unchanged due to that group op.
                            # Remember that only the rotational part of the group op will act on the vector.
                            if pairnew == pairI:
                                veclist.append(starset.crys.g_direc(starset.pdbcontainer.G_crys[gdumb], v))
                                break
                    self.vecvec.append(veclist)

        self.Nvstars_pure = len(self.vecpos)

        # Now do it for the mixed dumbbells - all negative checks disappear
        for star, indstar in zip(starset.stars[starset.mixedstartindex:],
                                 starset.starindexed[starset.mixedstartindex:]):
            pair0 = star[0]
            glist = []
            # Find group operations that leave state unchanged
            for gdumb in starset.mdbcontainer.G:
                pairnew = pair0.gop(starset.mdbcontainer, gdumb, complex=False)
                pairnew = pairnew - pairnew.R_s  # again, only the rotation part matters.
                # what about dumbbell rotations? Does not matter - the state has to remain unchanged
                # Is this valid for origin states too? verify - because we have origin states.
                if pairnew == pair0:
                    glist.append(starset.mdbcontainer.G_crys[gdumb])
            # Find the intersected vector basis for these group operations
            vb = reduce(crystal.CombineVectorBasis, [crystal.VectorBasis(*g.eigen()) for g in glist])
            # Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)  # This also normalizes with respect to length of the vectors.
            scale = 1. / np.sqrt(len(star))
            vlist = [v * scale for v in vlist]
            Nvect = len(vlist)
            if Nvect > 0:  # why did I put this? Makes sense to expand only if Nvects >0, otherwise there is zero bias.
                # verify this
                for v in vlist:
                    self.vecpos.append(star)
                    self.vecpos_indexed.append(indstar)
                    veclist = []
                    for pairI in star:
                        for gdumb in starset.mdbcontainer.G:
                            pairnew = pair0.gop(starset.mdbcontainer, gdumb, complex=False)
                            pairnew = pairnew - pairnew.R_s  # translate solute back to origin
                            if pairnew == pairI:
                                veclist.append(starset.crys.g_direc(starset.mdbcontainer.G_crys[gdumb], v))
                                break
                    self.vecvec.append(veclist)

        self.Nvstars = len(self.vecpos)

        # build the vector star for the bare bare dumbbell state
        self.vecpos_bare = []
        self.vecvec_bare = []
        for star in starset.barePeriodicStars:
            db0 = star[0]
            glist = []
            for gdumb in starset.pdbcontainer.G:
                dbnew = db0.gop(starset.pdbcontainer, gdumb)[0]
                dbnew = dbnew - dbnew.R  # cancel out the translation
                if dbnew == db0:
                    glist.append(starset.pdbcontainer.G_crys[gdumb])
            vb = reduce(crystal.CombineVectorBasis, [crystal.VectorBasis(*g.eigen()) for g in glist])
            # Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)  # This also normalizes with respect to length of the vectors.
            scale = 1. / np.sqrt(len(star))
            vlist = [v * scale for v in vlist]
            Nvect = len(vlist)
            if Nvect > 0:
                for v in vlist:
                    veclist = []
                    self.vecpos_bare.append(star)
                    for st in star:
                        for gdumb in starset.pdbcontainer.G:
                            dbnew, flip = db0.gop(starset.pdbcontainer, gdumb)
                            dbnew = dbnew - dbnew.R
                            if dbnew == st:
                                veclist.append(starset.crys.g_direc(starset.pdbcontainer.G_crys[gdumb], v))
                                break
                    self.vecvec_bare.append(veclist)

        # index the states into the vector stars - required in OnsagerCalc

        # for st in self.starset.complexStates:
        #     indlist = []
        #     for IndOfStar, crStar in enumerate(self.vecpos[:self.Nvstars_pure]):
        #         for IndOfState, state in enumerate(crStar):
        #             if state == st:
        #                 indlist.append((IndOfStar, IndOfState))
        #     self.stateToVecStar_pure[st] = indlist

        self.stateToVecStar_pure = defaultdict(list)
        for IndofStar, crStar in enumerate(self.vecpos[:self.Nvstars_pure]):
            for IndofState, state in enumerate(crStar):
                self.stateToVecStar_pure[state].append((IndofStar, IndofState))

        self.stateToVecStar_mixed = defaultdict(list)
        for IndOfStar, crStar in enumerate(self.vecpos[self.Nvstars_pure:]):
            for IndOfState, state in enumerate(crStar):
                self.stateToVecStar_mixed[state].append((IndOfStar + self.Nvstars_pure, IndOfState))

        self.stateToVecStar_bare = {}
        if len(self.vecpos_bare) > 0:
            for IndOfStar, crStar in enumerate(self.vecpos_bare):
                for IndOfState, state in enumerate(crStar):
                    self.stateToVecStar_bare[state].append((IndOfStar, IndOfState))

        # We must produce two expansions. One for pure dumbbell states pointing to pure dumbbell state
        # and the other from mixed dumbbell states to mixed states.
        self.vwycktowyck_bare = np.zeros(len(self.vecpos_bare), dtype=int)
        for vstarind, vstar in enumerate(self.vecpos_bare):
            # get the index of the wyckoff set (symorlist) in which the representative state belongs
            wyckindex = self.starset.bareindexdict[vstar[0]][1]
            self.vwycktowyck_bare[vstarind] = wyckindex

        # Need an indexing from the vector stars to the crystal stars
        self.vstar2star = np.zeros(self.Nvstars, dtype=int)
        for vstindex, vst in enumerate(self.vecpos[:self.Nvstars_pure]):
            # get the crystal star of the representative state of the vector star
            starindex = self.starset.complexIndexdict[vst[0]][1]
            self.vstar2star[vstindex] = starindex

        for vstindex, vst in enumerate(self.vecpos[self.Nvstars_pure:]):
            # get the crystal star of the representative state of the vector star
            starindex = self.starset.mixedindexdict[vst[0]][1]
            # The starindex is already with respect to the total number of (pure+mixed) crystal stars - see stars.py.
            self.vstar2star[vstindex + self.Nvstars_pure] = starindex

    def genGFstarset(self):
        """
        Makes symmetrically grouped connections between the states in the starset, to be used as GFstarset for the pure
        and mixed state spaces.
        The connections must lie within the starset and must connect only those states that are connected by omega_0 or
        omega_2 jumps.
        The GFstarset is to be returned in the form of (i,j),dx. where the indices i and j correspond to the states in
        the iorlist
        """
        complexStates = self.starset.complexStates
        mixedstates = self.starset.mixedstates
        # Connect the states - major bottleneck
        connectset= set([])
        self.connect_ComplexPair = {}
        start = time.time()
        for i, st1 in enumerate(complexStates):
            for j, st2 in enumerate(complexStates[:i+1]):
                try:
                    s = st1 ^ st2
                except:
                    continue
                connectset.add(s)
                connectset.add(-s)
                # if i==j and not s==-s:
                #     raise ValueError("Same state connection producing different connector")
                self.connect_ComplexPair[(st1, st2)] = s
                self.connect_ComplexPair[(st2, st1)] = -s
        print("\tComplex connections creation time: {}".format(time.time() - start))

        # Now group the connections
        GFstarset_pure=[]
        GFPureStarInd = {}
        start = time.time()
        for s in connectset:
            if s in GFPureStarInd:
                continue
            connectlist = []
            for gdumb in self.starset.pdbcontainer.G:
                snew = s.gop(self.starset.pdbcontainer, gdumb, pure=True)
                # Bring the dumbbell of the initial state to the origin
                # snew = snew.shift() No need for shifting. Automatically done in gop function.
                if snew in GFPureStarInd:
                    continue

                if snew not in connectset:
                    raise TypeError("connector list is not closed under symmetry operations for the complex starset.{}"
                                    .format(snew))

                dx = disp(self.starset.pdbcontainer, snew.state1, snew.state2)
                ind1 = self.starset.pdbcontainer.db2ind(snew.state1)
                # db2ind does not care about which unit cell the dumbbell is at
                ind2 = self.starset.pdbcontainer.db2ind(snew.state2)
                tup = ((ind1, ind2), dx.copy())
                connectlist.append(tup)
                GFPureStarInd[snew] = len(GFstarset_pure)
            GFstarset_pure.append(connectlist)
        print("\tComplex connections symmetry grouping time: {}".format(time.time() - start))

        GFstarset_mixed = []
        GFMixedStarInd = {}
        connectset_mixed = set([])
        self.connect_MixedPair = {}
        # For the mixed dumbbells, we need only the states in the initial unit cell.
        for i, state1 in enumerate(mixedstates):
            for j, state2 in enumerate(mixedstates[:i+1]):
                s = connector(state1.db, state2.db)
                connectset_mixed.add(s)
                connectset_mixed.add(-s)
                self.connect_MixedPair[(state1, state2)] = s
                self.connect_MixedPair[(state2, state1)] = -s

        # Now, group them symmetrically
        for s in connectset_mixed:
            if s in GFMixedStarInd:
                continue
            connectlist =[]
            for gdumb in self.starset.mdbcontainer.G:
                
                snew = s.gop(self.starset.mdbcontainer, gdumb, pure=False)
                # snew = connector(snew.state1 - snew.state1.R, snew.state2 - snew.state2.R)
                # snew = snew.shift()
                if snew not in connectset_mixed:

                    if np.allclose(snew.state1.R, np.zeros(3)) and np.allclose(snew.state2.R, np.zeros(3)):
                        raise ValueError("origin cell connection not in mixed connectset")

                    # If a group operation takes a connection's end point outside the origin unit cell,
                    # we can ignore it, since the problem dictates that both initial and final states in a connection
                    # lie inside the origin unit cell.
                    # The sum over the unit cells is implicit in the g2 calculation - see makeGF
                    continue
                if snew in GFMixedStarInd:
                    continue
                dx = disp(self.starset.mdbcontainer, snew.state1, snew.state2)
                ind1 = self.starset.mdbcontainer.db2ind(snew.state1)
                # db2ind does not care about which unit cell the dumbbell is at
                ind2 = self.starset.mdbcontainer.db2ind(snew.state2)
                tup = ((ind1, ind2), dx.copy())
                connectlist.append(tup)
                # slist.append(snew)
                GFMixedStarInd[snew] = len(GFstarset_mixed)

            GFstarset_mixed.append(connectlist)
            # GFstarset_mixed_snewlist.append(slist)
        print("No. of pure dumbbell connections: {}".format(len(connectset)))
        print("No. of mixed dumbbell connections: {}".format(len(connectset_mixed)))
        return GFstarset_pure, GFPureStarInd, GFstarset_mixed, GFMixedStarInd

    def GFexpansion(self):
        """
        carries out the expansion of the Green's function in the basis of the vector stars.
        """
        print("building GF starsets")
        start = time.time()
        GFstarset_pure, GFPureStarInd, GFstarset_mixed, GFMixedStarInd = self.genGFstarset()
        print("GF star sets built: {}".format(time.time() - start))

        Nvstars_pure = self.Nvstars_pure
        Nvstars_mixed = self.Nvstars - self.Nvstars_pure
        GFexpansion_mixed = np.zeros((Nvstars_mixed, Nvstars_mixed, len(GFstarset_mixed)))


        GFexpansion_pure = np.zeros((Nvstars_pure, Nvstars_pure, len(GFstarset_pure)))
        start = time.time()
        for ((st1, st2), s) in self.connect_ComplexPair.items():
            # get the vector stars in which the initial state belongs
            i = self.stateToVecStar_pure[st1]
            # get the vector stars in which the final state belongs
            j = self.stateToVecStar_pure[st2]
            k = GFPureStarInd[s]
            for (indOfStar_i, indOfState_i) in i:
                for (indOfStar_j, indOfState_j) in j:
                    GFexpansion_pure[indOfStar_i, indOfStar_j, k] += \
                        np.dot(self.vecvec[indOfStar_i][indOfState_i], self.vecvec[indOfStar_j][indOfState_j])


        print("Built Complex GF expansions: {}".format(time.time() - start))

        start = time.time()
        for ((st1, st2), s) in self.connect_MixedPair.items():
            i = self.stateToVecStar_mixed[st1]
            j = self.stateToVecStar_mixed[st2]
            k = GFMixedStarInd[s]
            for (indOfStar_i, indOfState_i) in i:
                for (indOfStar_j, indOfState_j) in j:
                    GFexpansion_mixed[indOfStar_i-self.Nvstars_pure, indOfStar_j-self.Nvstars_pure, k] += \
                        np.dot(self.vecvec[indOfStar_i][indOfState_i], self.vecvec[indOfStar_j][indOfState_j])

        print("Built Mixed GF expansions: {}".format(time.time() - start))

        # symmetrize
        for i in range(Nvstars_pure):
            for j in range(0, i):
                GFexpansion_pure[i, j, :] = GFexpansion_pure[j, i, :]

        for i in range(Nvstars_mixed):
            for j in range(0, i):
                GFexpansion_mixed[i, j, :] = GFexpansion_mixed[j, i, :]

        # print("Built GF expansions: {}".format(time.time() - start))

        return (GFstarset_pure, GFPureStarInd, zeroclean(GFexpansion_pure)),\
               (GFstarset_mixed, GFMixedStarInd, zeroclean(GFexpansion_mixed))

    # See group meeting update slides of sept 10th to see how this works.
    def biasexpansion(self, jumpnetwork_omega1, jumpnetwork_omega2, jumptype, jumpnetwork_omega34):
        """
        Returns an expansion of the bias vector in terms of the displacements produced by jumps.
        Parameters:
            jumpnetwork_omega* - the jumpnetwork for the "*" kind of jumps (1,2,3 or 4)
            jumptype - the omega_0 jump type that gives rise to a omega_1 jump type (see jumpnetwork_omega1 function
            in stars.py module)
        Returns:
            bias0, bias1, bias2, bias4 and bias3 expansions, one each for solute and solvent
            Note - bias0 for solute makes no sense, so we return only for solvent.
        """
        z = np.zeros(3, dtype=float)

        biasBareExpansion = np.zeros((len(self.vecpos_bare), len(self.starset.jnet0)))
        # Expansion of pure dumbbell initial state bias vectors and complex state bias vectors
        bias0expansion = np.zeros((self.Nvstars_pure, len(self.starset.jumpindices)))
        bias1expansion_solvent = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega1)))
        bias1expansion_solute = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega1)))

        bias4expansion_solvent = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega34)))
        bias4expansion_solute = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega34)))

        # Expansion of mixed dumbbell initial state bias vectors.
        bias2expansion_solvent = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega2)))
        bias2expansion_solute = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega2)))

        bias3expansion_solvent = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))
        bias3expansion_solute = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))

        # First, let's build the periodic bias expansions
        for i, star, vectors in zip(itertools.count(), self.vecpos_bare, self.vecvec_bare):
            for k, jumplist in zip(itertools.count(), self.starset.jnet0):
                for j in jumplist:
                    IS = j.state1
                    if star[0] == IS:
                        dx = disp(self.starset.pdbcontainer, j.state1, j.state2)
                        geom_bias_solvent = np.dot(vectors[0], dx) * len(star)
                        biasBareExpansion[i, k] += geom_bias_solvent

        for i, purestar, vectors in zip(itertools.count(), self.vecpos[:self.Nvstars_pure],
                                        self.vecvec[:self.Nvstars_pure]):
            # iterates over the rows of the matrix
            # First construct bias1expansion and bias0expansion
            # This contains the expansion of omega_0 jumps and omega_1 type jumps
            # See slides of Sept. 10 for diagram.
            # omega_0 : pure -> pure
            # omega_1 : complex -> complex
            for k, jumplist, jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
                # iterates over the columns of the matrix
                for j in jumplist:
                    IS = j.state1
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if purestar[0] == IS:
                        # sees if there is a jump of the kth type with purestar[0] as the initial state.
                        dx = disp(self.starset.pdbcontainer, j.state1, j.state2)
                        dx_solute = z
                        dx_solvent = dx.copy()  # just for clarity that the solvent mass transport is dx itself.

                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(
                            purestar)  # should this be square root? check with tests.
                        geom_bias_solute = np.dot(vectors[0], dx_solute) * len(purestar)

                        bias1expansion_solvent[
                            i, k] += geom_bias_solvent  # this is contribution of kth_type of omega_1 jumps, to the bias
                        bias1expansion_solute[i, k] += geom_bias_solute
                        # vector along v_i
                        # so to find the total bias along v_i due to omega_1 jumps, we sum over k
                        bias0expansion[i, jt] += geom_bias_solvent  # These are the contributions of the omega_0 jumps
                        # to the bias vector along v_i, for bare dumbbells
                        # so to find the total bias along v_i, we sum over k.

            # Next, omega_4: complex -> mixed
            for k, jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist[::2]:  # Start from the first element, skip every other
                    IS = j.state1
                    if not j.state2.is_zero(self.starset.mdbcontainer):
                        # check if initial state is mixed dumbbell -> then skip - it's omega_3
                        raise TypeError ("final state not origin in mixed dbcontainer for omega4")
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if purestar[0] == IS:
                        dx = disp4(self.starset.pdbcontainer, self.starset.mdbcontainer, j.state1, j.state2)
                        dx_solute = np.zeros(3)  # self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                        dx_solvent = dx  # - self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                        geom_bias_solute = np.dot(vectors[0], dx_solute) * len(purestar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(purestar)
                        bias4expansion_solute[i, k] += geom_bias_solute
                        # this is contribution of omega_4 jumps, to the bias
                        bias4expansion_solvent[i, k] += geom_bias_solvent
                        # vector along v_i
                        # So, to find the total bias along v_i due to omega_4 jumps, we sum over k.

        # Now, construct the bias2expansion and bias3expansion
        for i, mixedstar, vectors in zip(itertools.count(), self.vecpos[self.Nvstars_pure:],
                                         self.vecvec[self.Nvstars_pure:]):
            # First construct bias2expansion
            # omega_2 : mixed -> mixed
            for k, jumplist in zip(itertools.count(), jumpnetwork_omega2):
                for j in jumplist:
                    IS = j.state1
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if mixedstar[0] == IS:
                        dx = disp(self.starset.mdbcontainer, j.state1, j.state2)
                        dx_solute = dx  # + self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. - \
                                    # self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        dx_solvent = dx  #- self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. + \
                                     #self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        geom_bias_solute = np.dot(vectors[0], dx_solute) * len(mixedstar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(mixedstar)
                        bias2expansion_solute[i, k] += geom_bias_solute
                        bias2expansion_solvent[i, k] += geom_bias_solvent

            # Next, omega_3: mixed -> complex
            for k, jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist[1::2]:  # start from the second element, skip every other
                    if not j.state1.is_zero(self.starset.mdbcontainer):
                        # check if initial state is not a mixed state -> skip if not mixed
                        print(self.starset.mdbcontainer.iorlist)
                        print(j.state1)
                        raise TypeError("initial state not origin in mdbcontainer")
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if mixedstar[0] == j.state1:
                        try:
                            dx = -disp4(self.starset.pdbcontainer, self.starset.mdbcontainer, j.state2, j.state1)
                        except IndexError:
                            print(len(self.starset.pdbcontainer.iorlist), len(self.starset.mdbcontainer.iorlist))
                            print(j.state2.db.iorind, j.state1.db.iorind)
                            raise IndexError("list index out of range")
                        dx_solute = np.zeros(3)  # -self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        dx_solvent = dx  # + self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        geom_bias_solute = np.dot(vectors[0], dx_solute) * len(mixedstar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(mixedstar)
                        bias3expansion_solute[i, k] += geom_bias_solute
                        bias3expansion_solvent[i, k] += geom_bias_solvent

        if len(self.vecpos_bare) == 0:
            return zeroclean(bias0expansion), (zeroclean(bias1expansion_solute), zeroclean(bias1expansion_solvent)), \
                   (zeroclean(bias2expansion_solute), zeroclean(bias2expansion_solvent)), \
                   (zeroclean(bias3expansion_solute), zeroclean(bias3expansion_solvent)), \
                   (zeroclean(bias4expansion_solute), zeroclean(bias4expansion_solvent)), biasBareExpansion
        else:
            return zeroclean(bias0expansion), (zeroclean(bias1expansion_solute), zeroclean(bias1expansion_solvent)), \
                   (zeroclean(bias2expansion_solute), zeroclean(bias2expansion_solvent)), \
                   (zeroclean(bias3expansion_solute), zeroclean(bias3expansion_solvent)), \
                   (zeroclean(bias4expansion_solute), zeroclean(bias4expansion_solvent)), zeroclean(biasBareExpansion)

    def rateexpansion(self, jumpnetwork_omega1, jumptype, jumpnetwork_omega34):
        """
        Implements expansion of the jump rates in terms of the basis function of the vector stars.
        (Note to self) - Refer to earlier notes for details.
        """
        # See my slides of Sept. 10 for diagram
        rate0expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(self.starset.jnet0)))
        rate1expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(jumpnetwork_omega1)))
        rate0escape = np.zeros((self.Nvstars_pure, len(self.starset.jumpindices)))
        rate1escape = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega1)))

        # First, we do the rate1 and rate0 expansions
        for k, jumplist, jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
            for jmp in jumplist:
                # Get the vector star indices for the initial and final states of the jumps
                indlist1 = self.stateToVecStar_pure[jmp.state1]
                indlist2 = self.stateToVecStar_pure[jmp.state2]
                # indlists contain tuples of the form (IndOfVstar, IndInVstar)
                for tup1 in indlist1:
                    # print(tup1)
                    rate0escape[tup1[0], jt] -= np.dot(self.vecvec[tup1[0]][tup1[1]], self.vecvec[tup1[0]][tup1[1]])
                    rate1escape[tup1[0], k] -= np.dot(self.vecvec[tup1[0]][tup1[1]], self.vecvec[tup1[0]][tup1[1]])
                    for tup2 in indlist2:
                        rate0expansion[tup1[0], tup2[0], jt] += np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                       self.vecvec[tup2[0]][tup2[1]])

                        rate1expansion[tup1[0], tup2[0], k] += np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                      self.vecvec[tup2[0]][tup2[1]])

        # Next, we expand the omega3 an omega4 rates
        rate4expansion = np.zeros((self.Nvstars_pure, self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))
        rate3expansion = np.zeros((self.Nvstars - self.Nvstars_pure, self.Nvstars_pure, len(jumpnetwork_omega34)))
        rate3escape = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))
        rate4escape = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega34)))

        for k, jumplist in enumerate(jumpnetwork_omega34):
            for jmp in jumplist[::2]:  # iterate only through the omega4 jumps, the negatives are omega3

                indlist1 = self.stateToVecStar_pure[jmp.state1]  # The initial state is a complex in omega4
                indlist2 = self.stateToVecStar_mixed[jmp.state2]  # The final state is a mixed dumbbell in omega4

                for tup1 in indlist1:
                    rate4escape[tup1[0], k] -= np.dot(self.vecvec[tup1[0]][tup1[1]], self.vecvec[tup1[0]][tup1[1]])
                for tup2 in indlist2:
                    rate3escape[tup2[0] - self.Nvstars_pure, k] -= np.dot(self.vecvec[tup2[0]][tup2[1]],
                                                                          self.vecvec[tup2[0]][tup2[1]])

                for tup1 in indlist1:
                    for tup2 in indlist2:

                        rate4expansion[tup1[0], tup2[0] - self.Nvstars_pure, k] += np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                                          self.vecvec[tup2[0]][tup2[1]])

                        rate3expansion[tup2[0] - self.Nvstars_pure, tup1[0], k] += np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                                          self.vecvec[tup2[0]][tup2[1]])

        # Next, we expand omega2
        rate2expansion = np.zeros((self.Nvstars - self.Nvstars_pure, self.Nvstars - self.Nvstars_pure,
                                   len(self.starset.jnet2)))
        rate2escape = np.zeros((self.Nvstars - self.Nvstars_pure, len(self.starset.jnet2)))

        for k, jumplist in zip(itertools.count(), self.starset.jnet2):
            for jmp in jumplist:

                indlist1 = self.stateToVecStar_mixed[jmp.state1]
                indlist2 = self.stateToVecStar_mixed[jmp.state2 - jmp.state2.R_s]

                for tup1 in indlist1:
                    rate2escape[tup1[0] - self.Nvstars_pure, k] -= np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                          self.vecvec[tup1[0]][tup1[1]])
                    for tup2 in indlist2:
                        rate2expansion[tup1[0] - self.Nvstars_pure, tup2[0] - self.Nvstars_pure, k] +=\
                            np.dot(self.vecvec[tup1[0]][tup1[1]], self.vecvec[tup2[0]][tup2[1]])

        return (zeroclean(rate0expansion), zeroclean(rate0escape)),\
               (zeroclean(rate1expansion), zeroclean(rate1escape)),\
               (zeroclean(rate2expansion), zeroclean(rate2escape)),\
               (zeroclean(rate3expansion), zeroclean(rate3escape)),\
               (zeroclean(rate4expansion), zeroclean(rate4escape))

    def outer(self):
        """
        computes the outer product tensor to perform 'bias *outer* gamma', i.e., the uncorrelated part in the vector
        star basis.
        :return: outerprod, 3x3xNvstarsxNvstars outer product tensor.
        """
        # print("Building outer product tensor")
        outerprod = np.zeros((3, 3, self.Nvstars, self.Nvstars))
        # start = time.time()
        # for i in range(self.Nvstars_pure):
        #     for j in range(self.Nvstars_pure):
        #         for si, vi in zip(self.vecpos[i], self.vecvec[i]):
        #             for sj, vj in zip(self.vecpos[j], self.vecvec[j]):
        #                 if si == sj:
        #                     outerprod_old[:, :, i, j] += np.outer(vi, vj)
        # print("\tOld method: {}".format(time.time()-start))

        # start = time.time()
        for st in self.starset.complexStates:
            vecStarList = self.stateToVecStar_pure[st]
            for (indStar1, indState1) in vecStarList:
                for (indStar2, indState2) in vecStarList:
                    outerprod[:, :, indStar1, indStar2] += np.outer(self.vecvec[indStar1][indState1],
                                                                    self.vecvec[indStar2][indState2])
        # print("\tNew method: {}".format(time.time() - start))
        # print(np.allclose(outerprod, outerprod_old))

        # There should be no non-zero outer product tensors between the pure and mixed dumbbells.

        for st in self.starset.mixedstates:
            indlist = self.stateToVecStar_mixed[st]
            for (IndofStar1, IndofState1) in indlist:
                for (IndofStar2, IndofState2) in indlist:
                    outerprod[:, :, IndofStar1, IndofStar2] += np.outer(self.vecvec[IndofStar1][IndofState1],
                                                                      self.vecvec[IndofStar2][IndofState2])

        return zeroclean(outerprod)
