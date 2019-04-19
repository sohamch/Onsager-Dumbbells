from typing import Any, Union

import numpy as np
import onsager.crystal as crystal
from onsager.crystalStars import zeroclean, VectorStarSet, PairState
from states import *
from representations import *
from stars import *
from functools import reduce
import itertools


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

        # Now do it for the mixed dumbbells - all negative checks dissappear
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
            vlist = starset.crys.vectlist(vb)  # This also nomalizes with respect to length of the vectors.
            scale = 1. / np.sqrt(len(star))
            vlist = [v * scale for v in vlist]
            Nvect = len(vlist)
            if Nvect > 0:
                for v in vlist:
                    veclist = []
                    self.vecpos_bare.append(star)
                    for st in star:
                        for gdumb in starset.pdbcontainer.G:
                            dbnew = db0.gop(starset.pdbcontainer, gdumb)[0]
                            dbnew = dbnew - dbnew.R
                            if dbnew == st:
                                veclist.append(starset.crys.g_direc(starset.pdbcontainer.G_crys[gdumb], v))
                                break
                    self.vecvec_bare.append(veclist)

        # index the states into the vector stars - required in OnsagerCalc
        self.stateToVecStar_pure = {}
        for st in self.starset.complexStates:
            indlist = []
            for IndOfStar, crStar in enumerate(self.vecpos[:self.Nvstars_pure]):
                for IndOfState, state in enumerate(crStar):
                    if state == st:
                        indlist.append((IndOfStar, IndOfState))
            self.stateToVecStar_pure[st] = indlist

        self.stateToVecStar_mixed = {}
        for st in self.starset.mixedstates:
            indlist = []
            for IndOfStar, crStar in enumerate(self.vecpos[self.Nvstars_pure:]):
                for IndOfState, state in enumerate(crStar):
                    if state == st:
                        indlist.append((IndOfStar + self.Nvstars_pure, IndOfState))
            self.stateToVecStar_mixed[st] = indlist

        self.stateToVecStar_bare = {}
        if len(self.vecpos_bare) > 0:
            for st in self.starset.bareStates:
                indlist = []
                for IndOfStar, crStar in enumerate(self.vecpos_bare):
                    for IndOfState, state in enumerate(crStar):
                        if state == st:
                            indlist.append((IndOfStar, IndOfState))
                self.stateToVecStar_bare[st] = indlist

        # We must produce two expansions. One for pure dumbbell states pointing to pure dumbbell state
        # and the other from mixed dumbbell states to mixed states.

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
        # Connect the states
        connectset= set([])
        for st1 in complexStates:
            for st2 in complexStates:
                try:
                    s = st1 ^ st2
                except:
                    continue
                # Bring the initial dumbbell to the origin unit cell
                s = connector(s.state1 - s.state1.R, s.state2 - s.state1.R)
                if not s in connectset:
                    connectset.append(s)

        # Now group the connections
        GFstarset_pure=[]
        GFPureStarInd = {}
        for s in connectset:
            if s in GFPureStarInd:
                continue
            connectlist = []
            for gdumb in self.starset.pdbcontainer.G:
                snew = s.gop(self.starset.pdbcontainer, gdumb, pure=True)
                # Bring the dumbbell of the initial state to the origin
                snew.shift()

                if snew not in connectset:
                    raise TypeError("connector list is not closed under symmetry operations for the complex starset.")
                if snew in GFPureStarInd:
                    continue

                dx = disp(self.starset.pdbcontainer, db1new, db2new)
                ind1 = self.starset.pdbcontainer.db2ind(snew.state1)
                # db2ind does not care about which unit cell the dumbbell is at
                ind2 = self.starset.pdbcontainer.db2ind(snew.state2)
                tup = ((ind1, ind2), dx.copy())
                connectlist.append(tup)
                GFPureStarInd[snew] = len(GFstarset_pure)
            GFstarset_pure.append(connectlist)

        # for st1 in mixedstates:
        #     for st2 in mixedstates:
        #         # The mixed states are origin states - must make so after gop
        #         s = connector(st1.db, st2.db)
        #         ind1 = self.starset.mdbcontainer.iorindex.get(s.state1)
        #         ind2 = self.starset.mdbcontainer.iorindex.get(s.state2)
        #         if ind1 == None or ind2 == None:
        #             raise KeyError("dumbbell not found in iorlist")
        #         # dR = s.state2.R - s.state1.R
        #         if s in GFMixedStarInd:
        #             continue
        #         connectlist = []
        #         for g in self.starset.crys.G:
        #             snew = s.gop(self.starset.crys, self.starset.chem, g)
        #             # Bring back to 0th unit cell
        #             snew = connector(snew.state1 - snew.state1.R, snew.state2 - snew.state2.R)
        #             if snew in GFMixedStarInd:
        #                 continue
        #             # dR = snew.state2.R - snew.state1.R
        #             dx = disp(self.starset.crys, self.starset.chem, snew.state1, snew.state2)
        #             ind1 = self.starset.mdbcontainer.iorindex.get(snew.state1)
        #             ind2 = self.starset.mdbcontainer.iorindex.get(snew.state2)
        #             if ind1 == None or ind2 == None:
        #                 raise KeyError("dumbbell not found in iorlist")
        #             tup = ((ind1, ind2), dx)
        #             connectlist.append(tup)
        #             GFMixedStarInd[snew] = len(GFstarset_mixed)
        #         GFstarset_mixed.append(connectlist)

        # For the mixed GF starset, the mixed jumpnetwork is the one that creates the GF starset
        # Since the mixed starsets are periodic, their states are at the 0th unit cell. So, unlike the compleStates
        # there are no omega2 jumps that connect two states in the mixed starset. Rather, the omega2 jumpnetwork
        # and the mixed GF starset have to be created separately.
        GFstarset_mixed = []
        GFMixedStarInd = {}
        connectset_mixed = set([])
        for jlist in self.starset.jnet2:
            for jmp in jlist:
                db1 = jmp.state1.db
                db2 = jmp.state2.db
                s = connector(db1 - db1.R, db2 - db1.R)
                connectset.add(s)
        # Now add in the diagonal elements.
        for state in self.starset.mixedstates:
            s = connector(state.db, state.db)

        # Now, group them symmetrically
        for s in connectset_mixed:
            if s in GFMixedStarInd:
                continue
            connectlist =[]
            for gdumb in self.starset.mdbcontainer.G:
                snew = s.gop(self.starset.mdbcontainer, gdumb, pure=False)
                snew.shift()

                if snew not in connectset_mixed:
                    raise ValueError("the connector list for mixed dumbbells is not closed under symmetry")
                if snew in GFMixedStarInd:
                    continue

                dx = disp(self.starset.mdbcontainer, snew.state1, snew.state2)
                ind1 = self.starset.mdbcontainer.db2ind(snew.state1)
                # db2ind does not care about which unit cell the dumbbell is at
                ind2 = self.starset.mdbcontainer.db2ind(snew.state2)
                tup = ((ind1, ind2), dx.copy())
                connectlist.append(tup)
                GFMixedStarInd[snew] = len(GFstarset_mixed)
            GFstarset_mixed.append(connectlist)


        return GFstarset_pure, GFPureStarInd, GFstarset_mixed, GFMixedStarInd

    def GFexpansion(self):
        """
        carries out the expansion of the Green's function in the basis of the vector stars.
        """
        GFstarset_pure, GFPureStarInd, GFstarset_mixed, GFMixedStarInd = self.genGFstarset()

        Nvstars_pure = self.Nvstars_pure
        Nvstars_mixed = self.Nvstars - self.Nvstars_pure
        GFexpansion_pure = np.zeros((Nvstars_pure, Nvstars_pure, len(GFstarset_pure)))
        GFexpansion_mixed = np.zeros((Nvstars_mixed, Nvstars_mixed, len(GFstarset_mixed)))

        # build up the pure GFexpansion
        for i in range(Nvstars_pure):
            for si, vi in zip(self.vecpos[i], self.vecvec[i]):
                for j in range(Nvstars_pure):
                    for sj, vj in zip(self.vecpos[j], self.vecvec[j]):
                        try:
                            ds = si ^ sj
                        except:
                            continue
                        # Bring the initial dumbbell back to the origin
                        R_ref = ds.state1.R.copy()
                        ds = connector(ds.state1 - R_ref, ds.state2 - R_ref)
                        dx = disp(self.starset.pdbcontainer, ds.state1, ds.state2)
                        ind1 = self.starset.pdbcontainer.db2ind(ds.state1)
                        ind2 = self.starset.pdbcontainer.db2ind(ds.state2 - ds.state2.R)
                        if ind1 == None or ind2 == None:
                            raise KeyError("enpoint subtraction within starset not found in iorlist")
                        # k = getstar(((ind1,ind2),dx),GFstarset_pure)
                        k = GFPureStarInd[ds]
                        if k is None:
                            raise ArithmeticError(
                                "complex GF starset not big enough to accomodate state pair {}".format(tup))
                        GFexpansion_pure[i, j, k] += np.dot(vi, vj)

        # Build up the mixed GF expansion
        # In this case too, we need to build up from the jump network, and include the origin states later on.

        # First, we build up the terms from the jump network
        for jlist in self.starset.jnet2:
            for jmp in jlist:
                try:
                    ds = jmp.state1.db ^ jmp.state2.db
                    ds.shift()
                except:
                    raise ValueError("jump from the mixed jumps list cannot be converted to connector")

                # Now get the vector stars for the states
                si = jmp.state1 - jmp.state1.R_s
                sj = jmp.state2 - jmp.state2.R_s

                viList = self.stateToVecStar_mixed[si]  # (IndOfStar, IndOfState) format
                vjList = self.stateToVecStar_mixed[si]  # (IndOfStar, IndOfState) format
                k = GFMixedStarInd[ds]
                if k is None:
                    raise ArithmeticError(
                        "mixed GF starset not big enough to accomodate state pair {}".format((si, sj)))

                for i, vi in [(tup[0], self.vecpos[tup[0]][tup[1]]) for tup in viList]:
                    for j, vj in [(tup[0], self.vecpos[tup[0]][tup[1]]) for tup in viList]:
                        GFexpansion_mixed[i, j, k] += np.dot(vi, vj)





        # for i in range(Nvstars_mixed):
        #     for si, vi in zip(self.vecpos[Nvstars_pure + i], self.vecvec[Nvstars_pure + i]):
        #         for j in range(Nvstars_mixed):
        #             for sj, vj in zip(self.vecpos[Nvstars_pure + j], self.vecvec[Nvstars_pure + j]):
        #                 # Both si and sj are origin states
        #                 ds = connector(si.db, sj.db)
        #                 # except:
        #                 #     continue
        #                 dx = disp(self.starset.mdbcontainer, ds.state1, ds.state2)
        #                 ind1 = self.starset.mdbcontainer.iorindex.get(ds.state1)
        #                 ind2 = self.starset.mdbcontainer.iorindex.get(ds.state2)
        #                 if ind1 == None or ind2 == None:
        #                     raise KeyError("endpoint subtraction within starset not found in iorlist")
        #                 # k = getstar(((ind1,ind2),dx),GFstarset_mixed)
        #                 k = GFMixedStarInd[ds]
        #                 if k is None:
        #                     raise ArithmeticError(
        #                         "mixed GF starset not big enough to accomodate state pair {}".format(tup))
        #                 GFexpansion_mixed[i, j, k] += np.dot(vi, vj)

        # symmetrize
        for i in range(Nvstars_pure):
            for j in range(0, i):
                GFexpansion_pure[i, j, :] = GFexpansion_pure[j, i, :]

        for i in range(Nvstars_mixed):
            for j in range(0, i):
                GFexpansion_mixed[i, j, :] = GFexpansion_mixed[j, i, :]

        return (GFstarset_pure, GFPureStarInd, zeroclean(GFexpansion_pure)), (
            GFstarset_mixed, GFMixedStarInd, zeroclean(GFexpansion_mixed))

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
                for j in jumplist:
                    IS = j.state1
                    if IS.is_zero(self.starset.mdbcontainer):
                        # check if initial state is mixed dumbbell -> then skip - it's omega_3
                        continue
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if purestar[0] == IS:
                        dx = disp4(self.starset.pdbcontainer, self.starset.mdbcontainer, j.state1, j.state2)
                        dx_solute = self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                        dx_solvent = dx - self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
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
                        dx_solute = dx + self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. - \
                                    self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        dx_solvent = dx - self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. + \
                                     self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        geom_bias_solute = np.dot(vectors[0], dx_solute) * len(mixedstar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(mixedstar)
                        bias2expansion_solute[i, k] += geom_bias_solute
                        bias2expansion_solvent[i, k] += geom_bias_solvent
            # Next, omega_3: mixed -> complex
            for k, jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist:
                    IS = j.state1
                    if not IS.is_zero(self.starset.mdbcontainer):
                        # check if initial state is not a mixed state -> skip if not mixed
                        continue
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if mixedstar[0] == IS:
                        try:
                            dx = -disp4(self.starset.pdbcontainer, self.starset.mdbcontainer, j.state2, j.state1)
                        except IndexError:
                            print(len(self.starset.pdbcontainer), len(self.starset.mdbcontainer))
                            print(len(j.state2.db.iorind), len(j.state1.db.iorind))
                            raise IndexError("list index out of range")
                        dx_solute = -self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        dx_solvent = dx + self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
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
        rate0expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(self.starset.jumpnetwork_omega0)))
        rate1expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(jumpnetwork_omega1)))
        rate0escape = np.zeros((self.Nvstars_pure, len(self.starset.jumpindices)))
        rate1escape = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega1)))
        # First, we do the rate1 and rate0 expansions
        for k, jumplist, jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
            for jmp in jumplist:
                for i in range(self.Nvstars_pure):  # The first inner sum
                    for chi_i, vi in zip(self.vecpos[i], self.vecvec[i]):
                        if chi_i == jmp.state1:  # This is the delta functions of chi_0
                            rate0escape[i, jt] -= np.dot(vi, vi)
                            rate1escape[i, k] -= np.dot(vi, vi)
                            for j in range(self.Nvstars_pure):  # The second inner sum
                                for chi_j, vj in zip(self.vecpos[j], self.vecvec[j]):
                                    if chi_j == jmp.state2:  # this is the delta function of chi_1
                                        rate1expansion[i, j, k] += np.dot(vi, vj)
                                        rate0expansion[i, j, jt] += np.dot(vi, vj)

        # The initial states are complexes, the final states are mixed and there are as many symmetric jumps as in
        # jumpnetwork_omega34
        rate4expansion = np.zeros((self.Nvstars_pure, self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))
        rate3expansion = np.zeros((self.Nvstars - self.Nvstars_pure, self.Nvstars_pure, len(jumpnetwork_omega34)))
        # The initial states are mixed, the final states are complex except origin states and there are as many
        # symmetric jumps as in jumpnetwork_omega34
        rate3escape = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))
        rate4escape = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega34)))

        # We implement the matrix for omega4 and note that omega3 is the negative jump of omega4
        # This is because the order of the sum over stars in the rate expansion does not matter (see Sept. 30 slides).
        for k, jumplist in zip(itertools.count(), jumpnetwork_omega34):
            for jmp in jumplist:
                if jmp.state1.is_zero():  # the initial state must be a complex
                    # the negative of this jump is an omega_3 jump anyway
                    continue
                for i in range(self.Nvstars_pure):  # iterate over complex states - the first inner sum
                    for chi_i, vi in zip(self.vecpos[i], self.vecvec[i]):
                        # Go through the initial pure states
                        if chi_i == jmp.state1:
                            rate4escape[i, k] -= np.dot(vi, vi)
                            for j in range(self.Nvstars_pure,
                                           self.Nvstars):  # iterate over mixed states - the second inner sum
                                for chi_j, vj in zip(self.vecpos[j], self.vecvec[j]):
                                    # Go through the final complex states
                                    if chi_j == jmp.state2:
                                        rate3escape[j - self.Nvstars_pure, k] -= np.dot(vj, vj)
                                        rate4expansion[i, j - self.Nvstars_pure, k] += np.dot(vi, vj)
                                        rate3expansion[j - self.Nvstars_pure, i, k] += np.dot(vj, vi)
                                        # The jump type remains the same because they have the same transition state

        return (zeroclean(rate0expansion), zeroclean(rate0escape)), (zeroclean(rate1expansion), zeroclean(rate1escape)), \
               (zeroclean(rate3expansion), zeroclean(rate3escape)), (zeroclean(rate4expansion), zeroclean(rate4escape))

    def outer(self):
        outerprods = np.zeros((3, 3, self.Nvstars, self.Nvstars))
        for i in range(self.Nvstars):
            for j in range(self.Nvstars):
                for st_i, v_i in zip(self.vecpos[i], self.vecvec[i]):
                    for st_j, v_j in zip(self.vecpos[j], self.vecvec[j]):
                        if st_i == st_j:
                            outerprods[:, :, i, j] = np.outer(v_i, v_j)
        return zeroclean(outerprods)
