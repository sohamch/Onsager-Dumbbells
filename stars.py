import numpy as np
import onsager.crystal as crystal
# from jumpnet3 import *
from states import *
import itertools
from collections import defaultdict
from representations import *


class StarSet(object):
    """
    class to form the crystal stars, with shells indicated by the number of jumps.
    Almost exactly similar to CrystalStars.StarSet except now includes orientations.
    The minimum shell (Nshells=0) is composed of dumbbells situated atleast one jump away.
    """

    def __init__(self, pdbcontainer, mdbcontainer, jnetwrk0, jnetwrk2,
                 Nshells=None):
        """
        Parameters:
        pdbcontainer,mdbcontainer:
            -containers containing the pure and mixed dumbbell information respectively
        jnet0,jnet2 - jumpnetworks in pure and mixed dumbbell spaces respectively.
            Note - must send in both as pair states and indexed.
        Nshells - number of thermodynamic shells. Minimum - one jump away - corresponds to Nshells=0

        Index objects contained in the starset
        All the indexing are done into the following four lists
        ->pdbcontainer.iorlist, mdbcontainer.iorlist - the list of (site, orientation) tuples allowed for pure and mixed dumbbells respectively.
        ->complexStates,mixedstates - the list SdPair objects, containing the complex and mixed dumbbells that make up the starset

        --starindexed -> gives the indices to the states list of the states stored in the starset
        --complexIndex, mixedindex -> tells us which star (via it's index in the pure(or mixed)states list) a state belongs to.
        --complexIndexdict, mixedindexdict -> tell us given a pair state, what is its index in the states list and the starset, as elements of a 2-tuple.
        --complexStatesToContainer, mixedStatesToContainer -> tell us the index of the (i,o) of a dumbbell in a SdPair in pure/mixedstates in the
        respective iorlists.

        """
        # check that we have the same crystal structures for pdbcontainer and mdbcontainer
        if not np.allclose(pdbcontainer.crys.lattice, mdbcontainer.crys.lattice):
            raise TypeError("pdbcontainer and mdbcontainer have different crystals")

        if not len(pdbcontainer.crys.basis) == len(mdbcontainer.crys.basis):
            raise TypeError("pdbcontainer and mdbcontainer have different basis")

        for atom1, atom2 in zip(pdbcontainer.crys.chemistry, mdbcontainer.crys.chemistry):
            if not atom1 == atom2:
                raise TypeError("pdbcontainer and mdbcontainer basis atom types don't match")
        for l1, l2 in zip(pdbcontainer.crys.basis, mdbcontainer.crys.basis):
            if not l1 == l2:
                raise TypeError("basis atom types have different numbers in pdbcontainer and mdbcontainer")

        if not pdbcontainer.chem == mdbcontainer.chem:
            raise TypeError("pdbcontainer and mdbcontainer have states on different sublattices")

        self.crys = pdbcontainer.crys
        self.chem = pdbcontainer.chem
        self.pdbcontainer = pdbcontainer
        self.mdbcontainer = mdbcontainer

        self.jnet0 = jnetwrk0[0]
        self.jnet0_ind = jnetwrk0[1]

        self.jnet2 = jnetwrk2[0]
        self.jnet2_ind = jnetwrk2[1]

        self.jumplist = [j for l in self.jnet0 for j in l]
        # self.jumpset = set(self.jumplist)

        self.jumpindices = []
        count = 0
        for l in self.jnet0:
            self.jumpindices.append([])
            for j in l:
                if isinstance(j.state1, SdPair):
                    raise TypeError("The jumpnetwork for bare dumbbells cannot have Sdpairs")
                self.jumpindices[-1].append(count)
                count += 1
        if not Nshells == None:
            self.generate(Nshells)

    def _sortkey(self, entry):
        # Single underscore function means that if we import this class separately in another module, through
        # "from stars import StarSet", then we won't have access to StarSet._sortkey()
        # However, if we import the module "import stars", then we can access stars.StarSet._sortkey()
        # Usually, pre-underscored variables are understood as those which are meant only for internal use by the class.
        sol_pos = self.crys.unit2cart(entry.R_s, self.crys.basis[self.chem][entry.i_s])
        db_pos = self.crys.unit2cart(entry.db.R,
                                     self.crys.basis[self.chem][self.pdbcontainer.iorlist[entry.db.iorind][0]])
        return np.dot(db_pos - sol_pos, db_pos - sol_pos)

    def genIndextoContainer(self, complexStates, mixedstates):
        pureDict = {}
        mixedDict = {}
        for st in complexStates:
            db = st.db - st.db.R
            pureDict[st] = self.pdbcontainer.iorindex[db]

        for st in mixedstates:
            db = st.db - st.db.R
            mixedDict[st] = self.mdbcontainer.iorindex[db]
        return pureDict, mixedDict

    def generate(self, Nshells):
        # Return nothing if Nshells are not specified
        if Nshells is None:
            return
        self.Nshells = Nshells
        z = np.zeros(3, dtype=int)
        if Nshells < 1:
            Nshells = 0
        startshell = set([])
        stateset = set([])
        if Nshells >= 1:
            # build the starting shell
            for j in self.jumplist:
                # Build the first shell from the jump network
                # One by one, keeping the solute at the basis sites of the origin unit cell, put those dumbbell states
                # at those positions, as are dictated by the the jumps.
                # The idea is that a valid jump must be able to bring a dumbbell to a solute site.
                pair = SdPair(self.pdbcontainer.iorlist[j.state1.iorind][0], j.state1.R, j.state2)
                stateset.add(pair)
        lastshell = stateset.copy()
        # Now build the next shells:
        for step in range(Nshells - 1):
            nextshell = set([])
            for j in self.jumplist:
                for pair in lastshell:
                    if not np.allclose(pair.R_s, 0, atol=self.crys.threshold):
                        raise ValueError("The solute is not at the origin")
                    try:
                        pairnew = pair.addjump(j)
                    except ArithmeticError:
                        # If there is somehow a type error, we will get the message.
                        continue
                    if not (pair.i_s == pairnew.i_s and np.allclose(pairnew.R_s, pair.R_s, atol=self.crys.threshold)):
                        raise ArithmeticError("Solute shifted from a complex!(?)")
                    nextshell.add(pairnew)
                    stateset.add(pairnew)
            lastshell = nextshell.copy()

        self.stateset = stateset
        # group the states by symmetry - form the stars
        stars = []
        allset = set([])
        for state in self.stateset:
            if not (state in allset):
                newstar = []
                for gdumb in self.pdbcontainer.G:
                    newstate = state.gop(self.pdbcontainer, gdumb)[0]
                    newstate = newstate - newstate.R_s  # Shift the solute back to the origin unit cell.
                    if newstate in self.stateset:  # Check if this state is allowed to be present.
                        if not newstate in allset:  # Check if this state has already been considered.
                            newstar.append(newstate)
                            allset.add(newstate)
                if len(newstar) == 0:
                    raise ValueError("A star must have at least one state.")
                stars.append(newstar)
        self.stars = stars
        self.sortstars()

        # Keep the indices of the origin states. May be necessary when dealing with their rates and probabilities
        self.originstates = []
        for starind, star in enumerate(self.stars):
            if star[0].is_zero(self.pdbcontainer):
                self.originstates.append(starind)

        self.mixedstartindex = len(self.stars)
        # Now add in the mixed states
        self.mixedstates = []
        for idx, tup in enumerate(self.mdbcontainer.iorlist):
            db = dumbbell(idx, z)
            mdb = SdPair(tup[0], z, db)
            self.mixedstates.append(mdb)

        for l in self.mdbcontainer.symIndlist:
            # The sites and orientations are already grouped - convert them into SdPairs
            newlist = []
            for idx in l:
                db = dumbbell(idx, z)
                mdb = SdPair(self.mdbcontainer.iorlist[idx][0], z, db)
                newlist.append(mdb)
            self.stars.append(newlist)

        self.complexStates = sorted(list(self.stateset), key=self._sortkey)
        self.bareStates = [dumbbell(idx, z) for idx in range(len(self.pdbcontainer.iorlist))]

        # Next, we build up the jtags for omega2 (see Onsager_calc_db module).
        j2initlist = []
        for jt, jlist in enumerate(self.jnet2_ind):
            initindices = defaultdict(list)
            # defaultdict(list) - dictionary creator. (key, value) pairs are such that the value corresponding to a
            # given key is a list. If a key is created for the first time, an empty list is created simultaneously.
            for (i, j), dx in jlist:
                initindices[i].append(j)
            j2initlist.append(initindices)

        self.jtags2 = []
        for initdict in j2initlist:
            jtagdict = {}
            for IS, lst in initdict.items():
                jarr = np.zeros((len(lst), len(self.complexStates) + len(self.mixedstates)), dtype=int)
                for idx, FS in enumerate(lst):
                    if IS == FS:
                        # jarr[idx][IS+len(self.complexStates)]= 1
                        continue
                    jarr[idx][IS + len(self.complexStates)] += 1
                    jarr[idx][FS + len(self.complexStates)] -= 1
                jtagdict[IS] = jarr.copy()
            self.jtags2.append(jtagdict)

        # generate an indexed version of the starset to the iorlists in the container objects
        starindexed = []
        for star in self.stars[:self.mixedstartindex]:
            indlist = []
            for state in star:
                for j, st in enumerate(self.complexStates):
                    if st == state:
                        indlist.append(j)
            starindexed.append(indlist)

        for star in self.stars[self.mixedstartindex:]:
            indlist = []
            for state in star:
                for j, st in enumerate(self.mixedstates):
                    if st == state:
                        indlist.append(j)
            starindexed.append(indlist)

        self.starindexed = starindexed

        self.star2symlist = np.zeros(len(self.stars), dtype=int)
        # The i_th element of this index list gives the corresponding symorlist from which the dumbbell of the
        # representative state of the i_th star comes from.
        for starind, star in enumerate(self.stars[:self.mixedstartindex]):
            # get the dumbbell of the representative state of the star
            db = star[0].db - star[0].db.R
            # now get the symorlist index in which the dumbbell belongs
            symind = self.pdbcontainer.invmap[db.iorind]
            self.star2symlist[starind] = symind

        for starind, star in enumerate(self.stars[self.mixedstartindex:]):
            # get the dumbbell from the representative state of the star
            db = star[0].db - star[0].db.R
            # now get the symorlist index in which the dumbbell belongs
            symind = self.mdbcontainer.invmap[db.iorind]
            self.star2symlist[starind + self.mixedstartindex] = symind

        # self.starindexed -> gives the indices into the complexStates and mixedstates, of the states stored in the
        # starset, i.e, an indexed version of the starset. now generate the index dicts
        # --indexdict -> tell us given a pair state, what is its index in the states list and which star in the starset
        # it belongs to.

        self.complexIndexdict = {}
        for si, star, starind in zip(itertools.count(), self.stars[:self.mixedstartindex],
                                     self.starindexed[:self.mixedstartindex]):
            for state, ind in zip(star, starind):
                self.complexIndexdict[state] = (ind, si)

        self.mixedindexdict = {}
        for si, star, starind in zip(itertools.count(), self.stars[self.mixedstartindex:],
                                     self.starindexed[self.mixedstartindex:]):
            for state, ind in zip(star, starind):
                self.mixedindexdict[state] = (ind, si + self.mixedstartindex)

        # create the starset for the bare dumbbell space
        self.barePeriodicStars = [[dumbbell(idx, np.zeros(3, dtype=int)) for idx in idxlist] for idxlist in
                                  self.pdbcontainer.symIndlist]

        self.bareStarindexed = self.pdbcontainer.symIndlist.copy()
        # for star in self.barePeriodicStars:
        #     indlist = []
        #     for state in star:
        #         for j, st in enumerate(self.bareStates):
        #             if st == state:
        #                 indlist.append(j)
        #     self.bareStarindexed.append(indlist)

        self.bareindexdict = {}
        for si, star, starind in zip(itertools.count(), self.barePeriodicStars, self.bareStarindexed):
            for state, ind in zip(star, starind):
                self.bareindexdict[state] = (ind, si)

    def sortstars(self):
        """sorts the solute-dumbbell complex crystal stars in order of increasing solute-dumbbell separation distance.
        Note that this is called before mixed dumbbell stars are added in. The mixed dumbbells being in a periodic state
        space, all the mixed dumbbell states are at the origin anyway.
        """
        inddict = {}
        for i, star in enumerate(self.stars):
            inddict[i] = self._sortkey(star[0])
        # Now sort the stars according to dx^2, i.e, sort the dictionary by value
        sortlist = sorted(inddict.items(), key=lambda x: x[1])
        # print(sortlist)
        starnew = []
        for (ind, dx2) in sortlist:
            starnew.append(self.stars[ind])
        self.stars = starnew

    def jumpnetwork_omega1(self):
        jumpnetwork = []
        jumpindexed = []
        initstates = []  # list of dicitionaries that store numpy arrays of the form +1 for initial state, -1 for final state
        jumptype = []
        starpair = []
        jumpset = set([])  # set where newly produced jumps will be stored
        for jt, jlist in enumerate(self.jnet0):
            for jnum, j0 in enumerate(jlist):
                # these contain dumbell->dumbell jumps
                for pairind, pair in enumerate(self.complexStates):
                    try:
                        pairnew = pair.addjump(j0)
                    except ArithmeticError:
                        # If anything other than ArithmeticError occurs, we'll get the message.
                        continue
                    if pairnew not in self.stateset:
                        continue
                    # convert them to pair jumps
                    jpair = jump(pair, pairnew, j0.c1, j0.c2)
                    if not jpair in jumpset:  # see if the jump has not already been considered
                        newlist = []
                        indices = []
                        initdict = defaultdict(list)
                        for gdumb in self.pdbcontainer.G:
                            # The solute must be at the origin unit cell - shift it
                            state1new, flip1 = jpair.state1.gop(self.pdbcontainer, gdumb)
                            state2new, flip2 = jpair.state2.gop(self.pdbcontainer, gdumb)
                            state1new -= state1new.R_s
                            state2new -= state2new.R_s
                            if (not state1new in self.stateset) or (not state2new in self.stateset):
                                raise ValueError("symmetrically obtained complex state not found in stateset(?)")
                            jnew = jump(state1new, state2new, jpair.c1 * flip1, jpair.c2 * flip2)
                            if not jnew in jumpset:
                                # if not (np.allclose(jnew.state1.R_s, 0., atol=self.crys.threshold) and np.allclose(
                                #         jnew.state2.R_s, 0., atol=self.crys.threshold)):
                                #     raise RuntimeError("Solute shifted from origin")
                                # if not (jnew.state1.i_s == jnew.state1.i_s):
                                #     raise RuntimeError(
                                #         "Solute must remain in exactly the same position before and after the jump")
                                newlist.append(jnew)
                                newlist.append(-jnew)
                                # we can add the negative since solute always remains at the origin
                                jumpset.add(jnew)
                                jumpset.add(-jnew)

                        # remove redundant rotations, if present
                        if (np.allclose(disp(self.pdbcontainer, newlist[0].state1, newlist[0].state2), np.zeros(3),
                                       atol=self.pdbcontainer.crys.threshold) and
                                newlist[0].state1.i_s == newlist[0].state2.i_s):
                            newnewlist = set([])
                            for j in newlist:
                                j_equiv = jump(j.state1, j.state2, -1 * j.c1, -1 * j.c2)
                                if not j_equiv in newnewlist:
                                    newnewlist.add(j)
                            newlist = list(newnewlist)
                        for jmp in newlist:
                            if not (jmp.state1 in self.stateset):
                                raise ValueError("state not found in stateset?\n{}".format(jmp.state1))
                            if not (jmp.state2 in self.stateset):
                                raise ValueError("state not found in stateset?\n{}".format(jmp.state2))
                            initial = self.complexIndexdict[jmp.state1][0]
                            final = self.complexIndexdict[jmp.state2][0]
                            indices.append(((initial, final), disp(self.pdbcontainer, jmp.state1, jmp.state2)))
                            initdict[initial].append(final)
                        jumpnetwork.append(newlist)
                        jumpindexed.append(indices)
                        initstates.append(initdict)
                        # initdict contains all the initial states as keys, and the values as the lists final states
                        # from the initial states for the given jump type.
                        jumptype.append(jt)
        jtags = []
        for initdict in initstates:
            arrdict = {}
            for IS, lst in initdict.items():
                jtagarr = np.zeros((len(lst), len(self.complexStates) + len(self.mixedstates)), dtype=int)
                for jnum, FS in enumerate(lst):
                    jtagarr[jnum][IS] += 1
                    jtagarr[jnum][FS] -= 1
                arrdict[IS] = jtagarr.copy()
            jtags.append(arrdict)

        return (jumpnetwork, jumpindexed, jtags), jumptype

    def jumpnetwork_omega34(self, cutoff, solv_solv_cut, solt_solv_cut, closestdistance):
        # building omega_4 -> association - c2=-1 -> since solvent movement is tracked
        # cutoff required - solute-solvent as well as solvent solvent
        alljumpset_omega4 = set([])
        symjumplist_omega4 = []
        symjumplist_omega4_indexed = []
        omega4inits = []

        # alljumpset_omega3=set([])

        symjumplist_omega3 = []
        symjumplist_omega3_indexed = []
        omega3inits = []

        symjumplist_omega43_all = []
        symjumplist_omega43_all_indexed = []
        alljumpset_omega43_all = set([])

        for p_pure in self.complexStates:
            if p_pure.is_zero(self.pdbcontainer):  # Specator rotating into mixed does not make sense.
                continue
            for p_mixed in self.mixedstates:
                if not (np.allclose(p_pure.R_s, 0, atol=self.crys.threshold)
                        and np.allclose(p_mixed.R_s, 0, atol=self.crys.threshold)):
                    raise RuntimeError("Solute shifted from origin - cannot happen")
                if not (p_pure.i_s == p_mixed.i_s):
                    # The solute must remain in exactly the same position before and after the jump
                    continue
                for c1 in [-1, 1]:
                    j = jump(p_pure, p_mixed, c1, -1)
                    dx = disp4(self.pdbcontainer, self.mdbcontainer, j.state1, j.state2)
                    if np.dot(dx, dx) > cutoff ** 2: continue
                    if not j in alljumpset_omega4:  # check if jump already considered
                        # if a jump is in alljumpset_omega4, it's negative will have to be in alljumpset_omega3
                        if not collision_self(self.pdbcontainer, self.mdbcontainer, j, solv_solv_cut,
                                              solt_solv_cut) and not collision_others(self.pdbcontainer,
                                                                                      self.mdbcontainer, j,
                                                                                      closestdistance):
                            newset = set([])
                            for g in self.crys.G:
                                for gdumb, gval in self.pdbcontainer.G_crys.items():
                                    if gval == g:
                                        gdumb_pure = gdumb

                                for gdumb, gval in self.mdbcontainer.G_crys.items():
                                    if gval == g:
                                        gdumb_mixed = gdumb

                                if not self.pdbcontainer.G_crys[gdumb_pure] ==\
                                       self.mdbcontainer.G_crys[gdumb_mixed]:
                                    continue
                                state1new, flip1 = j.state1.gop(self.pdbcontainer, gdumb_pure)
                                state2new = j.state2.gop(self.mdbcontainer, gdumb_mixed, complex=False)
                                jnew = jump(state1new - state1new.R_s, state2new - state2new.R_s, j.c1*flip1, -1)
                                if not jnew in newset:
                                    if jnew.state1.i_s == self.pdbcontainer.iorlist[jnew.state1.db.iorind][0]:
                                        if np.allclose(jnew.state1.R_s, jnew.state1.db.R, atol=self.crys.threshold):
                                            raise RuntimeError("Initial state mixed")
                                    if not (jnew.state2.i_s == self.mdbcontainer.iorlist[jnew.state2.db.iorind][0]
                                            and np.allclose(jnew.state2.R_s, jnew.state2.db.R, self.crys.threshold)):
                                        raise RuntimeError("Final state not mixed")
                                    newset.add(jnew)
                                    alljumpset_omega4.add(jnew)

                            newset = list(newset)
                            newnegset = [-jmp for jmp in newset]
                            newallset = []
                            for i in range(len(newset)):
                                newallset.append(newset[i])
                                newallset.append(newnegset[i])

                            new4index = []
                            new3index = []
                            newallindex = []
                            jinitdict3 = defaultdict(list)
                            jinitdict4 = defaultdict(list)

                            for jmp in newset:
                                pure_ind = self.complexIndexdict[jmp.state1][0]
                                mixed_ind = self.mixedindexdict[jmp.state2][0]
                                # omega4 has pure as initial, omega3 has pure as final
                                jinitdict4[pure_ind].append(mixed_ind)
                                jinitdict3[mixed_ind].append(pure_ind)
                                dx = disp4(self.pdbcontainer, self.mdbcontainer, jmp.state1, jmp.state2)
                                new4index.append(((pure_ind, mixed_ind), dx.copy()))
                                new3index.append(((mixed_ind, pure_ind), -dx))
                                newallindex.append(((pure_ind, mixed_ind), dx.copy()))
                                newallindex.append(((mixed_ind, pure_ind), -dx))

                            symjumplist_omega4.append(newset)
                            omega4inits.append(jinitdict4)
                            symjumplist_omega4_indexed.append(new4index)

                            symjumplist_omega3.append(newnegset)
                            omega3inits.append(jinitdict3)
                            symjumplist_omega3_indexed.append(new3index)

                            symjumplist_omega43_all.append(newallset)
                            symjumplist_omega43_all_indexed.append(newallindex)

        # Now build the jtags
        jtags4 = []
        jtags3 = []

        for initdict in omega4inits:
            jarrdict = {}
            for IS, lst in initdict.items():
                jarr = np.zeros((len(lst), len(self.complexStates) + len(self.mixedstates)), dtype=int)
                for idx, FS in enumerate(lst):
                    jarr[idx][IS] += 1
                    jarr[idx][FS + len(self.complexStates)] -= 1
                jarrdict[IS] = jarr.copy()
            jtags4.append(jarrdict)

        for initdict in omega3inits:
            jarrdict = {}
            for IS, lst in initdict.items():
                jarr = np.zeros((len(lst), len(self.complexStates) + len(self.mixedstates)), dtype=int)
                for idx, FS in enumerate(lst):
                    jarr[idx][IS + len(self.complexStates)] += 1
                    jarr[idx][FS] -= 1
                jarrdict[IS] = jarr.copy()
            jtags3.append(jarrdict)

        return (symjumplist_omega43_all, symjumplist_omega43_all_indexed), (
            symjumplist_omega4, symjumplist_omega4_indexed, jtags4), (
                   symjumplist_omega3, symjumplist_omega3_indexed, jtags3)
