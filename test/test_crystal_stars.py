import numpy as np
# from jumpnet3 import *
from stars import *
from test_structs import *
from states import *
# from gensets import *
import unittest


class test_StarSet(unittest.TestCase):

    def setUp(self):
        famp0 = [np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126]
        family = [famp0]
        self.pdbcontainer = dbStates(cube, 0, family)
        self.mdbcontainer = mStates(cube, 0, family)
        jset0 = self.pdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        jset2 = self.mdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        self.crys_stars = StarSet(self.pdbcontainer, self.mdbcontainer, jset0, jset2, 2)

    def test_generate(self):
        # test if the starset is generated correctly
        tot_st = 0
        for l in (self.crys_stars.stars[:self.crys_stars.mixedstartindex]):
            tot_st += len(l)
        self.assertEqual(tot_st, len(self.crys_stars.stateset))

        o = np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126

        dbInd = self.pdbcontainer.getIndex((0, o))
        db = dumbbell(dbInd, np.array([1, 0, 0], dtype=int))
        pair_test = SdPair(0, np.zeros(3, dtype=int), db)
        idxlist = []
        for idx, l in enumerate(self.crys_stars.stars):
            for state in l:
                if state == pair_test:
                    idxlist.append(idx)
        self.assertEqual(len(idxlist), 1)
        self.assertEqual(len(self.crys_stars.stars[idxlist[0]]), 6)

        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        pdbcontainer = dbStates(DC_Si, 0, family)
        mdbcontainer = mStates(DC_Si, 0, family)
        jset0 = pdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        jset2 = mdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        crys_stars = StarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)
        # Check that if a state is present in the stateset, it's symmetrically equivalent states are also present in the
        # stateset
        for st in crys_stars.stateset:
            for gdumb in pdbcontainer.G:
                stnew, flipind = st.gop(pdbcontainer, gdumb)
                stnew -= stnew.R_s
                self.assertTrue(stnew in crys_stars.stateset)

        # Check that the stars are properly generated
        for star in crys_stars.stars[:crys_stars.mixedstartindex]:
            repr = star[0]
            considered_already = set([])
            count = 0
            for gdumb in crys_stars.pdbcontainer.G:
                stnew = repr.gop(crys_stars.pdbcontainer, gdumb)[0]
                stnew -= stnew.R_s
                if stnew in star and not stnew in considered_already:
                    count += 1
                    considered_already.add(stnew)
            self.assertEqual(count, len(star))

    def test_indexing_stars(self):
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        hcp_Mg = crystal.Crystal.HCP(0.3294, chemistry=["Mg"])
        pdbcontainer = dbStates(hcp_Mg, 0, family)
        mdbcontainer = mStates(hcp_Mg, 0, family)
        jset0 = pdbcontainer.jumpnetwork(0.45, 0.01, 0.01)
        jset2 = mdbcontainer.jumpnetwork(0.45, 0.01, 0.01)
        crys_stars = StarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)

        # Check that the stars are properly generated
        for star in crys_stars.stars[:crys_stars.mixedstartindex]:
            repr = star[0]
            considered_already = set([])
            count = 0
            for gdumb in crys_stars.pdbcontainer.G:
                stnew = repr.gop(crys_stars.pdbcontainer, gdumb)[0]
                stnew -= stnew.R_s
                if stnew in star and not stnew in considered_already:
                    count += 1
                    considered_already.add(stnew)
            self.assertEqual(count, len(star))

        # test indexing
        # check that all states are accounted for
        for i in range(len(crys_stars.stars)):
            self.assertEqual(len(crys_stars.stars[i]), len(crys_stars.starindexed[i]))

        for star, starind in zip(crys_stars.stars[:crys_stars.mixedstartindex],
                                 crys_stars.starindexed[:crys_stars.mixedstartindex]):
            for state, stateind in zip(star, starind):
                self.assertEqual(state, crys_stars.complexStates[stateind])

        for star, starind in zip(crys_stars.stars[crys_stars.mixedstartindex:],
                                 crys_stars.starindexed[crys_stars.mixedstartindex:]):
            for state, stateind in zip(star, starind):
                self.assertEqual(state, crys_stars.mixedstates[stateind])

    def test_dicts(self):
        hcp_Mg = crystal.Crystal.HCP(0.3294, chemistry=["Mg"])
        fcc_Ni = crystal.Crystal.FCC(0.352, chemistry=["Ni"])
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        crys_list = [hcp_Mg, fcc_Ni, DC_Si]
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        for struct, crys in enumerate(crys_list):
            pdbcontainer = dbStates(crys, 0, family)
            mdbcontainer = mStates(crys, 0, family)
            jset0 = pdbcontainer.jumpnetwork(0.45, 0.01, 0.01)
            jset2 = mdbcontainer.jumpnetwork(0.45, 0.01, 0.01)
            # 4.5 angst should cover atleast the nn distance in all the crystals
            # create starset
            crys_stars = StarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)

            # first, test the pure dictionary
            for key, value in crys_stars.complexIndexdict.items():
                self.assertEqual(key, crys_stars.complexStates[value[0]])
                self.assertTrue(crys_stars.complexStates[value[0]] in crys_stars.stars[value[1]])

            # Next, the mixed dictionary
            for key, value in crys_stars.mixedindexdict.items():
                self.assertEqual(key, crys_stars.mixedstates[value[0]])
                self.assertTrue(crys_stars.mixedstates[value[0]] in crys_stars.stars[value[1]])

    def test_jumpnetworks(self):
        # See the example file. Provides much clearer understanding.
        # The tests have just one Wyckoff site for now
        def inlist(jmp, jlist):
            return any(j == jmp for j in jlist)

        hcp_Mg = crystal.Crystal.HCP(0.3294, chemistry=["Mg"])
        fcc_Ni = crystal.Crystal.FCC(0.352, chemistry=["Ni"])
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        crys_list = [DC_Si]
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        for struct, crys in enumerate(crys_list):
            pdbcontainer = dbStates(crys, 0, family)
            mdbcontainer = mStates(crys, 0, family)
            jset0 = pdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
            jset2 = mdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
            # 4.5 angst should cover atleast the nn distance in all the crystals
            # create starset
            crys_stars = StarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)

            ##TEST omega_1
            (omega1_network, omega1_indexed, omega1tag), om1types = crys_stars.jumpnetwork_omega1()
            for jlist, initdict in zip(omega1_indexed, omega1tag):
                for IS, jtag in initdict.items():
                    # go through the rows of the jtag:
                    for row in range(len(jtag)):
                        self.assertTrue(jtag[row][IS] == 1)
                        for column in range(len(crys_stars.complexStates) + len(crys_stars.mixedstates)):
                            if jtag[row][column] == -1:
                                self.assertTrue(any(i == IS and j == column for (i, j), dx in jlist))
                                # If any is true, then that means only one is true, since a jump b/w two states is present only once.
            # select a jump list in random
            x = np.random.randint(0, len(omega1_network))
            # select any jump from this list at random
            y = np.random.randint(0, len(omega1_network[x]))
            jmp = omega1_network[x][y]
            jlist = []
            # reconstruct the list using the selected jump, without using has tables (sets)
            for g in crys.G:
                jnew1 = jmp.gop(crys, 0, g)
                db1new = pdbcontainer.gdumb(g, jmp.state1.db)
                db2new = pdbcontainer.gdumb(g, jmp.state2.db)
                # shift the states back to the origin unit cell
                state1new = SdPair(jnew1.state1.i_s, jnew1.state1.R_s, db1new[0]) - jnew1.state1.R_s
                state2new = SdPair(jnew1.state2.i_s, jnew1.state2.R_s, db2new[0]) - jnew1.state1.R_s
                jnew = jump(state1new, state2new, jnew1.c1 * db1new[1], jnew1.c2 * db2new[1])
                if not inlist(jnew, jlist):
                    jlist.append(jnew)
                    jlist.append(-jnew)
            if (jmp.state1.is_zero() and jmp.state2.is_zero()):
                self.assertEqual(len(jlist) / 2., len(omega1_network[x]))
            else:
                self.assertEqual(len(jlist), len(omega1_network[x]))
            count = 0
            for j in jlist:
                for j1 in omega1_network[x]:
                    if j == j1:
                        count += 1
                        break
            self.assertEqual(count, len(jlist))

            # See that irrespective of solute location, if the jumps of the dumbbells are the same, then the jump type is also the same
            for i, jlist1 in enumerate(omega1_network):
                for j, jlist2 in enumerate(omega1_network):
                    if i == j: continue
                    for j1 in jlist1:
                        for j2 in jlist2:
                            if j1.state1.db == j2.state1.db and j1.state2.db == j2.state2.db:
                                if j1.c1 == j2.c1 and j1.c2 == j2.c2:
                                    self.assertTrue(om1types[i] == om1types[j],
                                                    msg="{},{}\n{}\n{}".format(i, j, j1, j2))

            omega43, omega4, omega3 = crys_stars.jumpnetwork_omega34(0.45, 0.01, 0.01, 0.01)
            omega43_all, omega4_network, omega3_network = omega43[0], omega4[0], omega3[0]
            omega43_all_indexed, omega4_network_indexed, omega3_network_indexed = omega43[1], omega4[1], omega3[1]
            omega4tag, omega3tag = omega4[2], omega3[2]
            self.assertEqual(len(omega4_network), len(omega3_network))
            for jl4, jl3 in zip(omega4_network, omega3_network):
                self.assertEqual(len(jl3), len(jl4))
            ##TEST omega3 and omega4
            # test that the tag lists have proper length
            self.assertEqual(len(omega4tag), len(omega4_network))
            self.assertEqual(len(omega3tag), len(omega3_network))
            omeg34list = [omega3_network, omega4_network]
            for i, omega in enumerate(omeg34list):
                x = np.random.randint(0, len(omega))
                y = np.random.randint(0, len(omega[x]))
                jmp = omega[x][y]
                jlist = []
                if i == 0:  # then build omega3
                    for g in crys.G:
                        jnew = jmp.gop(crys, 0, g)
                        db2new = pdbcontainer.gdumb(g, jmp.state2.db)
                        state2new = SdPair(jnew.state2.i_s, jnew.state2.R_s, db2new[0]) - jnew.state2.R_s
                        jnew = jump(jnew.state1 - jnew.state1.R_s, state2new, -1, jnew.c2 * db2new[1])
                        if not inlist(jnew, jlist):
                            jlist.append(jnew)
                else:  # build omega4
                    for g in crys.G:
                        jnew = jmp.gop(crys, 0, g)
                        db1new = pdbcontainer.gdumb(g, jmp.state1.db)
                        state1new = SdPair(jnew.state1.i_s, jnew.state1.R_s, db1new[0]) - jnew.state1.R_s
                        jnew = jump(state1new, jnew.state2 - jnew.state2.R_s, jnew.c1 * db1new[1], -1)
                        if not inlist(jnew, jlist):
                            jlist.append(jnew)
                self.assertEqual(len(jlist), len(omega[x]))
                count = 0
                for j1 in omega[x]:
                    for j in jlist:
                        if j == j1:
                            count += 1
                            break
                self.assertEqual(count, len(omega[x]))

            ##Test indexing of the jump networks
            # First, omega_1
            for jlist, jindlist in zip(omega1_network, omega1_indexed):
                for jmp, indjmp in zip(jlist, jindlist):
                    self.assertTrue(jmp.state1 == crys_stars.complexStates[indjmp[0][0]])
                    self.assertTrue(jmp.state2 == crys_stars.complexStates[indjmp[0][1]])
            # Next, omega34
            for jlist, jindlist in zip(omega4_network, omega4_network_indexed):
                for jmp, indjmp in zip(jlist, jindlist):
                    self.assertTrue(jmp.state1 == crys_stars.complexStates[indjmp[0][0]])
                    self.assertTrue(jmp.state2 == crys_stars.mixedstates[indjmp[0][1]])

            for jlist, jindlist in zip(omega3_network, omega3_network_indexed):
                for jmp, indjmp in zip(jlist, jindlist):
                    # print(jmp.state1)
                    # print()
                    # print(crys_stars.mixedstates[indjmp[0][0]])
                    self.assertTrue(jmp.state1 == crys_stars.mixedstates[indjmp[0][0]], msg="{}".format(struct))
                    self.assertTrue(jmp.state2 == crys_stars.complexStates[indjmp[0][1]])
            # testing the tags
            # First, omega4
            for jlist, initdict in zip(omega4_network_indexed, omega4tag):
                for IS, jtag in initdict.items():
                    # go through the rows of the jtag:
                    for row in range(len(jtag)):
                        self.assertTrue(jtag[row][IS] == 1)
                        for column in range(len(crys_stars.complexStates) + len(crys_stars.mixedstates)):
                            if jtag[row][column] == -1:
                                self.assertTrue(
                                    any(i == IS and j == column - len(crys_stars.complexStates) for (i, j), dx in
                                        jlist))
            # Next, omega3
            for jlist, initdict in zip(omega3_network_indexed, omega3tag):
                for IS, jtag in initdict.items():
                    # go through the rows of the jtag:
                    for row in range(len(jtag)):
                        self.assertTrue(jtag[row][IS + len(crys_stars.complexStates)] == 1)
                        for column in range(len(crys_stars.complexStates) + len(crys_stars.mixedstates)):
                            if jtag[row][column] == -1:
                                self.assertTrue(any(i == IS and j == column for (i, j), dx in jlist))
            # Next, omega2 to mixedstates
            jnet2, jnet2stateindex = crys_stars.jumpnetwork_omega2, crys_stars.jumpnetwork_omega2_indexed
            for i in range(len(jnet2)):
                self.assertEqual(len(jnet2[i]), len(jnet2stateindex[i]))
                for jpair, jind in zip(jnet2[i], jnet2stateindex[i]):
                    IS = crys_stars.mixedstates[jind[0][0]]
                    FS = crys_stars.mixedstates[jind[0][1]]
                    self.assertEqual(IS, jpair.state1 - jpair.state1.R_s,
                                     msg="\n{} not equal to {}".format(IS, jpair.state1))
                    self.assertEqual(FS, jpair.state2 - jpair.state2.R_s,
                                     msg="\n{} not equal to {}".format(FS, jpair.state2))

            for jlist, initdict in zip(jnet2stateindex, crys_stars.jtags2):
                for IS, jtag in initdict.items():
                    # go through the rows of the jtag:
                    for row in range(len(jtag)):
                        # The column corresponding to the intial state must have 1.
                        self.assertTrue(jtag[row][IS + len(crys_stars.complexStates)] == 1 or jtag[row][
                            IS + len(crys_stars.complexStates)] == 0,
                                        msg="{}".format(jtag[row][IS + len(crys_stars.complexStates)]))
                        # the zero appears when the intial and final states are the same (but just translated in the lattice) so that they have the same periodic eta vector
                        for column in range(len(crys_stars.complexStates) + len(crys_stars.mixedstates)):
                            if jtag[row][column] == -1:
                                self.assertTrue(
                                    any(i == IS and j == column - len(crys_stars.complexStates) for (i, j), dx in
                                        jlist))

    def test_om1types(self):
        """
        This is an expensive test, so did not include in previous section
        """
        # hcp_Mg=crystal.Crystal.HCP(0.3294,chemistry=["Mg"])
        # fcc_Ni = crystal.Crystal.FCC(0.352,chemistry=["Ni"])
        # latt = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])*0.55
        DC_Si = crystal.Crystal(np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55,
                                [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        crys_list = [DC_Si]
        for crys in crys_list:
            pdbcontainer = dbStates(crys, 0, family)
            mdbcontainer = mStates(crys, 0, family)
            jset0 = pdbcontainer.jumpnetwork(0.24, 0.01, 0.01)
            jset2 = mdbcontainer.jumpnetwork(0.24, 0.01, 0.01)
            # 4.5 angst should cover atleast the nn distance in all the crystals
            # create starset
            crys_stars = StarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)
            (omega1_network, omega1_indexed, omega1tag), om1types = crys_stars.jumpnetwork_omega1()
            for i, jlist1 in enumerate(omega1_network):
                for j, jlist2 in enumerate(omega1_network):
                    if i == j: continue
                    for j1 in jlist1:
                        for j2 in jlist2:
                            if j1.state1.db == j2.state1.db and j1.state2.db == j2.state2.db:
                                if j1.c1 == j2.c1 and j1.c2 == j2.c2:
                                    self.assertTrue(om1types[i] == om1types[j],
                                                    msg="{},{}\n{}\n{}".format(i, j, j1, j2))

    def test_sort_stars(self):
        DC_Si = crystal.Crystal(np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55,
                                [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        pdbcontainer = dbStates(DC_Si, 0, family)
        mdbcontainer = mStates(DC_Si, 0, family)
        jset0 = pdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        jset2 = mdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        # 4.5 angst should cover atleast the nn distance in all the crystals
        # create starset
        crys_stars = StarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)
        dx_list = []
        for sts in zip(crys_stars.stars[:crys_stars.mixedstartindex]):
            st0 = sts[0][0]
            sol_pos = crys_stars.crys.unit2cart(st0.R_s, crys_stars.crys.basis[crys_stars.chem][st0.i_s])
            db_pos = crys_stars.crys.unit2cart(st0.db.R, crys_stars.crys.basis[crys_stars.chem][st0.db.i])
            dx = np.linalg.norm(db_pos - sol_pos)
            dx_list.append(dx)
        self.assertTrue(np.allclose(np.array(dx_list), np.array(sorted(dx_list))),
                        msg="\n{}\n{}".format(dx_list, sorted(dx_list)))
