import numpy as np
import onsager.crystal as crystal
from stars import *
from test_structs import *
from states import *
from representations import *
from functools import reduce
from vector_stars import *
import unittest
from collections import defaultdict
import pickle


class test_vecstars(unittest.TestCase):
    def setUp(self):
        latt = np.array([[0., 0.1, 0.5], [0.3, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        self.DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = dbStates(self.DC_Si, 0, family)
        self.mdbcontainer_si = mStates(self.DC_Si, 0, family)

        self.jset0, self.jset2 = self.pdbcontainer_si.jumpnetwork(0.3, 0.01, 0.01), self.mdbcontainer_si.jumpnetwork(
            0.3, 0.01, 0.01)

        self.crys_stars = StarSet(self.pdbcontainer_si, self.mdbcontainer_si, self.jset0, self.jset2, Nshells=1)
        self.vec_stars = vectorStars(self.crys_stars)

        self.om2tags = self.vec_stars.starset.jtags2
        # generate 1, 3 and 4 jumpnetworks
        (self.jnet_1, self.jnet_1_indexed, self.om1tags), self.jtype = self.crys_stars.jumpnetwork_omega1()
        (self.symjumplist_omega43_all, self.symjumplist_omega43_all_indexed), (
            self.symjumplist_omega4, self.symjumplist_omega4_indexed, self.om4tags), (
            self.symjumplist_omega3, self.symjumplist_omega3_indexed,
            self.om3tags) = self.crys_stars.jumpnetwork_omega34(
            0.3, 0.01, 0.01, 0.01)

        self.W0list = np.random.rand(len(self.vec_stars.starset.jnet0))
        self.W1list = np.random.rand(len(self.jnet_1))
        self.W2list = np.random.rand(len(self.jset2[0]))
        self.W3list = np.random.rand(len(self.symjumplist_omega3))
        self.W4list = np.random.rand(len(self.symjumplist_omega4))

        # generate all the bias expansions - will separate out later
        self.biases = self.vec_stars.biasexpansion(self.jnet_1, self.jset2[0], self.jtype, self.symjumplist_omega43_all)
        self.rateExps = self.vec_stars.rateexpansion(self.jnet_1, self.jtype, self.symjumplist_omega43_all)

    def test_basis(self):

        count_origin_states = 0
        for Rstar in self.vec_stars.vecpos[:self.vec_stars.Nvstars_pure]:
            if Rstar[0].is_zero(self.vec_stars.starset.pdbcontainer):
                count_origin_states += 1
        self.assertTrue(count_origin_states > 0)

        self.assertEqual(len(self.vec_stars.vecpos), len(self.vec_stars.vecvec))
        self.assertEqual(len(self.vec_stars.vecpos_bare), len(self.vec_stars.vecvec_bare))
        # First, complex states
        for vecstarind in range(self.vec_stars.Nvstars_pure):
            # get the representative state of the star
            testvecstate = self.vec_stars.vecpos[vecstarind][0]
            count = 0
            listind = []
            for i in range(self.vec_stars.Nvstars_pure):
                if self.vec_stars.vecpos[vecstarind][0] == self.vec_stars.vecpos[i][0]:
                    count += 1
                    listind.append(i)
                    # The number of times the position list is repeated is also the dimensionality of the basis.

            # Next see what is the algaebric multiplicity of the eigenvalue 1.
            glist = []
            for gdumb in self.vec_stars.starset.pdbcontainer.G:
                pairnew = testvecstate.gop(self.vec_stars.starset.pdbcontainer, gdumb, complex=True)[0]
                pairnew = pairnew - pairnew.R_s
                if pairnew == testvecstate:
                    glist.append(self.vec_stars.starset.pdbcontainer.G_crys[gdumb])

            sumg = sum([g.cartrot for g in glist]) / len(glist)
            vals, vecs = np.linalg.eig(sumg)
            count_eigs = 0

            for val in vals:
                if np.allclose(val, 1.0, atol=1e-8):
                    count_eigs += 1
            self.assertEqual(count, count_eigs, msg="{}".format(testvecstate))

            # Check that the basis vectors are also left invariant by the group operations in glist.
            # Note - a rotation might be 180 degrees, in which case the vector will be rotated as such.
            # Rotation of the dumbbell by 180 degrees is considered to leave the complex unchanged.
            # It also does not take the vector out of the space encompassed by the basis vectors.
            # for v in [self.vec_stars.vecvec[ind][0] for ind in listind]:
            #     for g in glist:
            #         self.assertTrue(np.allclose(v, np.dot(g.cartrot, v)) or np.allclose(-v, np.dot(g.cartrot, v)))

        # Now, mixed dumbbells
        for vecstarind in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):
            # get the representative state of the star
            testvecstate = self.vec_stars.vecpos[vecstarind][0]
            count = 0
            listind = []

            for i in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):
                if self.vec_stars.vecpos[vecstarind][0] == self.vec_stars.vecpos[i][0]:
                    count += 1
                    listind.append(i)
                    # The number of times the position list is repeated is also the dimensionality of the basis.

            # Next see what is the algaebric multiplicity of the eigenvalue 1.
            glist = []
            for gdumb in self.vec_stars.starset.mdbcontainer.G:
                pairnew = testvecstate.gop(self.vec_stars.starset.mdbcontainer, gdumb, complex=False)
                pairnew = pairnew - pairnew.R_s
                if pairnew == testvecstate:
                    glist.append(self.vec_stars.starset.mdbcontainer.G_crys[gdumb])

            sumg = sum([g.cartrot for g in glist]) / len(glist)
            vals, vecs = np.linalg.eig(sumg)
            count_eigs = 0

            for val in vals:
                if np.allclose(val, 1.0, atol=1e-8):
                    count_eigs += 1
            self.assertEqual(count, count_eigs, msg="{}".format(testvecstate))

            # Check that the basis vectors are also left invariant by the group operations in glist.
            # Note - a rotation might be 180 degrees, in which case the vector will be rotated as such.
            # Rotation of the dumbbell by 180 degrees is considered to leave the complex unchanged.
            # It also does not take the vector out of the space encompassed by the basis vectors.
            # for v in [self.vec_stars.vecvec[ind][0] for ind in listind]:
            #     for g in glist:
            #         self.assertTrue(np.allclose(v, np.dot(g.cartrot, v)))

        # Let's also do this for the bare vector stars
        for vecstarind in range(len(self.vec_stars.vecpos_bare)):
            # get the representative state of the star
            testvecstate = self.vec_stars.vecpos_bare[vecstarind][0]
            count = 0
            listind = []
            for i in range(len(self.vec_stars.vecpos_bare)):
                if self.vec_stars.vecpos_bare[vecstarind][0] == self.vec_stars.vecpos_bare[i][0]:
                    count += 1
                    listind.append(i)
                    # The number of times the position list is repeated is also the dimensionality of the basis.

            # Next see what is the algaebric multiplicity of the eigenvalue 1.
            glist = []
            for gdumb in self.vec_stars.starset.pdbcontainer.G:
                dbnew = testvecstate.gop(self.vec_stars.starset.pdbcontainer, gdumb, pure=True)[0]
                dbnew = dbnew - dbnew.R
                if dbnew == testvecstate:
                    glist.append(self.vec_stars.starset.pdbcontainer.G_crys[gdumb])

            sumg = sum([g.cartrot for g in glist]) / len(glist)
            vals, vecs = np.linalg.eig(sumg)
            count_eigs = 0
            for val in vals:
                if np.allclose(val, 1.0, atol=1e-8):
                    count_eigs += 1
            self.assertEqual(count, count_eigs, msg="{}".format(testvecstate))

            # for v in [self.vec_stars.vecvec[ind][0] for ind in listind]:
            #     for g in glist:
            #         self.assertTrue(np.allclose(v, np.dot(g.cartrot, v)) or np.allclose(-v, np.dot(g.cartrot, v)),
            #                         msg="\n{},\n{}\n{}"
            #                         .format(g.cartrot, v,
            #                                 self.vec_stars.starset.pdbcontainer.iorlist[testvecstate.iorind]))

    def test_state_indexing(self):
        for st in self.vec_stars.starset.complexStates:
            indToVecStars = self.vec_stars.stateToVecStar_pure[st]
            for tup in indToVecStars:
                self.assertEqual(st, self.vec_stars.vecpos[tup[0]][tup[1]])

        for st in self.vec_stars.starset.mixedstates:
            indToVecStars = self.vec_stars.stateToVecStar_mixed[st]
            for tup in indToVecStars:
                self.assertEqual(st, self.vec_stars.vecpos[tup[0]][tup[1]])

    def test_bare_bias_expansion(self):
        if len(self.vec_stars.vecpos_bare) > 0:
            for i in range(len(self.vec_stars.vecpos_bare)):
                starind = i
                st = self.vec_stars.vecpos_bare[starind][0]
                n = np.random.randint(1, len(self.vec_stars.vecpos_bare[starind]))
                st2 = self.vec_stars.vecpos_bare[starind][n]
                bias_st = np.zeros(3)
                bias_st2 = np.zeros(3)
                count = 0
                for jt, jumplist in enumerate(self.vec_stars.starset.jnet0):
                    for j in jumplist:
                        if st == j.state1:
                            count += 1
                            dx = disp(self.pdbcontainer_si, j.state1, j.state2)
                            bias_st += dx * self.W0list[jt]
                        if st2 == j.state1:
                            dx = disp(self.pdbcontainer_si, j.state1, j.state2)
                            bias_st2 += dx * self.W0list[jt]

                biasBareExp = self.biases[-1]
                self.assertTrue(count >= 1)
                tot_bias_bare = np.dot(biasBareExp, self.W0list)
                indlist = self.vec_stars.stateToVecStar_bare[st]
                # for ind, starlist in enumerate(self.vec_stars.vecpos_bare):
                #     if starlist[0] == st:
                #         indlist.append(ind)
                bias_bare_cartesian = sum(
                    [tot_bias_bare[tup[0]] * self.vec_stars.vecvec_bare[tup[0]][tup[1]] for tup in indlist])
                self.assertTrue(np.allclose(bias_bare_cartesian, bias_st), msg="\n{}\n{}\n{}\n{}".
                                format(bias_bare_cartesian, bias_st,
                                       [tot_bias_bare[tup[0]] * self.vec_stars.vecvec_bare[tup[0]][tup[1]] for tup in
                                        indlist],
                                       tot_bias_bare))

                indlist = self.vec_stars.stateToVecStar_bare[st2]
                bias_bare_cartesian2 = sum(
                    [tot_bias_bare[tup[0]] * self.vec_stars.vecvec_bare[tup[0]][tup[1]] for tup in indlist])
                self.assertTrue(np.allclose(bias_bare_cartesian2, bias_st2))

        else:  # we have to check that the non-local bias vectors coming out are zero
            print("checking zero non-local bias for pure dumbbells")
            for star in self.vec_stars.starset.barePeriodicStars:
                for st in star:
                    bias_st = np.zeros(3)
                    count = 0
                    for jt, jumplist in enumerate(self.vec_stars.starset.jnet0):
                        for j in jumplist:
                            if st == j.state1:
                                count += 1
                                dx = disp(self.pdbcontainer_si, j.state1, j.state2)
                                bias_st += dx * self.W0list[jt]
                                # print(bias_st)
                    self.assertTrue(np.allclose(bias_st, np.zeros(3)))
                    self.assertTrue(count >= 1)

    def test_bias1expansions(self):

        for i in range(self.vec_stars.Nvstars_pure):
            # test bias_1
            # select a representative state and another state in the same star at random
            # from complex state space
            starind = i
            st = self.vec_stars.vecpos[starind][0]  # get the representative state.
            n = np.random.randint(1, len(self.vec_stars.vecpos[starind]))
            st2 = self.vec_stars.vecpos[starind][n]
            # Now, we calculate the total bias vector - zero for solute in complex space
            bias_st_solvent = np.zeros(3)
            bias_st_solvent2 = np.zeros(3)
            count = 0
            for jt, jlist in enumerate(self.jnet_1):
                for j in jlist:
                    if st == j.state1:
                        count += 1
                        dx = disp(self.pdbcontainer_si, j.state1, j.state2)
                        bias_st_solvent += self.W1list[jt] * dx
                    if st2 == j.state1:
                        dx = disp(self.pdbcontainer_si, j.state1, j.state2)
                        bias_st_solvent2 += self.W1list[jt] * dx

            bias1expansion_solute, bias1expansion_solvent = self.biases[1]
            self.assertTrue(count >= 1)
            self.assertTrue(np.allclose(bias1expansion_solute, np.zeros_like(bias1expansion_solute)),
                            msg="{}\n{}".format(bias1expansion_solute, bias1expansion_solute))
            self.assertEqual(bias1expansion_solvent.shape[1], len(self.W1list))

            # get the total bias vector
            bias1expansion_solute, bias1expansion_solvent = self.biases[1]
            tot_bias_solvent = np.dot(bias1expansion_solvent, self.W1list)
            # now get the components of the given states
            indlist = []
            # bias_cartesian = np.zeros(3)
            for ind, starlist in enumerate(self.vec_stars.vecpos[:self.vec_stars.Nvstars_pure]):
                if starlist[0] == st:
                    indlist.append(ind)

            bias_cartesian = sum([tot_bias_solvent[i] * self.vec_stars.vecvec[i][0] for i in indlist])
            bias_cartesian2 = sum([tot_bias_solvent[i] * self.vec_stars.vecvec[i][n] for i in indlist])

            self.assertTrue(np.allclose(bias_cartesian, bias_st_solvent))
            self.assertTrue(np.allclose(bias_cartesian2, bias_st_solvent2))

    def test_bias2expansions(self):

        for i in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):
            # test omega2 expansion
            st = self.vec_stars.vecpos[i][0]  # get the representative state.
            n = np.random.randint(1, len(self.vec_stars.vecpos[i]))
            st2 = self.vec_stars.vecpos[i][n]
            # Now, we calculate the total bias vector
            bias_st_solute = np.zeros(3)
            bias_st_solute2 = np.zeros(3)
            bias_st_solvent = np.zeros(3)
            bias_st_solvent2 = np.zeros(3)
            count = 0
            for jt, jlist in enumerate(self.vec_stars.starset.jnet2):
                for j in jlist:
                    if st == j.state1:
                        count += 1
                        dx = disp(self.vec_stars.starset.mdbcontainer, j.state1, j.state2)
                        dx_solute = dx + self.vec_stars.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. - \
                                    self.vec_stars.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        dx_solvent = dx - self.vec_stars.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. + \
                                     self.vec_stars.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        bias_st_solute += self.W2list[jt] * dx_solute
                        bias_st_solvent += self.W2list[jt] * dx_solvent
                    if st2 == j.state1:
                        dx = disp(self.vec_stars.starset.mdbcontainer, j.state1, j.state2)
                        dx_solute = dx + self.vec_stars.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. - \
                                    self.vec_stars.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        dx_solvent = dx - self.vec_stars.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. + \
                                     self.vec_stars.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        bias_st_solute2 += self.W2list[jt] * dx_solute
                        bias_st_solvent2 += self.W2list[jt] * dx_solvent

            bias2expansion_solute, bias2expansion_solvent = self.biases[2]
            self.assertTrue(count >= 1)
            self.assertEqual(bias2expansion_solvent.shape[1], len(self.W2list))
            # vectors
            tot_bias_solvent = np.dot(bias2expansion_solvent, self.W2list)
            tot_bias_solute = np.dot(bias2expansion_solute, self.W2list)

            # now get the components
            indlist = []
            # bias_cartesian = np.zeros(3)
            for ind, starlist in enumerate(self.vec_stars.vecpos[self.vec_stars.Nvstars_pure:]):
                if starlist[0] == st:
                    indlist.append(ind + self.vec_stars.Nvstars_pure)

            bias_cartesian_solvent = sum(
                [tot_bias_solvent[idx - self.vec_stars.Nvstars_pure] * self.vec_stars.vecvec[idx][0] for idx in
                 indlist])
            bias_cartesian_solvent2 = sum(
                [tot_bias_solvent[idx - self.vec_stars.Nvstars_pure] * self.vec_stars.vecvec[idx][n] for idx in
                 indlist])

            bias_cartesian_solute = sum(
                [tot_bias_solute[idx - self.vec_stars.Nvstars_pure] * self.vec_stars.vecvec[idx][0] for idx in indlist])
            bias_cartesian_solute2 = sum(
                [tot_bias_solute[idx - self.vec_stars.Nvstars_pure] * self.vec_stars.vecvec[idx][n] for idx in indlist])

            self.assertTrue(np.allclose(bias_cartesian_solvent, bias_st_solvent),
                            msg="{}\n{}".format(bias_cartesian_solvent,
                                                bias_st_solvent))  # should get the same bias vector anyway
            self.assertTrue(np.allclose(bias_cartesian_solvent2, bias_st_solvent2),
                            msg="{}\n{}".format(bias_cartesian_solvent2, bias_st_solvent2))

            self.assertTrue(np.allclose(bias_cartesian_solute, bias_st_solute),
                            msg="{}\n{}".format(bias_cartesian_solute,
                                                bias_st_solute))  # should get the same bias vector anyway
            self.assertTrue(np.allclose(bias_cartesian_solute2, bias_st_solute2),
                            msg="{}\n{}".format(bias_cartesian_solute2, bias_st_solute2))

    def test_bias43expansions(self):

        print(self.vec_stars.Nvstars_pure)
        print(self.vec_stars.Nvstars)
        for pureind in range(self.vec_stars.Nvstars_pure):
            for mixind in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):
                starindpure = pureind  # np.random.randint(0,self.vec_stars.Nvstars_pure)
                starindmixed = mixind  # np.random.randint(self.vec_stars.Nvstars_pure,self.vec_stars.Nvstars)

                st_pure = self.vec_stars.vecpos[starindpure][0]  # get the representative state.
                n_pure = np.random.randint(0, len(self.vec_stars.vecpos[starindpure]))
                st2_pure = self.vec_stars.vecpos[starindpure][n_pure]

                st_mixed = self.vec_stars.vecpos[starindmixed][0]  # get the representative state.
                n_mixed = np.random.randint(0, len(self.vec_stars.vecpos[starindmixed]))
                st2_mixed = self.vec_stars.vecpos[starindmixed][n_mixed]

                # Now, we calculate the total bias vector
                bias4_st_solute = np.zeros(3)
                bias4_st_solute2 = np.zeros(3)
                bias4_st_solvent = np.zeros(3)
                bias4_st_solvent2 = np.zeros(3)

                bias3_st_solute = np.zeros(3)
                bias3_st_solute2 = np.zeros(3)
                bias3_st_solvent = np.zeros(3)
                bias3_st_solvent2 = np.zeros(3)

                count = 0
                for jt, jlist in enumerate(self.symjumplist_omega4):
                    # In omega_4, the intial state should be a complex
                    for j in jlist:
                        if st_pure == j.state1:
                            count += 1
                            dx = disp4(self.vec_stars.starset.pdbcontainer, self.vec_stars.starset.mdbcontainer,
                                       j.state1, j.state2)
                            dx_solute = self.vec_stars.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                            # state2 is the mixed dumbbell.
                            dx_solvent = dx - self.vec_stars.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                            bias4_st_solute += self.W4list[jt] * dx_solute
                            bias4_st_solvent += self.W4list[jt] * dx_solvent

                        if st2_pure == j.state1:
                            dx = disp4(self.vec_stars.starset.pdbcontainer, self.vec_stars.starset.mdbcontainer,
                                       j.state1, j.state2)
                            dx_solute = self.vec_stars.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                            # state2 is the mixed dumbbell.
                            dx_solvent = dx - self.vec_stars.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                            bias4_st_solute2 += self.W4list[jt] * dx_solute
                            bias4_st_solvent2 += self.W4list[jt] * dx_solvent

                bias4expansion_solute, bias4expansion_solvent = self.biases[4]
                if st_pure.is_zero(self.vec_stars.starset.pdbcontainer):
                    # print("got origin state")
                    self.assertTrue(count == 0)
                else:
                    self.assertTrue(count >= 0)
                self.assertEqual(bias4expansion_solvent.shape[1], len(self.W4list))
                self.assertEqual(bias4expansion_solute.shape[1], len(self.W4list))
                # vectors
                tot_bias_solvent = np.dot(bias4expansion_solvent, self.W4list)
                tot_bias_solute = np.dot(bias4expansion_solute, self.W4list)

                # now get the components
                indlist = []
                # bias_cartesian = np.zeros(3)
                for ind, starlist in enumerate(self.vec_stars.vecpos[:self.vec_stars.Nvstars_pure]):
                    if starlist[0] == st_pure:
                        indlist.append(ind)

                bias_cartesian_solvent = sum([tot_bias_solvent[i] * self.vec_stars.vecvec[i][0] for i in indlist])
                bias_cartesian_solvent2 = sum([tot_bias_solvent[i] * self.vec_stars.vecvec[i][n_pure] for i in indlist])

                bias_cartesian_solute = sum([tot_bias_solute[i] * self.vec_stars.vecvec[i][0] for i in indlist])
                bias_cartesian_solute2 = sum([tot_bias_solute[i] * self.vec_stars.vecvec[i][n_pure] for i in indlist])

                self.assertTrue(np.allclose(bias_cartesian_solvent, bias4_st_solvent),
                                msg="\n{}\n{}".format(bias_cartesian_solvent, bias4_st_solvent))

                self.assertTrue(np.allclose(bias_cartesian_solvent2, bias4_st_solvent2),
                                msg="\n{}\n{}".format(bias_cartesian_solvent2, bias4_st_solvent2))

                self.assertTrue(np.allclose(bias_cartesian_solute, bias4_st_solute),
                                msg="\n{}\n{}".format(bias_cartesian_solute, bias4_st_solute))

                self.assertTrue(np.allclose(bias_cartesian_solute2, bias4_st_solute2),
                                msg="\n{}\n{}".format(bias_cartesian_solute2, bias4_st_solute2))

                count = 0
                for jt, jlist in enumerate(self.symjumplist_omega3):
                    # In omega_3, the intial state should be a mixed dumbbell
                    for j in jlist:
                        if st_mixed == j.state1:
                            count += 1
                            dx = -disp4(self.vec_stars.starset.pdbcontainer, self.vec_stars.starset.mdbcontainer,
                                        j.state2, j.state1)
                            dx_solute = -self.vec_stars.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                            dx_solvent = dx + self.vec_stars.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                            bias3_st_solute += self.W3list[jt] * dx_solute
                            bias3_st_solvent += self.W3list[jt] * dx_solvent
                        if st2_mixed == j.state1:
                            dx = -disp4(self.vec_stars.starset.pdbcontainer, self.vec_stars.starset.mdbcontainer,
                                        j.state2, j.state1)
                            dx_solute = -self.vec_stars.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                            dx_solvent = dx + self.vec_stars.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                            bias3_st_solute2 += self.W3list[jt] * dx_solute
                            bias3_st_solvent2 += self.W3list[jt] * dx_solvent

                bias3expansion_solute, bias3expansion_solvent = self.biases[3]
                self.assertTrue(count >= 1)
                self.assertEqual(bias3expansion_solvent.shape[1], len(self.W3list))
                self.assertEqual(bias3expansion_solute.shape[1], len(self.W3list))
                # vectors
                tot_bias_solvent = np.dot(bias3expansion_solvent, self.W3list)
                tot_bias_solute = np.dot(bias3expansion_solute, self.W3list)

                # now get the components
                indlist = []
                # bias_cartesian = np.zeros(3)
                for ind, starlist in enumerate(self.vec_stars.vecpos[self.vec_stars.Nvstars_pure:]):
                    if starlist[0] == st_mixed:
                        indlist.append(ind + self.vec_stars.Nvstars_pure)
                # print(indlist)
                bias_cartesian_solvent = sum(
                    [tot_bias_solvent[idx - self.vec_stars.Nvstars_pure] * self.vec_stars.vecvec[idx][0] for idx in
                     indlist])
                bias_cartesian_solvent2 = sum(
                    [tot_bias_solvent[idx - self.vec_stars.Nvstars_pure] * self.vec_stars.vecvec[idx][n_mixed] for idx
                     in
                     indlist])

                bias_cartesian_solute = sum(
                    [tot_bias_solute[idx - self.vec_stars.Nvstars_pure] * self.vec_stars.vecvec[idx][0] for idx in
                     indlist])
                bias_cartesian_solute2 = sum(
                    [tot_bias_solute[idx - self.vec_stars.Nvstars_pure] * self.vec_stars.vecvec[idx][n_mixed] for idx in
                     indlist])

                self.assertTrue(np.allclose(bias_cartesian_solvent, bias3_st_solvent),
                                msg="{}\n{}".format(bias_cartesian_solvent,
                                                    bias3_st_solvent))  # should get the same bias vector anyway
                self.assertTrue(np.allclose(bias_cartesian_solvent2, bias3_st_solvent2),
                                msg="{}\n{}".format(bias_cartesian_solvent2, bias3_st_solvent2))

                self.assertTrue(np.allclose(bias_cartesian_solute, bias3_st_solute),
                                msg="{}\n{}".format(bias_cartesian_solute,
                                                    bias3_st_solute))  # should get the same bias vector anyway
                self.assertTrue(np.allclose(bias_cartesian_solute2, bias3_st_solute2),
                                msg="{}\n{}".format(bias_cartesian_solute2, bias3_st_solute2))

    def test_rateExps(self):
        """
        Here, we will create the rate expansions in the more expensive way and check if the results hold for the
        approach used in the module
        """
        # First, we do it for omega1 and omega0
        rate0expansion = np.zeros((self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars_pure,
                                   len(self.vec_stars.starset.jnet0)))
        rate1expansion = np.zeros((self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars_pure, len(self.jnet_1)))
        rate0escape = np.zeros((self.vec_stars.Nvstars_pure, len(self.vec_stars.starset.jnet0)))
        rate1escape = np.zeros((self.vec_stars.Nvstars_pure, len(self.jnet_1)))
        # First, we do the rate1 and rate0 expansions
        for k, jumplist, jt in zip(itertools.count(), self.jnet_1, self.jtype):
            for jmp in jumplist:
                for i in range(self.vec_stars.Nvstars_pure):  # The first inner sum
                    for chi_i, vi in zip(self.vec_stars.vecpos[i], self.vec_stars.vecvec[i]):
                        if chi_i == jmp.state1:  # This is the delta functions of chi_0
                            rate0escape[i, jt] -= np.dot(vi, vi)
                            rate1escape[i, k] -= np.dot(vi, vi)
                            for j in range(self.vec_stars.Nvstars_pure):  # The second inner sum
                                for chi_j, vj in zip(self.vec_stars.vecpos[j], self.vec_stars.vecvec[j]):
                                    if chi_j == jmp.state2:  # this is the delta function of chi_1
                                        rate1expansion[i, j, k] += np.dot(vi, vj)
                                        rate0expansion[i, j, jt] += np.dot(vi, vj)

        self.assertTrue(np.allclose(rate0expansion, self.rateExps[0][0]))
        self.assertTrue(np.allclose(rate0escape, self.rateExps[0][1]))
        self.assertTrue(np.allclose(rate1expansion, self.rateExps[1][0]))
        self.assertTrue(np.allclose(rate1escape, self.rateExps[1][1]))

        # Next, we do it for omega2
        Nvstars_mixed = self.vec_stars.Nvstars - self.vec_stars.Nvstars_pure
        rate2expansion = np.zeros((Nvstars_mixed, Nvstars_mixed, len(self.vec_stars.starset.jnet2)))
        rate2escape = np.zeros((Nvstars_mixed, len(self.vec_stars.starset.jnet2)))
        # First, we do the rate1 and rate0 expansions
        for k, jumplist in enumerate(self.vec_stars.starset.jnet2):
            for jmp in jumplist:
                for i in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):  # The first inner sum
                    for chi_i, vi in zip(self.vec_stars.vecpos[i], self.vec_stars.vecvec[i]):
                        if chi_i == jmp.state1:  # This is the delta functions of chi_0
                            rate2escape[i - self.vec_stars.Nvstars_pure, k] -= np.dot(vi, vi)
                            for j in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):  # The second inner sum
                                for chi_j, vj in zip(self.vec_stars.vecpos[j], self.vec_stars.vecvec[j]):
                                    if chi_j == jmp.state2 - jmp.state2.R_s:  # this is the delta function of chi_1
                                        rate2expansion[i - self.vec_stars.Nvstars_pure, j - self.vec_stars.Nvstars_pure,
                                                       k] += np.dot(vi, vj)
        # print(rate2escape.shape, self.rateExps[2][1].shape)
        self.assertTrue(np.allclose(rate2expansion, self.rateExps[2][0]))
        self.assertTrue(np.allclose(rate2escape, self.rateExps[2][1]))

        # Next, we do it for omega3 and omega4
        rate4expansion = np.zeros((self.vec_stars.Nvstars_pure, Nvstars_mixed, len(self.symjumplist_omega43_all)))
        rate3expansion = np.zeros((Nvstars_mixed, self.vec_stars.Nvstars_pure, len(self.symjumplist_omega43_all)))
        # The initial states are mixed, the final states are complex except origin states and there are as many
        # symmetric jumps as in jumpnetwork_omega34
        rate3escape = np.zeros((Nvstars_mixed, len(self.symjumplist_omega43_all)))
        rate4escape = np.zeros((self.vec_stars.Nvstars_pure, len(self.symjumplist_omega43_all)))
        for k, jumplist in enumerate(self.symjumplist_omega43_all):
            for jmp in jumplist[::2]:
                for i in range(self.vec_stars.Nvstars_pure):  # iterate over complex states - the first inner sum
                    for chi_i, vi in zip(self.vec_stars.vecpos[i], self.vec_stars.vecvec[i]):
                        # Go through the initial pure states
                        if chi_i == jmp.state1:
                            rate4escape[i, k] -= np.dot(vi, vi)
                            for j in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):
                                # iterate over mixed states - the second inner sum
                                for chi_j, vj in zip(self.vec_stars.vecpos[j], self.vec_stars.vecvec[j]):
                                    # Go through the final complex states - they must be at the origin unit cell.
                                    self.assertTrue(np.allclose(jmp.state2.R_s, np.zeros(3, dtype=int)))
                                    if chi_j == jmp.state2:
                                        rate3escape[j - self.vec_stars.Nvstars_pure, k] -= np.dot(vj, vj)
                                        rate4expansion[i, j - self.vec_stars.Nvstars_pure, k] += np.dot(vi, vj)
                                        rate3expansion[j - self.vec_stars.Nvstars_pure, i, k] += np.dot(vj, vi)

        self.assertFalse(np.allclose(rate3expansion, np.zeros_like(rate3expansion)))

        self.assertTrue(np.allclose(rate3expansion, self.rateExps[3][0]))
        self.assertTrue(np.allclose(rate3escape, self.rateExps[3][1]))
        self.assertTrue(np.allclose(rate4escape, self.rateExps[4][1]))
        self.assertTrue(np.allclose(rate4expansion, self.rateExps[4][0]))

    def test_tags(self):
        """
        See that the arrays tagging the jumps are produced properly
        """
        # First let us go through the omega1 jump network.
        for jt, jlist, jindlist in zip(itertools.count(), self.jnet_1, self.jnet_1_indexed):
            # indDictlist = self.om1tags[jt]
            count_dict = defaultdict(int)
            for (i, j), dx in jindlist:
                count_dict[i] += 1
            for key, arr in self.om1tags[jt].items():
                self.assertEqual(len(arr), count_dict[key])

        for jt, jlist, jindlist in zip(itertools.count(), self.vec_stars.starset.jnet2,
                                       self.vec_stars.starset.jnet2_ind):
            # indDictlist = self.om1tags[jt]
            count_dict = defaultdict(int)
            for (i, j), dx in jindlist:
                count_dict[i] += 1
            for key, arr in self.om2tags[jt].items():
                self.assertEqual(len(arr), count_dict[key])

        for jt, jlist, jindlist in zip(itertools.count(), self.symjumplist_omega4, self.symjumplist_omega4_indexed):
            # indDictlist = self.om1tags[jt]
            count_dict = defaultdict(int)
            for (i, j), dx in jindlist:
                count_dict[i] += 1
            for key, arr in self.om4tags[jt].items():
                self.assertEqual(len(arr), count_dict[key])

        for jt, jlist, jindlist in zip(itertools.count(), self.symjumplist_omega3, self.symjumplist_omega3_indexed):
            # indDictlist = self.om1tags[jt]
            count_dict = defaultdict(int)
            for (i, j), dx in jindlist:
                count_dict[i] += 1
            for key, arr in self.om3tags[jt].items():
                self.assertEqual(len(arr), count_dict[key])

    def test_GFstars(self):
        # Check that every possible pair has been considered in the gfstarsets
        GFstarset_pure, GFPureStarInd, GFstarset_mixed, GFMixedStarInd = self.vec_stars.genGFstarset()

        # Check that every state in the GFstarsets is present as keys in in the starinds
        sm = sum([len(star) for star in GFstarset_pure])
        self.assertEqual(sm, len(GFPureStarInd))

        sm = sum([len(star) for star in GFstarset_mixed])
        self.assertEqual(sm, len(GFMixedStarInd))

        # First for the complex states
        for st1 in self.vec_stars.starset.complexStates:
            for st2 in self.vec_stars.starset.complexStates:
                try:
                    s = st1 ^ st2
                    # s.shift()
                except:
                    continue
                dx = disp(self.vec_stars.starset.pdbcontainer, s.state1, s.state2)
                ind1 = self.vec_stars.starset.pdbcontainer.db2ind(s.state1)
                ind2 = self.vec_stars.starset.pdbcontainer.db2ind(s.state2)
                self.assertFalse(ind1 is None)
                self.assertFalse(ind2 is None)
                snewlist = []
                listind = None
                count = 0
                for star, tlist in enumerate(GFstarset_pure):
                    for tup in tlist:
                        if tup[0][0] == ind1 and tup[0][1] == ind2:
                            if np.allclose(tup[1], dx, atol=self.vec_stars.starset.crys.threshold):
                                listind = star
                                count += 1
                self.assertTrue(listind is not None)
                self.assertEqual(listind, GFPureStarInd[s])
                self.assertEqual(count, 1)
                # Now check symmetries
                # Now build up the symmetric list
                for gdumb in self.vec_stars.starset.pdbcontainer.G:
                    ind1new = gdumb.indexmap[0][ind1]
                    ind2new = gdumb.indexmap[0][ind2]
                    dxnew = self.vec_stars.starset.crys.g_direc(self.vec_stars.starset.pdbcontainer.G_crys[gdumb], dx)
                    if not any(
                            ind1new == t[0][0] and ind2new == t[0][1] and np.allclose(dxnew, t[1]) for t in snewlist):
                        snewlist.append(((ind1new, ind2new), dxnew))
                self.assertEqual(len(snewlist), len(GFstarset_pure[listind]))
                count = 0
                for s in snewlist:
                    for s2 in GFstarset_pure[listind]:
                        if s[0][0] == s2[0][0] and s[0][1] == s2[0][1] and np.allclose(s[1], s2[1]):
                            count += 1
                self.assertEqual(count, len(GFstarset_pure[listind]), msg="\n{}\n{}".format(snewlist,
                                                                                            GFstarset_pure[listind]))

        # Now, we do the same tests for the mixed GF starset
        for jlist in self.vec_stars.starset.jnet2:
            for jmp in jlist:
                s = connector(jmp.state1.db, jmp.state2.db)
                s.shift()
                ind1 = self.vec_stars.starset.mdbcontainer.db2ind(s.state1)
                ind2 = self.vec_stars.starset.mdbcontainer.db2ind(s.state2)
                dx = disp(self.vec_stars.starset.mdbcontainer, s.state1, s.state2)
                # Now see if this connection is present in the GF starset
                listind = None
                count = 0
                for starind, tlist in enumerate(GFstarset_mixed):
                    for t in tlist:
                        if t[0][0] == ind1 and t[0][1] == ind2 and np.allclose(t[1], dx, atol=1e-8):
                            count += 1
                            listind = starind

                self.assertTrue(listind is not None)
                self.assertEqual(listind, GFMixedStarInd[s])
                self.assertEqual(count, 1)

                # Now check the symmetries
                snewlist = []
                for gdumb in self.vec_stars.starset.mdbcontainer.G:
                    ind1new = gdumb.indexmap[0][ind1]
                    ind2new = gdumb.indexmap[0][ind2]
                    dxnew = self.vec_stars.starset.crys.g_direc(self.vec_stars.starset.mdbcontainer.G_crys[gdumb], dx)
                    if not any(t[0][0] == ind1new and t[0][1] == ind2new and np.allclose(t[1], dxnew, atol=1e-8)
                               for t in snewlist):
                        snewlist.append(((ind1new, ind2new), dxnew))

                self.assertEqual(len(snewlist), len(GFstarset_mixed[listind]))
                count = 0
                for s1 in snewlist:
                    for s2 in GFstarset_mixed[listind]:
                        if s1[0][0] == s2[0][0] and s1[0][1] == s2[0][1] and np.allclose(s1[1], s2[1], atol=1e-8):
                            count += 1

                self.assertEqual(count, len(GFstarset_mixed[listind]))

    def test_GF_expansions(self):
        """
        Here we will test with the slow but sure version of GFexpansion
        """

        def getstar(tup, star):
            for starind, list in enumerate(star):
                for t in list:
                    if (t[0][0] == tup[0][0] and t[0][1] == tup[0][1] and
                            np.allclose(t[1], tup[1], atol=self.vec_stars.starset.crys.threshold)):
                        return starind
            return None

        # First, pure GF expansion
        GFstarset_pure, GFPureStarInd, GFstarset_mixed, GFMixedStarInd = self.vec_stars.genGFstarset()

        Nvstars_pure = self.vec_stars.Nvstars_pure
        Nvstars_mixed = self.vec_stars.Nvstars - self.vec_stars.Nvstars_pure
        GFexpansion_pure = np.zeros((Nvstars_pure, Nvstars_pure, len(GFstarset_pure)))
        GFexpansion_mixed = np.zeros((Nvstars_mixed, Nvstars_mixed, len(GFstarset_mixed)))
        GFexpansion_pure_calc, GFexpansion_mixed_calc = self.vec_stars.GFexpansion()
        # build up the pure GFexpansion
        for i in range(Nvstars_pure):
            for si, vi in zip(self.vec_stars.vecpos[i], self.vec_stars.vecvec[i]):
                for j in range(Nvstars_pure):
                    for sj, vj in zip(self.vec_stars.vecpos[j], self.vec_stars.vecvec[j]):
                        try:
                            ds = si ^ sj
                        except:
                            continue
                        # print(ds)
                        ds.shift()
                        dx = disp(self.vec_stars.starset.pdbcontainer, ds.state1, ds.state2)
                        ind1 = self.vec_stars.starset.pdbcontainer.db2ind(ds.state1)
                        ind2 = self.vec_stars.starset.pdbcontainer.db2ind(ds.state2)
                        if ind1 is None or ind2 is None:
                            raise KeyError("enpoint subtraction within starset not found in iorlist")
                        k = getstar(((ind1, ind2), dx), GFstarset_pure)
                        k1 = GFPureStarInd[ds]
                        # print("here")
                        self.assertEqual(k, k1)
                        if k is None:
                            raise ArithmeticError(
                                "complex GF starset not big enough to accomodate state pair {}".format(tup))
                        GFexpansion_pure[i, j, k] += np.dot(vi, vj)
        # symmetrize
        for i in range(Nvstars_pure):
            for j in range(0, i):
                GFexpansion_pure[i, j, :] = GFexpansion_pure[j, i, :]

        self.assertTrue(np.allclose(zeroclean(GFexpansion_pure), GFexpansion_pure_calc[2]))

        for jlist in self.vec_stars.starset.jnet2:
            for jmp in jlist:

                ds = connector(jmp.state1.db, jmp.state2.db)
                ds.shift()

                ind1, ind2 = self.vec_stars.starset.mdbcontainer.db2ind(ds.state1), \
                             self.vec_stars.starset.mdbcontainer.db2ind(ds.state2)
                dx = disp(self.vec_stars.starset.mdbcontainer, ds.state1, ds.state2)

                si = jmp.state1 - jmp.state1.R_s
                sj = jmp.state2 - jmp.state2.R_s

                # Now get the vector stars for the states
                viList = self.vec_stars.stateToVecStar_mixed[si]  # (IndOfStar, IndOfState) format
                vjList = self.vec_stars.stateToVecStar_mixed[sj]  # (IndOfStar, IndOfState) format
                k = getstar(((ind1, ind2), dx), GFstarset_mixed)
                k1 = GFMixedStarInd[ds]
                self.assertEqual(k, k1, msg="\n{}\n{}\n{}\n{}".format(k, ((ind1, ind2), dx), k1, ds))

                for tup in viList:
                    self.assertTrue(self.vec_stars.vecpos[tup[0]][tup[1]] == si)
                for tup in vjList:
                    self.assertTrue(self.vec_stars.vecpos[tup[0]][tup[1]] == sj)

                # print(viList)
                # print(vjList)
                # print(self.vec_stars.Nvstars_pure)
                # print(self.vec_stars.Nvstars)

                for i, vi in [(tup[0] - self.vec_stars.Nvstars_pure,
                               self.vec_stars.vecvec[tup[0]][tup[1]]) for tup in viList]:
                    for j, vj in [(tup[0] - self.vec_stars.Nvstars_pure,
                                   self.vec_stars.vecvec[tup[0]][tup[1]]) for tup in viList]:
                        GFexpansion_mixed[i, j, k] += np.dot(vi, vj)

        for st in self.vec_stars.starset.mixedstates:
            ds = connector(st.db, st.db)
            k = GFMixedStarInd[ds]

            ind1 = self.vec_stars.starset.mdbcontainer.db2ind(ds.state1)
            dx = np.zeros(3)  # No displacement for the diagonals
            k1 = getstar(((ind1, ind1), dx), GFstarset_mixed)

            self.assertEqual(k, k1)

            vList = self.vec_stars.stateToVecStar_mixed[st]
            for i, vi in [(tup[0] - self.vec_stars.Nvstars_pure,
                           self.vec_stars.vecvec[tup[0]][tup[1]]) for tup in vList]:
                GFexpansion_mixed[i, i, k] += np.dot(vi, vj)

        for i in range(Nvstars_mixed):
            for j in range(0, i):
                GFexpansion_mixed[i, j, :] = GFexpansion_mixed[j, i, :]

        self.assertTrue(np.allclose(zeroclean(GFexpansion_mixed), GFexpansion_mixed_calc[2]))

    def test_order(self):
        "test that we have the origin spectator states at the begining"
        dx_list = []
        for sts in zip(self.vec_stars.vecpos[:self.vec_stars.Nvstars_pure]):
            st0 = sts[0][0]
            sol_pos = self.vec_stars.starset.crys.unit2cart(st0.R_s, self.vec_stars.starset.crys.basis[
                self.vec_stars.starset.chem][st0.i_s])
            db_pos = self.vec_stars.starset.crys.unit2cart(st0.db.R, self.vec_stars.starset.crys.basis[
                self.vec_stars.starset.chem][self.vec_stars.starset.pdbcontainer.iorlist[st0.db.iorind][0]])
            dx = np.linalg.norm(db_pos - sol_pos)
            dx_list.append(dx)
        self.assertTrue(np.allclose(zeroclean(np.array(dx_list)), zeroclean(np.array(sorted(dx_list)))),
                        msg="\n{}\n{}".format(dx_list, sorted(dx_list)))


class test_Si(unittest.TestCase):

    def setUp(self):
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        self.DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1., 1., 0.]) / np.linalg.norm(np.array([1., 1., 0.])) * 0.126
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = dbStates(self.DC_Si, 0, family)
        self.mdbcontainer_si = mStates(self.DC_Si, 0, family)

        self.jset0, self.jset2 = self.pdbcontainer_si.jumpnetwork(0.3, 0.01, 0.01), self.mdbcontainer_si.jumpnetwork(
            0.3, 0.01, 0.01)

        self.crys_stars = StarSet(self.pdbcontainer_si, self.mdbcontainer_si, self.jset0, self.jset2, Nshells=1)
        self.vec_stars = vectorStars(self.crys_stars)

        with open('vec_star_si_pkl.pkl', 'wb') as fl:
            pickle.dump(self.vec_stars, fl)

        self.om2tags = self.vec_stars.starset.jtags2
        # generate 1, 3 and 4 jumpnetworks
        (self.jnet_1, self.jnet_1_indexed, self.om1tags), self.jtype = self.crys_stars.jumpnetwork_omega1()
        (self.symjumplist_omega43_all, self.symjumplist_omega43_all_indexed), (
            self.symjumplist_omega4, self.symjumplist_omega4_indexed, self.om4tags), (
            self.symjumplist_omega3, self.symjumplist_omega3_indexed,
            self.om3tags) = self.crys_stars.jumpnetwork_omega34(
            0.3, 0.01, 0.01, 0.01)

        self.W0list = np.random.rand(len(self.vec_stars.starset.jnet0))
        self.W1list = np.random.rand(len(self.jnet_1))
        self.W2list = np.random.rand(len(self.jset2[0]))
        self.W3list = np.random.rand(len(self.symjumplist_omega3))
        self.W4list = np.random.rand(len(self.symjumplist_omega4))

        # generate all the bias expansions - will separate out later
        self.biases = self.vec_stars.biasexpansion(self.jnet_1, self.jset2[0], self.jtype, self.symjumplist_omega43_all)
        self.rateExps = self.vec_stars.rateexpansion(self.jnet_1, self.jtype, self.symjumplist_omega43_all)
        print("Instantiated")

    def test_bare_bias_expansion(self):
        if len(self.vec_stars.vecpos_bare) > 0:
            for i in range(len(self.vec_stars.vecpos_bare)):
                starind = i
                st = self.vec_stars.vecpos_bare[starind][0]
                n = np.random.randint(1, len(self.vec_stars.vecpos_bare[starind]))
                st2 = self.vec_stars.vecpos_bare[starind][n]
                bias_st = np.zeros(3)
                bias_st2 = np.zeros(3)
                count = 0
                for jt, jumplist in enumerate(self.vec_stars.starset.jnet0):
                    for j in jumplist:
                        if st == j.state1:
                            count += 1
                            dx = disp(self.pdbcontainer_si, j.state1, j.state2)
                            bias_st += dx * self.W0list[jt]
                        if st2 == j.state1:
                            dx = disp(self.pdbcontainer_si, j.state1, j.state2)
                            bias_st2 += dx * self.W0list[jt]

                biasBareExp = self.biases[-1]
                self.assertTrue(count >= 1)
                tot_bias_bare = np.dot(biasBareExp, self.W0list)
                indlist = self.vec_stars.stateToVecStar_bare[st]
                # for ind, starlist in enumerate(self.vec_stars.vecpos_bare):
                #     if starlist[0] == st:
                #         indlist.append(ind)
                bias_bare_cartesian = sum(
                    [tot_bias_bare[tup[0]] * self.vec_stars.vecvec_bare[tup[0]][tup[1]] for tup in indlist])
                self.assertTrue(np.allclose(bias_bare_cartesian, bias_st), msg="\n{}\n{}\n{}\n{}".
                                format(bias_bare_cartesian, bias_st,
                                       [self.vec_stars.vecvec_bare[tup[0]][tup[1]] for tup in indlist],
                                       self.vec_stars.starset.pdbcontainer.iorlist[st.iorind]))

                indlist = self.vec_stars.stateToVecStar_bare[st2]
                bias_bare_cartesian2 = sum(
                    [tot_bias_bare[tup[0]] * self.vec_stars.vecvec_bare[tup[0]][tup[1]] for tup in indlist])
                self.assertTrue(np.allclose(bias_bare_cartesian2, bias_st2))

        else:  # we have to check that the non-local bias vectors coming out are zero
            # print("checking zero non-local")
            for star in self.vec_stars.starset.barePeriodicStars:
                for st in star:
                    bias_st = np.zeros(3)
                    count = 0
                    for jt, jumplist in enumerate(self.vec_stars.starset.jnet0):
                        for j in jumplist:
                            if st == j.state1:
                                count += 1
                                dx = disp(self.pdbcontainer_si, j.state1, j.state2)
                                bias_st += dx * self.W0list[jt]
                                # print(bias_st)
                    self.assertTrue(np.allclose(bias_st, np.zeros(3)))
                    self.assertTrue(count >= 1)

    def test_rateExps(self):
        """
        Here, we will create the rate expansions in the more expensive way and check if the results hold for the
        approach used in the module
        """
        # First, we do it for omega1 and omega0
        rate0expansion = np.zeros((self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars_pure,
                                   len(self.vec_stars.starset.jnet0)))
        rate1expansion = np.zeros((self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars_pure, len(self.jnet_1)))
        rate0escape = np.zeros((self.vec_stars.Nvstars_pure, len(self.vec_stars.starset.jnet0)))
        rate1escape = np.zeros((self.vec_stars.Nvstars_pure, len(self.jnet_1)))
        # First, we do the rate1 and rate0 expansions
        for k, jumplist, jt in zip(itertools.count(), self.jnet_1, self.jtype):
            for jmp in jumplist:
                for i in range(self.vec_stars.Nvstars_pure):  # The first inner sum
                    for chi_i, vi in zip(self.vec_stars.vecpos[i], self.vec_stars.vecvec[i]):
                        if chi_i == jmp.state1:  # This is the delta functions of chi_0
                            rate0escape[i, jt] -= np.dot(vi, vi)
                            rate1escape[i, k] -= np.dot(vi, vi)
                            for j in range(self.vec_stars.Nvstars_pure):  # The second inner sum
                                for chi_j, vj in zip(self.vec_stars.vecpos[j], self.vec_stars.vecvec[j]):
                                    if chi_j == jmp.state2:  # this is the delta function of chi_1
                                        rate1expansion[i, j, k] += np.dot(vi, vj)
                                        rate0expansion[i, j, jt] += np.dot(vi, vj)

        self.assertTrue(np.allclose(rate0expansion, self.rateExps[0][0]))
        self.assertTrue(np.allclose(rate0escape, self.rateExps[0][1]))
        self.assertTrue(np.allclose(rate1expansion, self.rateExps[1][0]))
        self.assertTrue(np.allclose(rate1escape, self.rateExps[1][1]))

        # Next, we do it for omega2
        Nvstars_mixed = self.vec_stars.Nvstars - self.vec_stars.Nvstars_pure
        rate2expansion = np.zeros((Nvstars_mixed, Nvstars_mixed, len(self.vec_stars.starset.jnet2)))
        rate2escape = np.zeros((Nvstars_mixed, len(self.vec_stars.starset.jnet2)))
        # First, we do the rate1 and rate0 expansions
        for k, jumplist in enumerate(self.vec_stars.starset.jnet2):
            for jmp in jumplist:
                for i in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):  # The first inner sum
                    for chi_i, vi in zip(self.vec_stars.vecpos[i], self.vec_stars.vecvec[i]):
                        if chi_i == jmp.state1:  # This is the delta functions of chi_0
                            rate2escape[i - self.vec_stars.Nvstars_pure, k] -= np.dot(vi, vi)
                            for j in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):  # The second inner sum
                                for chi_j, vj in zip(self.vec_stars.vecpos[j], self.vec_stars.vecvec[j]):
                                    if chi_j == jmp.state2 - jmp.state2.R_s:  # this is the delta function of chi_1
                                        rate2expansion[i - self.vec_stars.Nvstars_pure, j - self.vec_stars.Nvstars_pure,
                                                       k] += np.dot(vi, vj)

        self.assertTrue(np.allclose(rate2expansion, self.rateExps[2][0]))
        self.assertTrue(np.allclose(rate2escape, self.rateExps[2][1]))

        # Next, we do it for omega3 and omega4
        rate4expansion = np.zeros((self.vec_stars.Nvstars_pure, Nvstars_mixed, len(self.symjumplist_omega43_all)))
        rate3expansion = np.zeros((Nvstars_mixed, self.vec_stars.Nvstars_pure, len(self.symjumplist_omega43_all)))
        # The initial states are mixed, the final states are complex except origin states and there are as many
        # symmetric jumps as in jumpnetwork_omega34
        rate3escape = np.zeros((Nvstars_mixed, len(self.symjumplist_omega43_all)))
        rate4escape = np.zeros((self.vec_stars.Nvstars_pure, len(self.symjumplist_omega43_all)))
        for k, jumplist in enumerate(self.symjumplist_omega43_all):
            for jmp in jumplist[::2]:
                for i in range(self.vec_stars.Nvstars_pure):  # iterate over complex states - the first inner sum
                    for chi_i, vi in zip(self.vec_stars.vecpos[i], self.vec_stars.vecvec[i]):
                        # Go through the initial pure states
                        if chi_i == jmp.state1:
                            rate4escape[i, k] -= np.dot(vi, vi)
                            for j in range(self.vec_stars.Nvstars_pure, self.vec_stars.Nvstars):
                                # iterate over mixed states - the second inner sum
                                for chi_j, vj in zip(self.vec_stars.vecpos[j], self.vec_stars.vecvec[j]):
                                    # Go through the final complex states - they must be at the origin unit cell.
                                    self.assertTrue(np.allclose(jmp.state2.R_s, np.zeros(3, dtype=int)))
                                    if chi_j == jmp.state2:
                                        rate3escape[j - self.vec_stars.Nvstars_pure, k] -= np.dot(vj, vj)
                                        rate4expansion[i, j - self.vec_stars.Nvstars_pure, k] += np.dot(vi, vj)
                                        rate3expansion[j - self.vec_stars.Nvstars_pure, i, k] += np.dot(vj, vi)

        self.assertFalse(np.allclose(rate3expansion, np.zeros_like(rate3expansion)))

        self.assertTrue(np.allclose(rate3expansion, self.rateExps[3][0]))
        self.assertTrue(np.allclose(rate3escape, self.rateExps[3][1]))
        self.assertTrue(np.allclose(rate4escape, self.rateExps[4][1]))
        self.assertTrue(np.allclose(rate4expansion, self.rateExps[4][0]))
