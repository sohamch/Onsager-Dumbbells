import onsager.crystal as crystal
from Onsager_calc_db import *
from test_structs import *
import states
from representations import *
import unittest
from stars import *
from vector_stars import *

class test_dumbbell_mediated(unittest.TestCase):
    def setUp(self):
        latt = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])*0.55
        self.DC_Si = crystal.Crystal(latt,[[np.array([0.,0.,0.]),np.array([0.25,0.25,0.25])]],["Si"])
        self.cube = cube
        #keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = dbStates(self.DC_Si,0,family)
        self.mdbcontainer_si = mStates(self.DC_Si,0,family)

        # self.jset0,self.jset2 = self.pdbcontainer_si.jumpnetwork(0.4,0.01,0.01), self.mdbcontainer_si.jumpnetwork(0.4,0.01,0.01)
        #
        # self.crys_stars = StarSet(self.pdbcontainer_si,self.mdbcontainer_si,self.jset0,self.jset2, Nshells=1)
        # self.vec_stars = vectorStars(self.crys_stars)
        #
        # #generate 1, 3 and 4 jumpnetworks
        # (self.jnet_1,self.jnet_1_indexed), self.jtype = self.crys_stars.jumpnetwork_omega1()
        # (self.symjumplist_omega43_all,self.symjumplist_omega43_all_indexed),(self.symjumplist_omega4,self.symjumplist_omega4_indexed),(self.symjumplist_omega3,self.symjumplist_omega3_indexed)=self.crys_stars.jumpnetwork_omega34(0.4,0.01,0.01,0.01)

        # self.W1list = np.ones(len(self.jnet_1))
        # self.W2list = np.ones(len(self.jset2[0]))
        # self.W3list = np.ones(len(self.symjumplist_omega3))
        # self.W4list = np.ones(len(self.symjumplist_omega4))

        #generate all the bias expansions - will separate out later
        #Previously, all the jumplists were created as arrays of ones.
        #To make things more concrete, we initialize them with random numbers
        self.onsagercalculator=dumbbellMediated(self.pdbcontainer_si,self.mdbcontainer_si,0.3,0.01,0.01,0.01,NGFmax=4,Nthermo=1)
        self.W1list = np.random.rand(len(self.onsagercalculator.jnet_1))
        self.W2list = np.random.rand(len(self.onsagercalculator.jnet0))
        self.W3list = np.random.rand(len(self.onsagercalculator.symjumplist_omega3))
        self.W4list = np.random.rand(len(self.onsagercalculator.symjumplist_omega4))
        self.biases =\
        self.onsagercalculator.vkinetic.biasexpansion(self.onsagercalculator.jnet_1,self.onsagercalculator.jnet2,self.onsagercalculator.om1types,self.onsagercalculator.symjumplist_omega43_all)

#     def test_displacements(self):
#
#         #Set up a simple jumpnetwork of three states in a cubic lattice
#         famp0 = [np.array([0.126,0.,0.])]
#         family = [famp0]
#         pdbcontainer = states.dbStates(cube,0,family)
#         mdbcontainer = states.mStates(cube,0,family)
#         jnet0_states,jnet_0_ind = pdbcontainer.jumpnetwork(0.3,0.01,0.01)
#         jnet2_states,jnet_2_ind = mdbcontainer.jumpnetwork(0.3,0.01,0.01)
#
#         jnet0_solute, jnet0_solvent = calc_dx_species(cube,jnet0_states,jnet_0_ind,type="bare")
#         jnet2_solute, jnet2_solvent = calc_dx_species(cube,jnet2_states,jnet_2_ind,type="mixed")
#
#         #We'll test if we have the correct displacement for the following jump, which should be there
# #         Jump object:
# #           Initial state:
# # 	             dumbbell :basis index = 0, lattice vector = [0 0 0], orientation = [0. 1. 0.]
# #           Final state:
# # 	             dumbbell :basis index = 0, lattice vector = [0 0 1], orientation = [0. 1. 0.]
# #                Jumping from c = -1 to c= -1
#         db1 = dumbbell(0,np.array([0.,.126,0.]),np.array([0,0,0],dtype=int))
#         db2 = dumbbell(0,np.array([.126,0.,0.]),np.array([0,0,1],dtype=int))
#         jmp_test = jump(db1,db2,-1,-1)
#         dx_test = np.array([0.0,0.0,0.28])
#         # dx_solvent = None
#         for i,jlist in enumerate(jnet0_states):
#             for j,jmp in enumerate(jlist):
#                 if  np.allclose(jmp_test.state1.o*jmp_test.c1,jmp.state1.o*jmp.c1):
#                     if  np.allclose(jmp_test.state2.o*jmp_test.c2,jmp.state2.o*jmp.c2):
#                         if np.allclose(jmp.state1.R,jmp_test.state1.R):
#                             if np.allclose(jmp.state2.R,jmp_test.state2.R):
#                                 dx_solvent = jnet0_solvent[i][j][1]                                #
#         # self.assertFalse(dx_solvent==None)
#         self.assertTrue(np.allclose(dx_solvent,dx_test),msg="dx_solvent,dx_test = {}, {}".format(dx_solvent,dx_test))
#         # self.assertTrue(jnet0_solute==None)

    def test_calc_eta(self):
        #We test a new weird lattice because it is more interesting
        latt = np.array([[0.,0.1,0.5],[0.3,0.,0.5],[0.5,0.5,0.]])*0.55
        self.DC_Si = crystal.Crystal(latt,[[np.array([0.,0.,0.]),np.array([0.25,0.25,0.25])]],["Si"])
        self.cube = cube
        #keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = dbStates(self.DC_Si,0,family)
        self.mdbcontainer_si = mStates(self.DC_Si,0,family)

        #generate all the bias expansions - will separate out later
        #Previously, all the jumplists were created as arrays of ones.
        #To make things more concrete, we initialize them with random numbers
        self.onsagercalculator=dumbbellMediated(self.pdbcontainer_si,self.mdbcontainer_si,0.3,0.01,0.01,0.01,NGFmax=4,Nthermo=1)
        # self.W1list = np.random.rand(len(self.onsagercalculator.jnet_1))
        # self.W2list = np.random.rand(len(self.onsagercalculator.jnet0))
        # self.W3list = np.random.rand(len(self.onsagercalculator.symjumplist_omega3))
        # self.W4list = np.random.rand(len(self.onsagercalculator.symjumplist_omega4))
        self.biases =\
        self.onsagercalculator.vkinetic.biasexpansion(self.onsagercalculator.jnet_1,self.onsagercalculator.jnet2,self.onsagercalculator.om1types,self.onsagercalculator.symjumplist_omega43_all)
        #set up the pre-factors and energies for rate calculations
        pre0 = np.random.rand(len(self.onsagercalculator.pdbcontainer.symorlist))
        betaene0 = np.random.rand(len(self.onsagercalculator.pdbcontainer.symorlist))
        pre0T = np.random.rand(len(self.onsagercalculator.jnet0))
        betaene0T = np.random.rand(len(self.onsagercalculator.jnet0))
        pre2 = np.random.rand(len(self.onsagercalculator.mdbcontainer.symorlist))
        betaene2 = np.random.rand(len(self.onsagercalculator.mdbcontainer.symorlist))
        pre2T = np.random.rand(len(self.onsagercalculator.jnet2))
        betaene2T = np.random.rand(len(self.onsagercalculator.jnet2))

        rate0list = ratelist(self.onsagercalculator.jnet0_indexed, pre0, betaene0, pre0T, betaene0T, self.onsagercalculator.vkinetic.starset.pdbcontainer.invmap)
        rate2list = ratelist(self.onsagercalculator.jnet2_indexed, pre2, betaene2, pre2T, betaene2T, self.onsagercalculator.vkinetic.starset.mdbcontainer.invmap)
        # (eta00_solvent,eta00_solute), (eta02_solvent,eta02_solute) = \
        self.onsagercalculator.calc_eta(pre0, betaene0, pre0T, betaene0T, pre2, betaene2, pre2T, betaene2T)

        self.assertEqual(len(self.onsagercalculator.eta00_solvent),len(self.onsagercalculator.vkinetic.starset.purestates))
        self.assertEqual(len(self.onsagercalculator.eta00_solute),len(self.onsagercalculator.vkinetic.starset.purestates))

        self.assertEqual(len(self.onsagercalculator.eta02_solvent),len(self.onsagercalculator.vkinetic.starset.mixedstates))
        self.assertEqual(len(self.onsagercalculator.eta02_solute),len(self.onsagercalculator.vkinetic.starset.mixedstates))

        if len(self.onsagercalculator.vkinetic.vecpos_bare)==0:
            self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent,np.zeros((len(self.onsagercalculator.vkinetic.starset.purestates),3))))
            self.assertTrue(np.allclose(self.onsagercalculator.eta00_solute,np.zeros((len(self.onsagercalculator.vkinetic.starset.purestates),3))))

        else:
            #The non-local pure dumbbell biases have been tested in test_vecstars.
            #Here, we check if for periodic dumbbells, we have the same solvent bias.
            for i in range(len(self.onsagercalculator.vkinetic.starset.purestates)):
                for j in range(len(self.onsagercalculator.vkinetic.starset.purestates)):
                    if self.onsagercalculator.vkinetic.starset.purestates[i].db.i == self.onsagercalculator.vkinetic.starset.purestates[j].db.i:
                        if np.allclose(self.onsagercalculator.vkinetic.starset.purestates[i].db.o,self.onsagercalculator.vkinetic.starset.purestates[j].db.o):
                            self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent[i,:],self.onsagercalculator.eta00_solvent[j,:]))
                            self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent[i,:],self.onsagercalculator.eta00_solvent_bare[self.onsagercalculator.vkinetic.starset.bareindexdict[self.onsagercalculator.vkinetic.starset.purestates[i].db - self.onsagercalculator.vkinetic.starset.purestates[i].db.R][0]]))

            #Check that we get the correct non-local bias vector
            for i in range(len(self.onsagercalculator.vkinetic.starset.bareStates)):
                bias_true = self.onsagercalculator.NlsolventBias_bare[i,:]
                bias_calc = np.zeros(3)
                for jt,jlist in enumerate(self.onsagercalculator.jnet0_indexed):
                    for (IS,FS),dx in jlist:
                        if i==IS:
                            bias_calc += rate0list[jt][0]*dx
                self.assertTrue(np.allclose(bias_calc,bias_true))

            #Now we check if the eta vectors are okay.
            #The above test confirms that our NlsolventBias_bare is correct
            for i in range(len(self.onsagercalculator.vkinetic.starset.bareStates)):
                bias_test = np.zeros(3)
                db = self.onsagercalculator.vkinetic.starset.bareStates[i]
                #First, let's check the indexing
                self.assertEqual(i,self.onsagercalculator.vkinetic.starset.bareindexdict[db][0])
                bias_true = self.onsagercalculator.NlsolventBias_bare[i,:]
                # print(len(self.onsagercalculator.vkinetic.vecpos_bare[self.onsagercalculator.vkinetic.bareStTobareStar[db][0][0]]))
                # eta_true *= len(self.onsagercalculator.vkinetic.vecpos_bare[self.onsagercalculator.vkinetic.bareStTobareStar[db][0][0]])
                for jt,jindlist,jlist in zip(itertools.count(),self.onsagercalculator.jnet0_indexed,self.onsagercalculator.jnet0):
                    for ((IS,FS),dx),jmp in zip(jindlist,jlist):
                        if i==IS:
                            self.assertTrue(db==jmp.state1)
                            bias_test += rate0list[jt][0]*(self.onsagercalculator.eta00_solvent_bare[FS,:]-self.onsagercalculator.eta00_solvent_bare[IS,:])
                self.assertTrue(np.allclose(bias_test,bias_true),msg="{}{}".format(bias_test,bias_true))


            for i in range(len(self.onsagercalculator.eta00_solvent_bare)):
                #get the indices of the state
                st = self.onsagercalculator.vkinetic.starset.bareStates[i]
                #next get the vectorstar,state indices
                indlist = self.onsagercalculator.vkinetic.bareStTobareStar[st]
                #The IndOfStar in indlist for the mixed case is already shifted by the number of pure vector stars.
                if len(indlist)!=0:
                    vlist=[]
                    for tup in indlist:
                        vlist.append(self.onsagercalculator.vkinetic.vecvec_bare[tup[0]][tup[1]])
                    eta_test_solvent = sum([np.dot(self.onsagercalculator.eta00_solvent_bare[i,:],v)*v for v in vlist])
                    # eta_test_solute = sum([np.dot(self.onsagercalculator.eta02_solute[i,:],v)*v for v in vlist])
                    # print(vlist)
                    self.assertTrue(np.allclose(eta_test_solvent*len(self.onsagercalculator.vkinetic.vecvec_bare[indlist[0][0]]),self.onsagercalculator.eta00_solvent_bare[i]),msg="{} {}".format(eta_test_solvent,self.onsagercalculator.eta00_solvent_bare[i]))
                    # self.assertTrue(np.allclose(eta_test_solute*len(self.onsagercalculator.vkinetic.vecvec[indlist[0][0]]),self.onsagercalculator.eta02_solute[i]),msg="{} {}".format(eta_test_solute,self.onsagercalculator.eta02_solute[i]))


        #Repeat the above tests for mixed dumbbells
        #First check if we have the correct bias vectors
        bias_calc_list_solute = []
        bias_calc_list_solvent = []
        for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
            bias_true_solute = self.onsagercalculator.NlsoluteBias2[i]
            bias_true_solvent = self.onsagercalculator.NlsolventBias2[i]
            bias_calc_solute = np.zeros(3)
            bias_calc_solvent = np.zeros(3)
            for jt,jlist,jindlist in zip(itertools.count(),self.onsagercalculator.jnet2,self.onsagercalculator.jnet2_indexed):
                for ((IS,FS),dx),jmp in zip(jindlist,jlist):
                    self.assertEqual(jmp.state1,self.onsagercalculator.vkinetic.starset.mixedstates[IS])
                    self.assertEqual(jmp.state2-jmp.state2.R_s,self.onsagercalculator.vkinetic.starset.mixedstates[FS],msg="\n{}\n{}".format(jmp.state2,self.onsagercalculator.vkinetic.starset.mixedstates[FS]))
                    if i==IS:
                        dx_solute = dx + jmp.state2.db.o/2. - jmp.state1.db.o/2.
                        dx_solvent = dx - jmp.state2.db.o/2. + jmp.state1.db.o/2.
                        bias_calc_solute += rate2list[jt][0]*dx_solute
                        bias_calc_solvent += rate2list[jt][0]*dx_solvent
            self.assertTrue(np.allclose(bias_calc_solute,bias_true_solute))
            self.assertTrue(np.allclose(bias_calc_solvent,bias_true_solvent))

        Nvstars_pure = self.onsagercalculator.vkinetic.Nvstars_pure
        for i in range(len(self.onsagercalculator.eta02_solvent)):
            #get the indices of the state
            st = self.onsagercalculator.vkinetic.starset.mixedstates[i]
            #next get the vectorstar,state indices
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[st]
            #The IndOfStar in indlist for the mixed case is already shifted by the number of pure vector stars.
            if len(indlist)!=0:
                vlist=[]
                for tup in indlist:
                    vlist.append(self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]])
                eta_test_solvent = sum([np.dot(self.onsagercalculator.eta02_solvent[i,:],v)*v for v in vlist])
                eta_test_solute = sum([np.dot(self.onsagercalculator.eta02_solute[i,:],v)*v for v in vlist])
                # print(vlist)
                self.assertTrue(np.allclose(eta_test_solvent*len(self.onsagercalculator.vkinetic.vecvec[indlist[0][0]]),self.onsagercalculator.eta02_solvent[i]),msg="{} {}".format(eta_test_solvent,self.onsagercalculator.eta02_solvent[i]))
                self.assertTrue(np.allclose(eta_test_solute*len(self.onsagercalculator.vkinetic.vecvec[indlist[0][0]]),self.onsagercalculator.eta02_solute[i]),msg="{} {}".format(eta_test_solute,self.onsagercalculator.eta02_solute[i]))

        # for i in range(len(self.onsagercalculator.eta02_solvent)):
        #     bias_test = np.zeros(3)
        #     db = self.onsagercalculator.vkinetic.starset.mixedStates[i]
        #     #First, let's check the indexing
        #     self.assertEqual(i,self.onsagercalculator.vkinetic.starset.bareindexdict[db][0])
        #     bias_true = self.onsagercalculator.NlsolventBias_bare[i,:]
        #     print(len(self.onsagercalculator.vkinetic.vecpos_bare[self.onsagercalculator.vkinetic.bareStTobareStar[db][0][0]]))
        #     # eta_true *= len(self.onsagercalculator.vkinetic.vecpos_bare[self.onsagercalculator.vkinetic.bareStTobareStar[db][0][0]])
        #     for jt,jindlist,jlist in zip(itertools.count(),self.onsagercalculator.jnet0_indexed,self.onsagercalculator.jnet0):
        #         for ((IS,FS),dx),jmp in zip(jindlist,jlist):
        #             if i==IS:
        #                 self.assertTrue(db==jmp.state1)
        #                 bias_test += rate0list[jt][0]*(self.onsagercalculator.eta00_solvent_bare[FS,:]-self.onsagercalculator.eta00_solvent_bare[IS,:])
        #     self.assertTrue(np.allclose(bias_test,bias_true),msg="{}{}".format(bias_test,bias_true))

    def test_bias_updates(self):
        """
        This is to check if the del_bias expansions are working fine, prod.
        """

        latt = np.array([[0.,0.1,0.5],[0.3,0.,0.5],[0.5,0.5,0.]])*0.55
        self.DC_Si = crystal.Crystal(latt,[[np.array([0.,0.,0.]),np.array([0.25,0.25,0.25])]],["Si"])
        self.cube = cube
        #keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = dbStates(self.DC_Si,0,family)
        self.mdbcontainer_si = mStates(self.DC_Si,0,family)

        self.onsagercalculator=dumbbellMediated(self.pdbcontainer_si,self.mdbcontainer_si,0.3,0.01,0.01,0.01,NGFmax=4,Nthermo=1)

        self.biases =\
        self.onsagercalculator.vkinetic.biasexpansion(self.onsagercalculator.jnet_1,self.onsagercalculator.jnet2,self.onsagercalculator.om1types,self.onsagercalculator.symjumplist_omega43_all)

        pre0 = np.random.rand(len(self.onsagercalculator.pdbcontainer.symorlist))
        betaene0 = np.random.rand(len(self.onsagercalculator.pdbcontainer.symorlist))
        pre0T = np.random.rand(len(self.onsagercalculator.jnet0))
        betaene0T = np.random.rand(len(self.onsagercalculator.jnet0))
        pre2 = np.random.rand(len(self.onsagercalculator.mdbcontainer.symorlist))
        betaene2 = np.random.rand(len(self.onsagercalculator.mdbcontainer.symorlist))
        pre2T = np.random.rand(len(self.onsagercalculator.jnet2))
        betaene2T = np.random.rand(len(self.onsagercalculator.jnet2))

        rate0list = ratelist(self.onsagercalculator.jnet0_indexed, pre0, betaene0, pre0T, betaene0T, self.onsagercalculator.vkinetic.starset.pdbcontainer.invmap)
        rate2list = ratelist(self.onsagercalculator.jnet2_indexed, pre2, betaene2, pre2T, betaene2T, self.onsagercalculator.vkinetic.starset.mdbcontainer.invmap)

        self.onsagercalculator.update_bias_expansions(pre0, betaene0, pre0T, betaene0T, pre2, betaene2, pre2T, betaene2T)

        #Next, we calculate the bias updates explicitly
        #First, we make random rate lists to test against
        W1list = np.random.rand(len(self.onsagercalculator.jnet_1))
        W2list = np.random.rand(len(self.onsagercalculator.jnet2))
        W3list = np.random.rand(len(self.onsagercalculator.symjumplist_omega3))
        W4list = np.random.rand(len(self.onsagercalculator.symjumplist_omega4))

        #Now, do it first for omega1
        bias1solute,bias1solvent = self.biases[1]
        bias1_solute_total = np.dot(bias1solute,W1list)
        bias1_solvent_total = np.dot(bias1solvent,W1list)
        #Now, convert this into the Nstates x 3 form
        solute_bias_1 = np.zeros((len(self.onsagercalculator.vkinetic.starset.purestates),3))
        solvent_bias_1 = np.zeros((len(self.onsagercalculator.vkinetic.starset.purestates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.purestates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[self.onsagercalculator.vkinetic.starset.purestates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            for tup in indlist:
                solute_bias_1[i,:] = sum([bias1_solute_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
                solvent_bias_1[i,:] = sum([bias1_solvent_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #Next, manually update with the eta0 vectors
        for i in range(len(self.onsagercalculator.vkinetic.starset.purestates)):
            for jt,jlist in enumerate(self.onsagercalculator.jnet1_indexed):
                for ((IS,FS),dx) in jlist:
                    if i==IS:
                        solute_bias_1[i,:] += W1list[jt]*(self.onsagercalculator.eta00_solute[IS] - self.onsagercalculator.eta00_solute[FS])
                        solvent_bias_1[i,:] += W1list[jt]*(self.onsagercalculator.eta00_solvent[IS] - self.onsagercalculator.eta00_solvent[FS])

        #Now, get the version from the updated expansion
        bias1_solute_new_total,bias1_solvent_new_total = np.dot(self.onsagercalculator.bias1_solute_new,W1list),np.dot(self.onsagercalculator.bias1_solvent_new,W1list)
        solute_bias_1_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.purestates),3))
        solvent_bias_1_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.purestates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.purestates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[self.onsagercalculator.vkinetic.starset.purestates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            for tup in indlist:
                solute_bias_1_new[i,:] = sum([bias1_solute_new_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
                solvent_bias_1_new[i,:] = sum([bias1_solvent_new_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        self.assertTrue(np.allclose(solute_bias_1,solute_bias_1_new))
        self.assertTrue(np.allclose(solute_bias_1,np.zeros_like(solute_bias_1)))
        self.assertTrue(np.allclose(solvent_bias_1,solvent_bias_1_new))

        # #Now, do it for omega2
        # bias2solute,bias2solvent = self.biases[2]
        # bias2_solute_total = np.dot(bias2solute,W2list)
        # bias2_solvent_total = np.dot(bias2solvent,W2list)
        # #Now, convert this into the Nstates x 3 form in the mixed state space - write a function to generalize this later on
        # solute_bias_2 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        # solvent_bias_2 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        # for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
        #     indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[self.onsagercalculator.vkinetic.starset.mixedstates[i]]
        #     #We have indlist as (IndOfStar, IndOfState)
        #     for tup in indlist:
        #         solute_bias_2[i,:] = sum([bias2_solute_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #         solvent_bias_2[i,:] = sum([bias2_solvent_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        # #Next, manually update with the eta0 vectors
        # for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
        #     for jt,jlist in enumerate(self.onsagercalculator.jnet2_indexed):
        #         for ((IS,FS),dx) in jlist:
        #             if i==IS:
        #                 solute_bias_1[i,:] += W2list[jt]*(self.onsagercalculator.eta02_solute[IS] - self.onsagercalculator.eta02_solute[FS])
        #                 solvent_bias_1[i,:] += W2list[jt]*(self.onsagercalculator.eta02_solvent[IS] - self.onsagercalculator.eta02_solvent[FS])
        #
        # #Now, get the version from the updated expansion
        # bias2_solute_new_total,bias2_solvent_new_total = np.dot(self.onsagercalculator.bias2_solute_new,W2list),np.dot(self.onsagercalculator.bias2_solvent_new,W2list)
        # solute_bias_2_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.purestates),3))
        # solvent_bias_2_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.purestates),3))
        # for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
        #     indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[self.onsagercalculator.vkinetic.starset.mixedstates[i]]
        #     #We have indlist as (IndOfStar, IndOfState)
        #     for tup in indlist:
        #         solute_bias_2_new[i,:] = sum([bias2_solute_new_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #         solvent_bias_2_new[i,:] = sum([bias2_solvent_new_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #
        # self.assertTrue(np.allclose(solute_bias_2,solute_bias_2_new))
        # self.assertTrue(np.allclose(solvent_bias_2,solvent_bias_2_new))
        # # self.assertTrue(np.allclose(solute_bias_2,np.zeros_like(solute_bias_1)))
