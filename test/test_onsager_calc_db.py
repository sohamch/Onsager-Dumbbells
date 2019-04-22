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
        # We test a new weird lattice because it is more interesting
        # latt = np.array([[0., 0.1, 0.5], [0.3, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        # self.DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # # keep it simple with [1.,0.,0.] type orientations for now
        # o = np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126
        # famp0 = [o.copy()]
        # family = [famp0]

        latt = np.array([[0., 0.1, 0.5], [0.3, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        self.DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = dbStates(self.DC_Si, 0, family)
        self.mdbcontainer_si = mStates(self.DC_Si, 0, family)

        self.pdbcontainer_si = dbStates(self.DC_Si, 0, family)
        self.mdbcontainer_si = mStates(self.DC_Si, 0, family)
        self.jset0, self.jset2 = \
            self.pdbcontainer_si.jumpnetwork(0.3, 0.01, 0.01), self.mdbcontainer_si.jumpnetwork(0.3, 0.01, 0.01)

        self.onsagercalculator = dumbbellMediated(self.pdbcontainer_si, self.mdbcontainer_si, self.jset0, self.jset2,
                                                  0.3, 0.01, 0.01, 0.01, NGFmax=4, Nthermo=1)
        # generate all the bias expansions - will separate out later
        self.biases = \
            self.onsagercalculator.vkinetic.biasexpansion(self.onsagercalculator.jnet_1, self.onsagercalculator.jnet2,
                                                          self.onsagercalculator.om1types,
                                                          self.onsagercalculator.symjumplist_omega43_all)

        self.W1list = np.random.rand(len(self.onsagercalculator.jnet_1))
        self.W2list = np.random.rand(len(self.onsagercalculator.jnet0))
        self.W3list = np.random.rand(len(self.onsagercalculator.symjumplist_omega3))
        self.W4list = np.random.rand(len(self.onsagercalculator.symjumplist_omega4))

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
        # set up random pre-factors and energies for rate calculations
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
        self.onsagercalculator.calc_eta(rate0list, rate2list)

        self.assertEqual(len(self.onsagercalculator.eta00_solvent),len(self.onsagercalculator.vkinetic.starset.complexStates))
        self.assertEqual(len(self.onsagercalculator.eta00_solute),len(self.onsagercalculator.vkinetic.starset.complexStates))

        self.assertEqual(len(self.onsagercalculator.eta02_solvent),len(self.onsagercalculator.vkinetic.starset.mixedstates))
        self.assertEqual(len(self.onsagercalculator.eta02_solute),len(self.onsagercalculator.vkinetic.starset.mixedstates))

        if len(self.onsagercalculator.vkinetic.vecpos_bare)==0:
            self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent,np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))))
            self.assertTrue(np.allclose(self.onsagercalculator.eta00_solute,np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))))

        else:
            # The non-local pure dumbbell biases have been tested in test_vecstars.
            # Here, we check if for periodic dumbbells, we have the same solvent bias.
            for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
                for j in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
                    if self.onsagercalculator.vkinetic.starset.complexStates[i].db.iorind == self.onsagercalculator.vkinetic.starset.complexStates[j].db.iorind:
                        # if np.allclose(self.onsagercalculator.vkinetic.starset.complexStates[i].db.o,self.onsagercalculator.vkinetic.starset.complexStates[j].db.o):
                        self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent[i,:],self.onsagercalculator.eta00_solvent[j,:]))
                        self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent[i,:],self.onsagercalculator.eta00_solvent_bare[self.onsagercalculator.vkinetic.starset.bareindexdict[self.onsagercalculator.vkinetic.starset.complexStates[i].db - self.onsagercalculator.vkinetic.starset.complexStates[i].db.R][0]]))

            # Check that we get the correct non-local bias vector
            for i, db in self.onsagercalculator.vkinetic.starset.bareStates:
                bias_true = self.onsagercalculator.NlsolventBias_bare[i, :]
                bias_calc = np.zeros(3)
                for jt,jlist in enumerate(self.onsagercalculator.jnet0_indexed):
                    for (IS,FS),dx in jlist:
                        if i==IS:
                            bias_calc += rate0list[jt][0]*dx
                self.assertTrue(np.allclose(bias_calc,bias_true), msg="\n{}\n{}".format(bias_calc, bias_true))

            # The above test confirms that our NlsolventBias_bare is correct
            # Now we check if the eta vectors are otrue
            for i in range(len(self.onsagercalculator.vkinetic.starset.bareStates)):
                bias_test = np.zeros(3)
                db = self.onsagercalculator.vkinetic.starset.bareStates[i]
                # First, let's check the indexing
                self.assertEqual(i,self.onsagercalculator.vkinetic.starset.bareindexdict[db][0])
                bias_true = self.onsagercalculator.NlsolventBias_bare[i,:]

                for jt,jindlist,jlist in zip(itertools.count(),self.onsagercalculator.jnet0_indexed,self.onsagercalculator.jnet0):
                    for ((IS,FS),dx),jmp in zip(jindlist,jlist):
                        if i==IS:
                            self.assertTrue(db==jmp.state1)
                            bias_test += rate0list[jt][0]*(self.onsagercalculator.eta00_solvent_bare[FS,:]-self.onsagercalculator.eta00_solvent_bare[IS,:])
                self.assertTrue(np.allclose(bias_test,bias_true),msg="{}{}".format(bias_test,bias_true))

            # A small test to reaffirm that vector bases are calculated properly for the bare states.
            for i in range(len(self.onsagercalculator.vkinetic.starset.bareStates)):
                # get the indices of the state
                st = self.onsagercalculator.vkinetic.starset.bareStates[i]
                # next get the vectorstar,state indices
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_bare[st]
                # The IndOfStar in indlist for the mixed case is already shifted by the number of pure vector stars.
                if len(indlist)!=0:
                    vlist=[]
                    for tup in indlist:
                        vlist.append(self.onsagercalculator.vkinetic.vecvec_bare[tup[0]][tup[1]])
                    eta_test_solvent = sum([np.dot(self.onsagercalculator.eta00_solvent_bare[i,:],v)*v for v in vlist])
                    self.assertTrue(np.allclose(eta_test_solvent*len(self.onsagercalculator.vkinetic.vecvec_bare[indlist[0][0]]),self.onsagercalculator.eta00_solvent_bare[i]),msg="{} {}".format(eta_test_solvent,self.onsagercalculator.eta00_solvent_bare[i]))


        # Repeat the above tests for mixed dumbbells
        # First check if we have the correct bias vectors
        for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
            bias_calc_solute = self.onsagercalculator.NlsoluteBias2[i]
            bias_calc_solvent = self.onsagercalculator.NlsolventBias2[i]
            bias_true_solute = np.zeros(3)
            bias_true_solvent = np.zeros(3)
            for jt,jlist,jindlist in zip(itertools.count(),self.onsagercalculator.jnet2,self.onsagercalculator.jnet2_indexed):
                for ((IS,FS),dx),jmp in zip(jindlist,jlist):
                    self.assertEqual(jmp.state1,self.onsagercalculator.vkinetic.starset.mixedstates[IS])
                    self.assertEqual(jmp.state2-jmp.state2.R_s,self.onsagercalculator.vkinetic.starset.mixedstates[FS],msg="\n{}\n{}".format(jmp.state2,self.onsagercalculator.vkinetic.starset.mixedstates[FS]))
                    if i==IS:
                        dx_solute =\
                            dx +\
                            self.onsagercalculator.vkinetic.starset.mdbcontainer.iorlist[jmp.state2.db.iorind][1]/2. -\
                            self.onsagercalculator.vkinetic.starset.mdbcontainer.iorlist[jmp.state1.db.iorind][1]/2.
                        dx_solvent = \
                            dx -\
                            self.onsagercalculator.vkinetic.starset.mdbcontainer.iorlist[jmp.state2.db.iorind][1]/2. +\
                            self.onsagercalculator.vkinetic.starset.mdbcontainer.iorlist[jmp.state1.db.iorind][1]/2.
                        bias_true_solute += rate2list[jt][0]*dx_solute
                        bias_true_solvent += rate2list[jt][0]*dx_solvent
            # print("testing lines 202,203")
            self.assertTrue(np.allclose(bias_calc_solute,bias_true_solute))
            self.assertTrue(np.allclose(bias_calc_solvent,bias_true_solvent))

        # Now that the bias vectors are correct, we should be able to recover them from the eta vectors
        for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
            bias_test_solute = np.zeros(3)
            bias_test_solvent = np.zeros(3)
            # db = self.onsagercalculator.vkinetic.starset.mixedStates[i]
            #First, let's check the indexing
            bias_true_solute = self.onsagercalculator.NlsoluteBias2[i]
            bias_true_solvent = self.onsagercalculator.NlsolventBias2[i]
            # print(len(self.onsagercalculator.vkinetic.vecpos_bare[self.onsagercalculator.vkinetic.bareStTobareStar[db][0][0]]))
            # eta_true *= len(self.onsagercalculator.vkinetic.vecpos_bare[self.onsagercalculator.vkinetic.bareStTobareStar[db][0][0]])
            for jt,jindlist,jlist in zip(itertools.count(),self.onsagercalculator.jnet2_indexed,self.onsagercalculator.jnet2):
                for ((IS,FS),dx),jmp in zip(jindlist,jlist):
                    if i==IS:
                        dx_solute =\
                            dx +\
                            self.onsagercalculator.vkinetic.starset.mdbcontainer.iorlist[jmp.state2.db.iorind][1]/2. -\
                            self.onsagercalculator.vkinetic.starset.mdbcontainer.iorlist[jmp.state1.db.iorind][1]/2.
                        dx_solvent =\
                            dx -\
                            self.onsagercalculator.vkinetic.starset.mdbcontainer.iorlist[jmp.state2.db.iorind][1]/2. +\
                            self.onsagercalculator.vkinetic.starset.mdbcontainer.iorlist[jmp.state1.db.iorind][1]/2.
                        self.assertTrue(self.onsagercalculator.vkinetic.starset.mixedstates[IS]==jmp.state1)
                        self.assertTrue(self.onsagercalculator.vkinetic.starset.mixedstates[FS]==jmp.state2-jmp.state2.R_s)
                        bias_test_solute += rate2list[jt][0]*(self.onsagercalculator.eta02_solute[FS,:]-self.onsagercalculator.eta02_solute[IS,:])
                        bias_test_solvent += rate2list[jt][0]*(self.onsagercalculator.eta02_solvent[FS,:]-self.onsagercalculator.eta02_solvent[IS,:])
            self.assertTrue(np.allclose(bias_test_solute,bias_true_solute))
            self.assertTrue(np.allclose(bias_test_solvent,bias_true_solvent))
        
    def test_bias_updates(self):
        """
        This is to check if the del_bias expansions are working fine, prod.
        """
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

        self.onsagercalculator.update_bias_expansions(rate0list,rate2list)

        # Next, we calculate the bias updates explicitly First, we make lists to test against While the non-local
        # rates are determined by rate0list and rate2list, we are free to use random local corrections
        W0list = np.array([rate0list[i][0] for i in range(len(rate0list))])
        W2list = np.array([rate2list[i][0] for i in range(len(rate2list))])
        # Now, local corrections (randomized)
        W1list = np.random.rand(len(self.onsagercalculator.jnet_1))
        W3list = np.random.rand(len(self.onsagercalculator.symjumplist_omega3))
        W4list = np.random.rand(len(self.onsagercalculator.symjumplist_omega4))

        # First, we verify that the non-local bias out of all the bare dumbbell states dissappear
        bias0_solvent_nonloc = np.dot(self.onsagercalculator.biasBareExpansion,W0list)
        # solute_bias_Nl = np.zeros((len(self.onsagercalculator.vkinetic.starset.bareStates),3))
        solvent_bias_Nl = np.zeros((len(self.onsagercalculator.vkinetic.starset.bareStates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.bareStates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_bare[self.onsagercalculator.vkinetic.starset.bareStates[i]]
            # We have indlist as (IndOfStar, IndOfState)
            solvent_bias_Nl[i,:] = sum([bias0_solvent_nonloc[tup[0]]*self.onsagercalculator.vkinetic.vecvec_bare[tup[0]][tup[1]] for tup in indlist])

        #Next, update with eta vectors manually
        for jt,jindlist in enumerate(self.onsagercalculator.jnet0_indexed):
            for (i,j),dx in jindlist:
                solvent_bias_Nl[i,:] += rate0list[jt][0]*(self.onsagercalculator.eta00_solvent_bare[i] - self.onsagercalculator.eta00_solvent_bare[j])

        self.assertTrue(np.allclose(solvent_bias_Nl,np.zeros_like(solvent_bias_Nl)))
        # self.assertTrue(np.allclose(solute_bias_Nl,np.zeros_like(solute_bias_Nl)))

        #Now, do check eta vectors for omega1
        bias1solute,bias1solvent = self.biases[1]
        bias1_solute_total = np.dot(bias1solute,W1list)
        bias1_solvent_total = np.dot(bias1solvent,W1list)
        #Now, convert this into the Nstates x 3 form
        solute_bias_1 = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        solvent_bias_1 = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[self.onsagercalculator.vkinetic.starset.complexStates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            for tup in indlist:
                solute_bias_1[i,:] = sum([bias1_solute_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
                solvent_bias_1[i,:] = sum([bias1_solvent_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #Next, manually update with the eta0 vectors
        # for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
        for jt,jlist in enumerate(self.onsagercalculator.jnet1_indexed):
            for ((IS,FS),dx) in jlist:
                # if i==IS:
                solute_bias_1[IS,:] += W1list[jt]*(self.onsagercalculator.eta00_solute[IS] - self.onsagercalculator.eta00_solute[FS])
                solvent_bias_1[IS,:] += W1list[jt]*(self.onsagercalculator.eta00_solvent[IS] - self.onsagercalculator.eta00_solvent[FS])

        #Now, get the version from the updated expansion
        bias1_solute_new_total,bias1_solvent_new_total = np.dot(self.onsagercalculator.bias1_solute_new,W1list),np.dot(self.onsagercalculator.bias1_solvent_new,W1list)
        solute_bias_1_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        solvent_bias_1_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[self.onsagercalculator.vkinetic.starset.complexStates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            solute_bias_1_new[i,:] = sum([bias1_solute_new_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            solvent_bias_1_new[i,:] = sum([bias1_solvent_new_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        #Check that they are the same
        self.assertTrue(np.allclose(solute_bias_1,solute_bias_1_new))
        self.assertTrue(np.allclose(solute_bias_1,np.zeros_like(solute_bias_1)))
        self.assertTrue(np.allclose(solvent_bias_1,solvent_bias_1_new))

        #For the kinetic shell, we check that for those states, out of which every (omega0-allowed) jump leads
        #to another state in the kinetic shell, the non-local bias becomes zero, using omega1 jumps.
        #For those states, out of which not every omega0-allowed jump is also there in omega1, since it leads to
        #a state outside the shell, the corresponding non-local bias using just the jumps in omega1 should be non-zero.
        elim_list=np.zeros(len(self.onsagercalculator.vkinetic.starset.complexStates))
        #This array stores how many omega0 jumps are not considered in omega1, for each state
        for stindex,st in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            for jlist in self.onsagercalculator.jnet0:
                for jmp in jlist:
                    try:
                        stnew = st.addjump(jmp)
                    except:
                        continue
                    if not stnew in self.onsagercalculator.vkinetic.starset.stateset:
                        elim_list[stindex] += 1
        #Get the non-local rates corresponding to a omega1 jump
        W01list = [rate0list[i][0] for i in self.onsagercalculator.om1types]
        #Get the omega1 contribution to the non-local bias vectors
        #First check that there is no movement in the solutes
        self.assertTrue(np.allclose(self.onsagercalculator.bias1_solute_new,np.zeros((self.onsagercalculator.vkinetic.Nvstars_pure,len(self.onsagercalculator.jnet_1)))))
        bias1_solvent_new_total = np.dot(self.onsagercalculator.bias1_solvent_new,W01list)
        solvent_bias_1_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[self.onsagercalculator.vkinetic.starset.complexStates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            solvent_bias_1_new[i,:] = sum([bias1_solvent_new_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        #Calculate the updated bias explicitly
        bias10solvent = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        for jt,jlist,jindlist in zip(itertools.count(),self.onsagercalculator.jnet_1,self.onsagercalculator.jnet1_indexed):
            for ((IS,FS),dx), jmp in zip(jindlist,jlist):
                bias10solvent[IS,:] += W01list[jt]*dx

        #Now, update with eta vectors
        for jt,jlist,jindlist in zip(itertools.count(),self.onsagercalculator.jnet_1,self.onsagercalculator.jnet1_indexed):
            for ((IS,FS),dx), jmp in zip(jindlist,jlist):
                bias10solvent[IS,:] += W01list[jt]*(self.onsagercalculator.eta00_solvent[IS]-self.onsagercalculator.eta00_solvent[FS])
        #Check that we have the same vectors using the W01list as well
        self.assertTrue(np.allclose(bias10solvent,solvent_bias_1_new))

        #Now, check that the proper states leave zero non-local bias vectors.
        for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
            if elim_list[i]==0: #if no omega0 jumps have been eliminated
                self.assertTrue(np.allclose(bias10solvent[i],0))
            if elim_list[i]!=0:
                self.assertFalse(np.allclose(bias10solvent[i],0))

        #Now, do it for omega2
        bias2solute,bias2solvent = self.biases[2]
        bias2_solute_total = np.dot(bias2solute,W2list)
        bias2_solvent_total = np.dot(bias2solvent,W2list)
        #Now, convert this into the Nstates x 3 form in the mixed state space - write a function to generalize this later on
        solute_bias_2 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        solvent_bias_2 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[self.onsagercalculator.vkinetic.starset.mixedstates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            for tup in indlist:
                solute_bias_2[i,:] = sum([bias2_solute_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
                solvent_bias_2[i,:] = sum([bias2_solvent_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #Next, manually update with the eta0 vectors
        # for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
        for jt,jlist in enumerate(self.onsagercalculator.jnet2_indexed):
            for ((IS,FS),dx) in jlist:
                # if i==IS:
                solute_bias_2[IS,:] += W2list[jt]*(self.onsagercalculator.eta02_solute[IS] - self.onsagercalculator.eta02_solute[FS])
                solvent_bias_2[IS,:] += W2list[jt]*(self.onsagercalculator.eta02_solvent[IS] - self.onsagercalculator.eta02_solvent[FS])

        #Now, get the version from the updated expansion
        bias2_solute_new_total,bias2_solvent_new_total = np.dot(self.onsagercalculator.bias2_solute_new,W2list),np.dot(self.onsagercalculator.bias2_solvent_new,W2list)
        solute_bias_2_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        solvent_bias_2_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[self.onsagercalculator.vkinetic.starset.mixedstates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            solute_bias_2_new[i,:] = sum([bias2_solute_new_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            solvent_bias_2_new[i,:] = sum([bias2_solvent_new_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        self.assertTrue(np.allclose(solute_bias_2,solute_bias_2_new))
        self.assertTrue(np.allclose(solvent_bias_2,solvent_bias_2_new))
        #The following tests must hold - the non-local biases in omega2_space must become zero after eta updates
        self.assertTrue(np.allclose(solute_bias_2_new,np.zeros_like(solute_bias_2)),msg="\n{}\n".format(solute_bias_2))
        self.assertTrue(np.allclose(solvent_bias_2_new,np.zeros_like(solvent_bias_2)))

        #Now, do it for omega3
        bias3solute,bias3solvent = self.biases[3]
        bias3_solute_total = np.dot(bias3solute,W3list)
        bias3_solvent_total = np.dot(bias3solvent,W3list)
        #Now, convert this into the Nstates x 3 form in the mixed state space - write a function to generalize this later on
        solute_bias_3 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        solvent_bias_3 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[self.onsagercalculator.vkinetic.starset.mixedstates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            for tup in indlist:
                solute_bias_3[i,:] = sum([bias3_solute_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
                solvent_bias_3[i,:] = sum([bias3_solvent_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #Next, manually update with the eta0 vectors
        for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
            for jt,jlist in enumerate(self.onsagercalculator.symjumplist_omega3_indexed):
                for ((IS,FS),dx) in jlist:
                    if i==IS:
                        solute_bias_3[i,:] += W3list[jt]*(self.onsagercalculator.eta02_solute[IS] - self.onsagercalculator.eta00_solute[FS])
                        solvent_bias_3[i,:] += W3list[jt]*(self.onsagercalculator.eta02_solvent[IS] - self.onsagercalculator.eta00_solvent[FS])

        #Now, get the version from the updated expansion
        bias3_solute_new_total,bias3_solvent_new_total = np.dot(self.onsagercalculator.bias3_solute_new,W3list),np.dot(self.onsagercalculator.bias3_solvent_new,W3list)
        solute_bias_3_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        solvent_bias_3_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[self.onsagercalculator.vkinetic.starset.mixedstates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            solute_bias_3_new[i,:] = sum([bias3_solute_new_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            solvent_bias_3_new[i,:] = sum([bias3_solvent_new_total[tup[0]-self.onsagercalculator.vkinetic.Nvstars_pure]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        self.assertTrue(np.allclose(solute_bias_3,solute_bias_3_new))
        self.assertTrue(np.allclose(solvent_bias_3,solvent_bias_3_new))

        #Now, do it for omega4
        bias4solute,bias4solvent = self.biases[4]
        bias4_solute_total = np.dot(bias4solute,W4list)
        bias4_solvent_total = np.dot(bias4solvent,W4list)
        #Now, convert this into the Nstates x 3 form in the mixed state space - write a function to generalize this later on
        solute_bias_4 = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        solvent_bias_4 = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[self.onsagercalculator.vkinetic.starset.complexStates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            for tup in indlist:
                solute_bias_4[i,:] = sum([bias4_solute_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
                solvent_bias_4[i,:] = sum([bias4_solvent_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #Next, manually update with the eta0 vectors
        for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
            for jt,jlist in enumerate(self.onsagercalculator.symjumplist_omega4_indexed):
                for ((IS,FS),dx) in jlist:
                    if i==IS:
                        solute_bias_4[i,:] += W4list[jt]*(self.onsagercalculator.eta00_solute[IS] - self.onsagercalculator.eta02_solute[FS])
                        solvent_bias_4[i,:] += W4list[jt]*(self.onsagercalculator.eta00_solvent[IS] - self.onsagercalculator.eta02_solvent[FS])

        #Now, get the version from the updated expansion
        bias4_solute_new_total,bias4_solvent_new_total = np.dot(self.onsagercalculator.bias4_solute_new,W4list),np.dot(self.onsagercalculator.bias4_solvent_new,W4list)
        solute_bias_4_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        solvent_bias_4_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),3))
        for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[self.onsagercalculator.vkinetic.starset.complexStates[i]]
            #We have indlist as (IndOfStar, IndOfState)
            solute_bias_4_new[i,:] = sum([bias4_solute_new_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            solvent_bias_4_new[i,:] = sum([bias4_solvent_new_total[tup[0]]*self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        self.assertTrue(np.allclose(solute_bias_4,solute_bias_4_new))
        self.assertTrue(np.allclose(solvent_bias_4,solvent_bias_4_new))
