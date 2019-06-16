from onsager.crystal import Crystal
from onsager.crystalStars import zeroclean
from Onsager_calc_db import *
from test_structs import *
from representations import *
from states import *
from stars import *
from vector_stars import *
import unittest

class test_dumbbell_mediated(unittest.TestCase):
    def setUp(self):
        # We test a new weird lattice because it is more interesting
        # latt = np.array([[0., 0.1, 0.5], [0.3, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        # self.DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # # keep it simple with [1.,0.,0.] type orientations for now
        # o = np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126
        # famp0 = [o.copy()]
        # family = [famp0]

        latt = np.array([[0.5, 0.5, 0.], [0., 0.5, 0.5], [0.5, 0., 0.5]]) * 0.55
        self.DC_Si = Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1., 0., 0.]) * 0.126
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
        print("Initiated")

    def test_thermo2kin(self):
        for th_ind, k_ind in enumerate(self.onsagercalculator.thermo2kin):
            count = 0
            for state1 in self.onsagercalculator.thermo.stars[th_ind]:
                for state2 in self.onsagercalculator.vkinetic.starset.stars[k_ind]:
                    if state1 == state2:
                        count += 1
            self.assertEqual(count, len(self.onsagercalculator.thermo.stars[th_ind]))

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

        # First, we verify that the non-local bias out of all the bare dumbbell states disappear
        if not len(self.onsagercalculator.vkinetic.vecpos_bare) == 0:
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

        self.assertTrue(np.allclose(solute_bias_2, solute_bias_2_new))
        self.assertTrue(np.allclose(solvent_bias_2, solvent_bias_2_new))
        # The following tests must hold - the non-local biases in omega2_space must become zero after eta updates
        self.assertTrue(np.allclose(solute_bias_2_new, np.zeros_like(solute_bias_2)),msg="\n{}\n".format(solute_bias_2))
        self.assertTrue(np.allclose(solvent_bias_2_new, np.zeros_like(solvent_bias_2)))

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

    def test_uncorrelated_del_om(self):
        """
        Test the uncorrelated contribution to diffusivity part by part.
        Also in the process check the omega rate list creation and everything.
        """
        # First, we need some thermodynamic data
        # We randomize site and transition energies for now.

        # Set up energies and pre-factors
        kT = 1.

        predb0, enedb0 = np.ones(len(self.onsagercalculator.vkinetic.starset.pdbcontainer.symorlist)), \
                         np.random.rand(len(self.onsagercalculator.vkinetic.starset.pdbcontainer.symorlist))

        preS, eneS = np.ones(
            len(self.onsagercalculator.vkinetic.starset.crys.sitelist(self.onsagercalculator.vkinetic.starset.chem))), \
                     np.random.rand(len(self.onsagercalculator.vkinetic.starset.crys.sitelist(
                         self.onsagercalculator.vkinetic.starset.chem)))

        # These are the interaction or the excess energies and pre-factors for solutes and dumbbells.
        preSdb, eneSdb = np.ones(self.onsagercalculator.thermo.mixedstartindex), \
                         np.random.rand(self.onsagercalculator.thermo.mixedstartindex)

        predb2, enedb2 = np.ones(len(self.onsagercalculator.vkinetic.starset.mdbcontainer.symorlist)), \
                         np.random.rand(len(self.onsagercalculator.vkinetic.starset.mdbcontainer.symorlist))

        preT0, eneT0 = np.ones(len(self.onsagercalculator.vkinetic.starset.jnet0)), np.random.rand(
            len(self.onsagercalculator.vkinetic.starset.jnet0))
        preT2, eneT2 = np.ones(len(self.onsagercalculator.vkinetic.starset.jnet2)), np.random.rand(
            len(self.onsagercalculator.vkinetic.starset.jnet2))
        preT1, eneT1 = np.ones(len(self.onsagercalculator.jnet_1)), np.random.rand(len(self.onsagercalculator.jnet_1))

        preT43, eneT43 = np.ones(len(self.onsagercalculator.symjumplist_omega43_all)), \
                         np.random.rand(len(self.onsagercalculator.symjumplist_omega43_all))

        # Now get the beta*free energy values.
        bFdb0, bFdb2, bFS, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4 =\
            self.onsagercalculator.preene2betafree(kT, predb0, enedb0, preS, eneS, preSdb, eneSdb, predb2, enedb2,
                                                   preT0, eneT0, preT2, eneT2, preT1, eneT1, preT43, eneT43)

        (L_uc_aa, L_c_aa), (L_uc_bb, L_c_bb), (L_uc_ab, L_c_ab), GF_total, GF20, del_om, part_func, probs, omegas,\
        stateprobs = self.onsagercalculator.L_ij(bFdb0, bFT0, bFdb2, bFT2, bFS, bFSdb, bFT1, bFT3, bFT4)

        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),\
        (omega4, omega4escape) = omegas

        # First, check the omega1 rates coming out of origin states are zero
        for jt, rate in enumerate(omega1):
            if self.onsagercalculator.jnet_1[jt][0].state1.is_zero(self.onsagercalculator.pdbcontainer) \
                    or self.onsagercalculator.jnet_1[jt][0].state2.is_zero(self.onsagercalculator.pdbcontainer):
                self.assertEqual(rate, 0.)

        eta0total_solute = self.onsagercalculator.eta0total_solute
        eta0total_solvent = self.onsagercalculator.eta0total_solvent
        # Now, let's get the bias expansions
        (D1expansion_aa, D1expansion_bb, D1expansion_ab), \
        (D2expansion_aa, D2expansion_bb, D2expansion_ab), \
        (D3expansion_aa, D3expansion_bb, D3expansion_ab), \
        (D4expansion_aa, D4expansion_bb, D4expansion_ab) =\
            self.onsagercalculator.bareExpansion(eta0total_solute,
                                                 eta0total_solvent)

        complex_prob, mixed_prob = stateprobs

        # Now set up the multiplicative quantity for each jump type.
        prob_om1 = np.zeros(len(omega1))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.onsagercalculator.jnet1_indexed]):
            prob_om1[jt] = np.sqrt(complex_prob[IS] * complex_prob[FS]) * omega1[jt]

        prob_om2 = np.zeros(len(self.onsagercalculator.jnet2))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.onsagercalculator.jnet2_indexed]):
            prob_om2[jt] = np.sqrt(mixed_prob[IS] * mixed_prob[FS]) * omega2[jt]

        prob_om4 = np.zeros(len(self.onsagercalculator.symjumplist_omega4))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.onsagercalculator.symjumplist_omega4_indexed]):
            prob_om4[jt] = np.sqrt(complex_prob[IS] * mixed_prob[FS]) * omega4[jt]

        prob_om3 = np.zeros(len(self.onsagercalculator.symjumplist_omega3))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.onsagercalculator.symjumplist_omega3_indexed]):
            prob_om3[jt] = np.sqrt(mixed_prob[IS] * complex_prob[FS]) * omega3[jt]

        # Now, let's compute the contribution by omega1 jumps
        # For solutes, it's zero anyway - let's check for solvents
        L_uc_om1_test_solvent = np.zeros((3,3))
        for jt, jlist in enumerate(self.onsagercalculator.jnet1_indexed):
            for (IS, FS), dx in jlist:
                L_uc_om1_test_solvent += np.outer(dx + eta0total_solvent[IS] - eta0total_solvent[FS],
                                                  dx + eta0total_solvent[IS] - eta0total_solvent[FS])*prob_om1[jt]*0.5
        L_uc_om1 = np.dot(D1expansion_bb, prob_om1)
        self.assertTrue(np.allclose(L_uc_om1_test_solvent, L_uc_om1))

        # Now, let's check the omega2 contributions
        L_uc_om2_bb = np.dot(D2expansion_bb, prob_om2)
        L_uc_om2_aa = np.dot(D2expansion_aa, prob_om2)
        L_uc_om2_ab = np.dot(D2expansion_ab, prob_om2)
        L_uc_om2_test_aa = np.zeros((3, 3))
        L_uc_om2_test_bb = np.zeros((3, 3))
        L_uc_om2_test_ab = np.zeros((3, 3))
        Ncomp = len(self.onsagercalculator.vkinetic.starset.complexStates)
        for jt, jlist in enumerate(self.onsagercalculator.jnet2_indexed):
            for (IS, FS), dx in jlist:
                o1 = self.onsagercalculator.mdbcontainer.iorlist[
                    self.onsagercalculator.vkinetic.starset.mixedstates[IS].db.iorind][1]
                o2 = self.onsagercalculator.mdbcontainer.iorlist[
                    self.onsagercalculator.vkinetic.starset.mixedstates[FS].db.iorind][1]

                dx_solute = dx - o1/2. + o2/2. + eta0total_solute[IS + Ncomp] - eta0total_solute[FS + Ncomp]
                dx_solvent = dx + o1/2. - o2/2. + eta0total_solvent[IS + Ncomp] - eta0total_solvent[FS + Ncomp]

                L_uc_om2_test_aa += np.outer(dx_solute, dx_solute)* prob_om2[jt] * 0.5
                L_uc_om2_test_bb += np.outer(dx_solvent, dx_solvent) * prob_om2[jt] * 0.5
                L_uc_om2_test_ab += np.outer(dx_solute, dx_solvent) * prob_om2[jt] * 0.5

        self.assertTrue(np.allclose(L_uc_om2_test_aa, L_uc_om2_aa))
        self.assertTrue(np.allclose(L_uc_om2_test_bb, L_uc_om2_bb))
        self.assertTrue(np.allclose(L_uc_om2_test_ab, L_uc_om2_ab))

        # Now, let's check the omega3 contributions
        L_uc_om3_bb = np.dot(D3expansion_bb, prob_om3)
        L_uc_om3_aa = np.dot(D3expansion_aa, prob_om3)
        L_uc_om3_ab = np.dot(D3expansion_ab, prob_om3)
        L_uc_om3_test_aa = np.zeros((3, 3))
        L_uc_om3_test_bb = np.zeros((3, 3))
        L_uc_om3_test_ab = np.zeros((3, 3))

        for jt, jlist in enumerate(self.onsagercalculator.symjumplist_omega3_indexed):
            # The initial state is a  mixed dumbbell and the final is a pure dumbbell
            sm_aa = np.zeros((3, 3))
            sm_bb = np.zeros((3, 3))
            sm_ab = np.zeros((3, 3))
            for (IS, FS), dx in jlist:
                o1 = self.onsagercalculator.mdbcontainer.iorlist[
                    self.onsagercalculator.vkinetic.starset.mixedstates[IS].db.iorind][1]
                o2 = self.onsagercalculator.pdbcontainer.iorlist[
                    self.onsagercalculator.vkinetic.starset.complexStates[IS].db.iorind][1]

                dx_solute = -o1/2. + eta0total_solute[IS + Ncomp] - eta0total_solute[FS]
                dx_solvent = o1/2. + dx + eta0total_solvent[IS + Ncomp] - eta0total_solvent[FS]
                sm_aa += zeroclean(np.outer(dx_solute, dx_solute)) * 0.5
                sm_bb += zeroclean(np.outer(dx_solvent, dx_solvent)) * 0.5
                sm_ab += zeroclean(np.outer(dx_solute, dx_solvent)) * 0.5
                L_uc_om3_test_aa += zeroclean(np.outer(dx_solute, dx_solute)) * prob_om3[jt] * 0.5
                L_uc_om3_test_bb += zeroclean(np.outer(dx_solvent, dx_solvent)) * prob_om3[jt] * 0.5
                L_uc_om3_test_ab += zeroclean(np.outer(dx_solute, dx_solvent)) * prob_om3[jt] * 0.5
            self.assertTrue(np.allclose(D3expansion_aa[:, :, jt], sm_aa), msg="{}".format(jt))

        self.assertTrue(np.allclose(L_uc_om3_test_aa, L_uc_om3_aa), msg="\n {} \n {}".format(L_uc_om3_test_aa, L_uc_om3_aa))
        self.assertTrue(np.allclose(L_uc_om3_test_bb, L_uc_om3_bb))
        self.assertTrue(np.allclose(L_uc_om3_test_ab, L_uc_om3_ab))

        # Now, let's check the omega4 contributions
        L_uc_om4_bb = np.dot(D4expansion_bb, prob_om4)
        L_uc_om4_aa = np.dot(D4expansion_aa, prob_om4)
        L_uc_om4_ab = np.dot(D4expansion_ab, prob_om4)
        L_uc_om4_test_aa = np.zeros((3, 3))
        L_uc_om4_test_bb = np.zeros((3, 3))
        L_uc_om4_test_ab = np.zeros((3, 3))

        for jt, jlist in enumerate(self.onsagercalculator.symjumplist_omega4_indexed):
            # The initial state is a  pure dumbbell and the final is a mixed dumbbell
            for (IS, FS), dx in jlist:
                o1 = self.onsagercalculator.pdbcontainer.iorlist[
                    self.onsagercalculator.vkinetic.starset.complexStates[IS].db.iorind][1]
                o2 = self.onsagercalculator.mdbcontainer.iorlist[
                    self.onsagercalculator.vkinetic.starset.mixedstates[FS].db.iorind][1]

                dx_solute = o2 / 2. + eta0total_solute[IS] - eta0total_solute[FS + Ncomp]
                dx_solvent = -o2 / 2. + dx + eta0total_solvent[IS] - eta0total_solvent[FS + Ncomp]
                L_uc_om4_test_aa += np.outer(dx_solute, dx_solute) * prob_om4[jt] * 0.5
                L_uc_om4_test_bb += np.outer(dx_solvent, dx_solvent) * prob_om4[jt] * 0.5
                L_uc_om4_test_ab += np.outer(dx_solute, dx_solvent) * prob_om4[jt] * 0.5

        self.assertTrue(np.allclose(L_uc_om4_test_aa, L_uc_om4_aa))
        self.assertTrue(np.allclose(L_uc_om4_test_bb, L_uc_om4_bb))
        self.assertTrue(np.allclose(L_uc_om4_test_ab, L_uc_om4_ab))

    # construct a test for the way the rates are constructed.

    def test_delom_G0_probs(self):
        """
        This is to check the delta omega matrix
        """
        # 1.  First get the rates and thermodynamic data.

        # 1a. Energies and pre-factors
        kT = 1.

        predb0, enedb0 = np.ones(len(self.onsagercalculator.vkinetic.starset.pdbcontainer.symorlist)), \
                         np.random.rand(len(self.onsagercalculator.vkinetic.starset.pdbcontainer.symorlist))

        preS, eneS = np.ones(
            len(self.onsagercalculator.vkinetic.starset.crys.sitelist(self.onsagercalculator.vkinetic.starset.chem))), \
                     np.random.rand(len(self.onsagercalculator.vkinetic.starset.crys.sitelist(
                         self.onsagercalculator.vkinetic.starset.chem)))

        # These are the interaction or the excess energies and pre-factors for solutes and dumbbells.
        preSdb, eneSdb = np.ones(self.onsagercalculator.thermo.mixedstartindex), \
                         np.random.rand(self.onsagercalculator.thermo.mixedstartindex)

        predb2, enedb2 = np.ones(len(self.onsagercalculator.vkinetic.starset.mdbcontainer.symorlist)), \
                         np.random.rand(len(self.onsagercalculator.vkinetic.starset.mdbcontainer.symorlist))

        preT0, eneT0 = np.ones(len(self.onsagercalculator.vkinetic.starset.jnet0)), np.random.rand(
            len(self.onsagercalculator.vkinetic.starset.jnet0))
        preT2, eneT2 = np.ones(len(self.onsagercalculator.vkinetic.starset.jnet2)), np.random.rand(
            len(self.onsagercalculator.vkinetic.starset.jnet2))
        preT1, eneT1 = np.ones(len(self.onsagercalculator.jnet_1)), np.random.rand(len(self.onsagercalculator.jnet_1))

        preT43, eneT43 = np.ones(len(self.onsagercalculator.symjumplist_omega43_all)), \
                         np.random.rand(len(self.onsagercalculator.symjumplist_omega43_all))

        # 1c. Now get the beta*free energy values.
        bFdb0, bFdb2, bFS, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4 = \
            self.onsagercalculator.preene2betafree(kT, predb0, enedb0, preS, eneS, preSdb, eneSdb, predb2, enedb2,
                                                   preT0, eneT0, preT2, eneT2, preT1, eneT1, preT43, eneT43)

        bFdb0_min = np.min(bFdb0)
        bFdb2_min = np.min(bFdb2)
        bFS_min = np.min(bFS)

        # 1d. Modify the solute-dumbell interaction energies
        bFSdb_total = np.zeros(self.onsagercalculator.vkinetic.starset.mixedstartindex)
        bFSdb_total_shift = np.zeros(self.onsagercalculator.vkinetic.starset.mixedstartindex)

        # first, just add up the solute and dumbbell energies. We will add in the corrections to the thermo shell states
        # later.
        # Also, we need to keep an unshifted version to be able to normalize probabilities
        for starind, star in enumerate(self.onsagercalculator.vkinetic.starset.stars[
                                       :self.onsagercalculator.vkinetic.starset.mixedstartindex]):
            # For origin complex states, do nothing - leave them as zero.
            if star[0].is_zero(self.onsagercalculator.vkinetic.starset.pdbcontainer):
                continue
            symindex = self.onsagercalculator.vkinetic.starset.star2symlist[starind]
            # First, get the unshifted value
            # check that invmaps are okay
            solwyck = self.onsagercalculator.invmap_solute[star[0].i_s]
            for state in star:
                self.assertEqual(self.onsagercalculator.invmap_solute[state.i_s], solwyck)

            bFSdb_total[starind] = bFdb0[symindex] + bFS[solwyck]
            bFSdb_total_shift[starind] = bFSdb_total[starind] - (bFdb0_min + bFS_min)

        # Now add in the changes for the complexes inside the thermodynamic shell.
        for starind, star in enumerate(self.onsagercalculator.thermo.stars[
                                       :self.onsagercalculator.thermo.mixedstartindex]):
            # Get the symorlist index for the representative state of the star
            if star[0].is_zero(self.onsagercalculator.thermo.pdbcontainer):
                continue
            # keep the total energies zero for origin states.

            # check thermo2kin
            kinind = self.onsagercalculator.thermo2kin[starind]
            self.assertEqual(len(star), len(self.onsagercalculator.vkinetic.starset.stars[kinind]))
            count = 0
            for state in star:
                for statekin in self.onsagercalculator.vkinetic.starset.stars[kinind]:
                    if state == statekin:
                        count += 1
            self.assertEqual(count, len(star))

            bFSdb_total[kinind] += bFSdb[starind]
            bFSdb_total_shift[kinind] += bFSdb[starind]

        # 1e. Now get the symmetrized rates
        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),\
        (omega4, omega4escape) = self.onsagercalculator.getsymmrates(bFdb0 - bFdb0_min, bFdb2 - bFdb2_min,
                                                                     bFSdb_total_shift, bFT0, bFT1, bFT2, bFT3, bFT4)

        for jt in range(len(self.onsagercalculator.symjumplist_omega43_all)):
            self.assertEqual(omega4[jt], omega3[jt])

        omegas = (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),\
                 (omega4, omega4escape)

        # 2. we get the delta omega matrix, as it is calculated in the code in the makeGF function
        # Need to run calc_eta first to get g2 - this should be made universal later on
        pre0, pre0T = np.ones_like(bFdb0), np.ones_like(bFT0)
        pre2, pre2T = np.ones_like(bFdb2), np.ones_like(bFT2)
        symrate0list = symmratelist(self.onsagercalculator.jnet0_indexed, pre0, bFdb0 - bFdb0_min, pre0T, bFT0,
                                    self.onsagercalculator.vkinetic.starset.pdbcontainer.invmap)

        symrate2list = symmratelist(self.onsagercalculator.jnet2_indexed, pre2, bFdb2 - bFdb2_min, pre2T, bFT2,
                                    self.onsagercalculator.vkinetic.starset.mdbcontainer.invmap)
        self.onsagercalculator.calc_eta(symrate0list, symrate2list)
        # Check consistency between the two rate
        for jt in range(len(self.onsagercalculator.jnet0)):
            self.assertEqual(symrate0list[jt][0], omega0[jt])

        for jt in range(len(self.onsagercalculator.jnet2)):
            self.assertEqual(symrate2list[jt][0], omega2[jt])

        GF_total, GF20, del_om = self.onsagercalculator.makeGF(bFdb0 - bFdb0_min, bFdb2 - bFdb2_min, bFT0, bFT2, omegas)
        print("GF made")
        (L_uc_aa, L_c_aa), (L_uc_bb, L_c_bb), (L_uc_ab, L_c_ab), GF_total, GF20, del_om, part_func, probs, omegas, stateprobs = \
            self.onsagercalculator.L_ij(bFdb0, bFT0, bFdb2, bFT2, bFS, bFSdb, bFT1, bFT3, bFT4)

        # 3. Now, that we have the symmetrized rates, we need to construct the delta_om matrix using it's mathematical
        # form

        Nvstars = self.onsagercalculator.vkinetic.Nvstars
        # Nvstars_pure = self.onsagercalculator.vkinetic.Nvstars_pure
        # Nvstars_mixed = Nvstars - Nvstars_pure

        delta_om_test = np.zeros((Nvstars, Nvstars))
        # 3a. First, we do the non-diagonal parts
        delta_om_test = np.zeros((Nvstars, Nvstars))
        # 3a.1 - First, we concentrate on the complex-complex block and the omega1 jumps
        for jt, jlist in enumerate(self.onsagercalculator.jnet_1):
            delom1 = omega1[jt] - omega0[self.onsagercalculator.om1types[jt]]
            for jmp in jlist:
                indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state1]
                indlist2 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state2]
                for vi, invi in indlist1:
                    for vj, invj in indlist2:
                        delta_om_test[vi, vj] += \
                            delom1 * np.dot(self.onsagercalculator.vkinetic.vecvec[vi][invi],
                                            self.onsagercalculator.vkinetic.vecvec[vj][invj])

        # 3a.2 - Next, we consider the contribution by omega3 jumps - mixed to complex
        for jt, jlist in enumerate(self.onsagercalculator.symjumplist_omega3):
            for jmp in jlist:
                indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_mixed[jmp.state1]
                indlist2 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state2]
                for vi, invi in indlist1:
                    for vj, invj in indlist2:
                        delta_om_test[vi, vj] += \
                            omega3[jt] * np.dot(self.onsagercalculator.vkinetic.vecvec[vi][invi],
                                                self.onsagercalculator.vkinetic.vecvec[vj][invj])

        # 3a.3 - Next, we consider the contribution by only the omega4 jumps - complex to mixed
        for jt, jlist in enumerate(self.onsagercalculator.symjumplist_omega4):
            for jmp in jlist:
                indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state1]
                indlist2 = self.onsagercalculator.vkinetic.stateToVecStar_mixed[jmp.state2]
                for vi, invi in indlist1:
                    for vj, invj in indlist2:
                        delta_om_test[vi, vj] += \
                            omega4[jt] * np.dot(self.onsagercalculator.vkinetic.vecvec[vi][invi],
                                                self.onsagercalculator.vkinetic.vecvec[vj][invj])

        # 3b - Now, we do the off diagonal parts
        # 3b.1 - contribution by omega1
        diags = np.zeros(Nvstars)
        for jt, jlist in enumerate(self.onsagercalculator.jnet_1):
            for jmp in jlist:
                si = jmp.state1

                if si.is_zero(self.onsagercalculator.pdbcontainer):
                    if not omega1[jt] == 0.:
                        raise ValueError
                if jmp.state2.is_zero(self.onsagercalculator.pdbcontainer):
                    if not omega1[jt] == 0.:
                        raise ValueError

                star_i = self.onsagercalculator.vkinetic.starset.complexIndexdict[si][1]
                dbwyck_i = self.onsagercalculator.pdbcontainer.invmap[si.db.iorind]

                indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_pure[si]

                for vi, invi in indlist1:
                    vec = self.onsagercalculator.vkinetic.vecvec[vi][invi]
                    vdot = np.dot(vec, vec)
                    if jmp.state2.is_zero(self.onsagercalculator.pdbcontainer) or\
                            jmp.state1.is_zero(self.onsagercalculator.pdbcontainer):
                        diags[vi] -= 0 - np.exp(-bFT0[self.onsagercalculator.om1types[jt]]
                                                                + bFdb0[dbwyck_i] - bFdb0_min) * vdot
                    else:
                        diags[vi] -= np.exp(-bFT1[jt] + bFSdb_total_shift[star_i]) * vdot - \
                                                     np.exp(-bFT0[self.onsagercalculator.om1types[jt]] + bFdb0[
                                                         dbwyck_i] - bFdb0_min) * vdot

        # 3b.2 - contribution by omega4
        for jt, jlist in enumerate(self.onsagercalculator.symjumplist_omega4):
            for jmp in jlist:
                si = jmp.state1

                star_i = self.onsagercalculator.vkinetic.starset.complexIndexdict[si][1]
                #         dbwyck_i = self.onsagercalculator.pdbcontainer.invmap[si.db.iorind]

                indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_pure[si]
                #         indlist2 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state2]

                for vi, invi in indlist1:
                    vec = self.onsagercalculator.vkinetic.vecvec[vi][invi]
                    vdot = np.dot(vec, vec)
                    diags[vi] -= np.exp(-bFT4[jt] + bFSdb_total_shift[star_i]) * vdot

        # 3b.3 - contribution by omega3
        for jt, jlist in enumerate(self.onsagercalculator.symjumplist_omega3):
            for jmp in jlist:
                si = jmp.state1

                star_i = self.onsagercalculator.vkinetic.starset.mixedindexdict[si][1]
                dbwyck_i = self.onsagercalculator.mdbcontainer.invmap[si.db.iorind]

                indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_mixed[si]
                #         indlist2 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state2]

                for vi, invi in indlist1:
                    vec = self.onsagercalculator.vkinetic.vecvec[vi][invi]
                    vdot = np.dot(vec, vec)
                    diags[vi] -= np.exp(-bFT3[jt] + bFdb2[dbwyck_i] - bFdb2_min) * vdot

        # add the diagonal contributions
        for i in range(Nvstars):
            delta_om_test[i, i] += diags[i]

        self.assertTrue(np.allclose(delta_om_test, del_om))

        # NEXT WE VERIFY G0

        # to verify GF20, we need the GF starsets and the GF2 and GF0 lists for each starset
        # setrates has already been run while running makeGF
        GF0 = np.array([self.onsagercalculator.GFcalc_pure(tup[0][0], tup[0][1], tup[1]) for tup in
                        [star[0] for star in self.onsagercalculator.GFstarset_pure]])

        GF2 = np.array([self.onsagercalculator.g2[tup[0][0], tup[0][1]] for tup in
                        [star[0] for star in self.onsagercalculator.GFstarset_mixed]])

        Nvstars = self.onsagercalculator.vkinetic.Nvstars
        Nvstars_pure = self.onsagercalculator.vkinetic.Nvstars_pure

        GFstarset_pure, GFPureStarInd, GFstarset_mixed, GFMixedStarInd = self.onsagercalculator.vkinetic.genGFstarset()

        # Now, we must evaluate the GF20 tensor through explicit summation
        GF20_test = np.zeros((Nvstars, Nvstars))
        # First, we do it for the GF0 part
        for i in range(Nvstars_pure):
            for j in range(Nvstars_pure):
                for si, vi in zip(self.onsagercalculator.vkinetic.vecpos[i], self.onsagercalculator.vkinetic.vecvec[i]):
                    for sj, vj in zip(self.onsagercalculator.vkinetic.vecpos[j],
                                      self.onsagercalculator.vkinetic.vecvec[j]):
                        # try to form a connection between the states
                        try:
                            ds = si ^ sj
                        except:
                            continue
                        # If the connection has formed, locate it's star index
                        Gstarind = GFPureStarInd[ds]
                        GF20_test[i, j] += GF0[Gstarind] * np.dot(vi, vj)

        # Now, we do it for the mixed part
        for i in range(Nvstars_pure, Nvstars):
            for j in range(Nvstars_pure, Nvstars):
                for si, vi in zip(self.onsagercalculator.vkinetic.vecpos[i],
                                  self.onsagercalculator.vkinetic.vecvec[i]):
                    for sj, vj in zip(self.onsagercalculator.vkinetic.vecpos[j],
                                      self.onsagercalculator.vkinetic.vecvec[j]):
                        # try to form a connection between the states
                        s = connector(si.db, sj.db)
                        # If the connection has formed, locate it's star index
                        Gstarind = GFMixedStarInd[s]
                        GF20_test[i, j] += GF2[Gstarind] * np.dot(vi, vj)

        np.allclose(GF20, GF20_test)

        # NEXT, WE TEST THE STATE PROBABILITIES
        complex_prob, mixed_prob = stateprobs
        for prob in complex_prob:
            self.assertTrue(prob >= 0.)
        for prob in mixed_prob:
            self.assertTrue(prob >= 0.)

        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            if state.is_zero(self.onsagercalculator.pdbcontainer):
                continue
            starind = self.onsagercalculator.vkinetic.starset.complexIndexdict[state][1]
            self.assertTrue(np.allclose(np.exp(-bFSdb_total[starind]), complex_prob[i] * part_func))

        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.mixedstates):
            wyckind = self.onsagercalculator.mdbcontainer.invmap[state.db.iorind]
            self.assertTrue(np.allclose(np.exp(-bFdb2[wyckind]), mixed_prob[i] * part_func))

        # Test consistency with omega1 rates
        for jt, jlist in enumerate(self.onsagercalculator.jnet_1):
            for jmp in jlist:
                st1 = jmp.state1
                st2 = jmp.state2
                stateind1, starind1 = self.onsagercalculator.vkinetic.starset.complexIndexdict[st1]
                stateind2, starind2 = self.onsagercalculator.vkinetic.starset.complexIndexdict[st2]

                if st1.is_zero(self.onsagercalculator.pdbcontainer):
                    self.assertTrue(np.allclose(complex_prob[stateind1], 0.))
                    continue
                if st2.is_zero(self.onsagercalculator.pdbcontainer):
                    self.assertTrue(np.allclose(complex_prob[stateind2], 0.))
                    continue

                rate = np.exp(-bFT1[jt] + bFSdb_total_shift[starind1])
                symrate = np.sqrt(complex_prob[stateind1]) * rate * 1. / np.sqrt(complex_prob[stateind2])
                self.assertTrue(np.allclose(symrate, omega1[jt]))

        # Test consistency with omega2 rates
        for jt, jlist in enumerate(self.onsagercalculator.jnet2):
            for jmp in jlist:
                st1 = jmp.state1
                st2 = jmp.state2
                stateind1, wyck1 = self.onsagercalculator.mdbcontainer.db2ind(st1.db),\
                                   self.onsagercalculator.mdbcontainer.invmap[st1.db.iorind]

                stateind2, wyck2 = self.onsagercalculator.mdbcontainer.db2ind(st2.db),\
                                   self.onsagercalculator.mdbcontainer.invmap[st2.db.iorind]

                rate = np.exp(-bFT2[jt] + bFdb2 - bFdb2_min)
                symrate = np.sqrt(mixed_prob[stateind1]) * rate * 1 / np.sqrt(mixed_prob[stateind2])
                self.assertTrue(np.allclose(symrate, omega2[jt]))

        # Test consistency of omega3 and omega4 rates
        for jt, jlist in enumerate(self.onsagercalculator.symjumplist_omega3):
            for jmp in jlist:
                st1 = jmp.state1
                st2 = jmp.state2
                stateind1, wyck1 = self.onsagercalculator.mdbcontainer.db2ind(st1.db), \
                                   self.onsagercalculator.mdbcontainer.invmap[st1.db.iorind]

                stateind2, starind2 = self.onsagercalculator.vkinetic.starset.complexIndexdict[st2]

                rate3 = np.exp(-(bFT3[jt] + bFdb2_min) + bFdb2[wyck1])
                symrate3 = np.sqrt(mixed_prob[stateind1]) * rate3 * 1. / np.sqrt(complex_prob[stateind2])
                if not np.allclose(omega3[jt], symrate3):
                    print(jt, symrate3, omega3[jt])

                rate4 = np.exp(-(bFT4[jt] + bFdb0_min + bFS_min) + bFSdb_total[starind2])
                symrate4 = 1. / np.sqrt(mixed_prob[stateind1]) * rate4 * np.sqrt(complex_prob[stateind2])
                self.assertTrue(np.allclose(omega4[jt], symrate4))

                self.assertTrue( np.allclose(bFT4[jt] + bFdb0_min + bFS_min, bFT3[jt] + bFdb2_min))

                self.assertTrue(np.allclose(symrate3, symrate4))

class test_distorted(test_dumbbell_mediated):
    def setUp(self):
        # We test a new weird lattice because it is more interesting
        # latt = np.array([[0., 0.1, 0.5], [0.3, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        # self.DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # # keep it simple with [1.,0.,0.] type orientations for now
        # o = np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126
        # famp0 = [o.copy()]
        # family = [famp0]

        latt = np.array([[0., 0.1, 0.5], [0.3, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        self.DC_Si = Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
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