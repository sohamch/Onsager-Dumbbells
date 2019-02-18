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

        self.jset0,self.jset2 = self.pdbcontainer_si.jumpnetwork(0.4,0.01,0.01), self.mdbcontainer_si.jumpnetwork(0.4,0.01,0.01)

        self.crys_stars = StarSet(self.pdbcontainer_si,self.mdbcontainer_si,self.jset0,self.jset2, Nshells=2)
        self.vec_stars = vectorStars(self.crys_stars)

        #generate 1, 3 and 4 jumpnetworks
        (self.jnet_1,self.jnet_1_indexed), self.jtype = self.crys_stars.jumpnetwork_omega1()
        (self.symjumplist_omega43_all,self.symjumplist_omega43_all_indexed),(self.symjumplist_omega4,self.symjumplist_omega4_indexed),(self.symjumplist_omega3,self.symjumplist_omega3_indexed)=self.crys_stars.jumpnetwork_omega34(0.4,0.01,0.01,0.01)

        self.W1list = np.ones(len(self.jnet_1))
        self.W2list = np.ones(len(self.jset2[0]))
        self.W3list = np.ones(len(self.symjumplist_omega3))
        self.W4list = np.ones(len(self.symjumplist_omega4))

        #generate all the bias expansions - will separate out later
        self.biases = self.vec_stars.biasexpansion(self.jnet_1,self.jset2[0],self.jtype,self.symjumplist_omega43_all)

        self.onsagercalculator=dumbbellMediated(self.pdbcontainer_si,self.mdbcontainer_si,0.4,0.01,0.01,0.01,NGFmax=4,Nthermo=0)

    def test_displacements(self):

        #Set up a simple jumpnetwork of three states in a cubic lattice
        famp0 = [np.array([0.126,0.,0.])]
        family = [famp0]
        pdbcontainer = states.dbStates(cube,0,family)
        mdbcontainer = states.mStates(cube,0,family)
        jnet0_states,jnet_0_ind = pdbcontainer.jumpnetwork(0.3,0.01,0.01)
        jnet2_states,jnet_2_ind = mdbcontainer.jumpnetwork(0.3,0.01,0.01)

        jnet0_solute, jnet0_solvent = calc_dx_species(cube,jnet0_states,jnet_0_ind,type="bare")
        jnet2_solute, jnet2_solvent = calc_dx_species(cube,jnet2_states,jnet_2_ind,type="mixed")

        #We'll test if we have the correct displacement for the following jump, which should be there
#         Jump object:
#           Initial state:
# 	             dumbbell :basis index = 0, lattice vector = [0 0 0], orientation = [0. 1. 0.]
#           Final state:
# 	             dumbbell :basis index = 0, lattice vector = [0 0 1], orientation = [0. 1. 0.]
#                Jumping from c = -1 to c= -1
        db1 = dumbbell(0,np.array([0.,.126,0.]),np.array([0,0,0],dtype=int))
        db2 = dumbbell(0,np.array([.126,0.,0.]),np.array([0,0,1],dtype=int))
        jmp_test = jump(db1,db2,-1,-1)
        dx_test = np.array([0.0,0.0,0.28])
        # dx_solvent = None
        for i,jlist in enumerate(jnet0_states):
            for j,jmp in enumerate(jlist):
                if  np.allclose(jmp_test.state1.o*jmp_test.c1,jmp.state1.o*jmp.c1):
                    if  np.allclose(jmp_test.state2.o*jmp_test.c2,jmp.state2.o*jmp.c2):
                        if np.allclose(jmp.state1.R,jmp_test.state1.R):
                            if np.allclose(jmp.state2.R,jmp_test.state2.R):
                                dx_solvent = jnet0_solvent[i][j][1]                                #
        # self.assertFalse(dx_solvent==None)
        self.assertTrue(np.allclose(dx_solvent,dx_test),msg="dx_solvent,dx_test = {}, {}".format(dx_solvent,dx_test))
        # self.assertTrue(jnet0_solute==None)

    def test_calc_eta(self):
        pass
