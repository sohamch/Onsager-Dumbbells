import onsager.crystal as crystal
from Onsager_calc_db import *
from test_structs import *
import states
from representations import *
class test_dumbbell_mediated(unittest.TestCase):

    def test_displacements(self):

        #Set up a simple jumpnetwork of three states in a cubic lattice
        famp0 = [np.array([0.126,0.,0.])]
        family = [famp0]
        pdbcontainer = states.dbStates(cube,0,family)
        mdbcontainer = states.mStates(cube,0,family)
        jnet0_states,jnet_0_ind = pdbcontainer.jumpnetwork(0.3,0.01,0.01)
        jnet2_states,jnet_2_ind = pdbcontainer.jumpnetwork(0.3,0.01,0.01)

        jnet0_solute, jnet0_solvent = calc_dx_species(cube,jnet0_states,jnet_0_ind,type="baredb")
        jnet2_solute, jnet2_solvent = calc_dx_species(cube,jnet0_states,jnet_0_ind,type="complex")
        jnet2_solute, jnet2_solvent = calc_dx_species(cube,jnet0_states,jnet_0_ind,type="mdb")

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
        dx_test = np.array([0.063,0.063,0.28])
        dx_solvent = None
        for i,jlist in enumerate(jnet0_states):
            for j,jmp in enumerate(jlist):
                if  np.allclose(jmp_test.state1.o*jmp_test.c1,jmp.state1.o*jmp.c1):
                    if  np.allclose(jmp_test.state2.o*jmp_test.c2,jmp.state2.o*jmp.c2):
                        if np.allclose(jmp.state1.R,jmp_test.state1.R):
                            if np.allclose(jmp.state2.R,jmp_test.state2.R):
                            #first check that the correct displacement is applied
                                dx_solvent = jnet0_solvent[i][j][1]
        self.assertFalse(dx_solvent==None)
        self.assertTrue(np.allclose(dx_solvent,dx_test))
        self.assertTrue(jnet0_solute==None)
