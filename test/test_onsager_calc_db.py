import onsager.crystal as crystal
from Onsager_calc_db import *
from test_structs import *
import states
from representations import *
class test_dumbbell_mediated(unittest.TestCase):

    def test_displacements(self):

        #Set up a simple jumpnetwork of three states in a cubic lattice
        famp0 = [np.array([1.,0.,0.])]
        family = [famp0]
        pdbcontainer = states.dbStates(cube,0,family)
        mdbcontainer = states.mStates(cube,0,family)
        jnet0_states,jnet_0_ind = pdbcontainer.jumpnetwork(0.3,0.01,0.01)
        jnet2_states,jnet_2_ind = pdbcontainer.jumpnetwork(0.3,0.01,0.01)

        jnet0_solute = calc_dx_species(cube,jnet0_states,jnet_0_ind,species='solute')
        jnet0_solvent = calc_dx_species(cube,jnet0_states,jnet_0_ind,species='solvent')

        #We'll test if we have the correct displacement for the following jump, which should be there
#         Jump object:
#           Initial state:
# 	             dumbbell :basis index = 0, lattice vector = [0 0 0], orientation = [0. 1. 0.]
#           Final state:
# 	             dumbbell :basis index = 0, lattice vector = [0 0 1], orientation = [0. 1. 0.]
#                Jumping from c = -1 to c= -1
        db1 = dumbbell(0,np.array([0.,1.,0.]),np.array([0,0,0],dtype=int))
        db2 = dumbbell(0,np.array([0.,1.,0.]),np.array([0,0,1],dtype=int))
        jmp_test = jump(db1,db2,-1,-1)
        for i,jlist in enumerate(jnet0_solute):
            for j,jmp in jlist:
                if  jmp_test==jmp:
                    #first check that the correct displacement is applied
                    dx_test = states.disp(cube,0,jmp_test.db1,jmp_test.db2)
