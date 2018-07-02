import numpy as np
import onsager.crystal as crystal
from representations import *
from test_structs import *
from collision import *
import unittest

class collision_tests(unittest.TestCase):
    def test_self(self):
        or1 = np.array([1.,1.,0.])/np.linalg.norm(np.array([1.,1.,0.]))*0.252
        or2 = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.252
        # with simple cubic structure
        db1 = dumbbell(0,or2,np.array([0,0,0]))
        db2 = dumbbell(0,or2,np.array([1,0,0]))
        jmp = jump(db1,db2,-1,1)
        check = collision_self(cube,0,jmp,0.01)
        self.assertTrue(check)

        db3 = dumbbell(0,or2,np.array([0,0,0]))
        db4 = dumbbell(0,or2,np.array([1,0,0]))
        jmp2 = jump(db3,db4,1,-1)
        check = collision_self(cube,0,jmp2,0.001)
        self.assertFalse(check)

        #with BCC structure
        db1 = dumbbell(0,or1,np.array([0,0,0]))
        db2 = dumbbell(0,or1,np.array([1,1,1]))
        jmp = jump(db1,db2,1,-1)
        check = collision_self(Fe_bcc,0,jmp,0.01)
        self.assertFalse(check)

    def test_others(self):
        #construct supervect seperately just for testing phase.
        nmax = [int(np.round(np.sqrt(tet2.metric[i, i]))) + 1 for i in range(3)]
        supervect = [np.array([n0, n1, n2])
                         for n0 in range(-nmax[0], nmax[0] + 1)
                         for n1 in range(-nmax[1], nmax[1] + 1)
                         for n2 in range(-nmax[2], nmax[2] + 1)]

        #Tilted TEST
        or1 = np.array([1.,1.,0.])/np.linalg.norm(np.array([1.,1.,0.]))*0.14
        db1 = dumbbell(2,or1,np.array([0,0,0]))
        db2 = dumbbell(2,or1,np.array([1,-1,0]))
        jmp = jump(db1,db2,1,1)
        check1 = collision_self(tet2,0,jmp,0.01,0.01)
        check2 = collision_others(tet2,0,jmp,supervect,0.01)
        self.assertFalse(check1)
        self.assertFalse(check2)
        #TEST WHERE COLLISION IS SURE TO BE DETECTED
        # or1 = np.array([1.,-1.,0.])/np.linalg.norm(np.array([1.,-1.,0.]))*0.14
        # db1 = dumbbell(2,or1,np.array([0,0,0]))
        # db2 = dumbbell(2,or1,np.array([1,-1,0]))
        # jmp = jump(db1,db2,1,-1)
        # check1 = collision_self(tet2,0,jmp,0.01,0.01)
        # check2 = collision_others(tet2,0,jmp,supervect,0.01)
        # self.assertFalse(check1)
        # self.assertTrue(check2)
        #
        # #TEST WHERE COLLISION IS SURE TO BE NOT DETECTED
        # or1 = np.array([1.2,-1.2,0.])/np.linalg.norm(np.array([1.2,-1.2,0.]))*0.14
        # db1 = dumbbell(2,or1,np.array([0,0,0]))
        # db2 = dumbbell(1,or1,np.array([0,0,0]))
        # jmp = jump(db1,db2,1,-1)
        # check1 = collision_self(tet2,0,jmp,0.01,0.01)
        # check2 = collision_others(tet2,0,jmp,supervect,0.01)
        # self.assertFalse(check1)
        # self.assertFalse(check2)
        #
        # #SURE TEST WITH THE ATOM IN A DIFFERENT SUBLATTICE
        # #TEST WHERE COLLISION IS SURE TO BE DETECTED
        # or1 = np.array([1.2,-1.2,0.])/np.linalg.norm(np.array([1.2,-1.2,0.]))*0.14
        # db1 = dumbbell(1,or1,np.array([0,0,0]))
        # db2 = dumbbell(1,or1,np.array([1,-1,0]))
        # jmp = jump(db1,db2,1,-1)
        # check1 = collision_self(tet3,0,jmp,0.01,0.01)
        # check2 = collision_others(tet3,0,jmp,supervect,0.01)
        # self.assertFalse(check1)
        # self.assertTrue(check2)
