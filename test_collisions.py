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
        jmp = jump(db1,db2,1,1)
        check = collsion_self(cube,0,jmp,0.01)
        self.assertTrue(check)

        db3 = dumbbell(0,or2,np.array([0,0,0]))
        db4 = dumbbell(0,or2,np.array([1,0,0]))
        jmp2 = jump(db3,db4,1,-1)
        check = collsion_self(cube,0,jmp2,0.001)
        self.assertFalse(check)

        #with BCC structure
        db1 = dumbbell(0,or1,np.array([0,0,0]))
        db2 = dumbbell(0,or1,np.array([1,1,1]))
        jmp = jump(db1,db2,1,-1)
        check = collsion_self(Fe_bcc,0,jmp,0.01)
        self.assertFalse(check)

    def test_others(unittest.TestCase):
        or1 = np.array([1.2,1.2,0.])/np.linalg.norm(np.array([1.2,1.2,0.]))*0.14
        db1 = dumbbell(2,or2,np.array([0,0,0]))
        db2 = dumbbell(2,or2,np.array([1,-1,0]))
        jmp = jump(db1,db2,1,-1)
        check1 = collsion_self(tet2,0,jmp,0.01,0.01)
        check2 = collsion_others(tet2,0,jmp,0.01)
        self.assertFalse(check1)
        self.assertTrue(check2)
