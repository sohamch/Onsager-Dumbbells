import numpy as np
import onsager.crystal as crystal
from representations import *
from test_structs import *
from collision import *
import unittest

class collision_tests(unittest.TestCase):
    def test_self(self):
        db1 = dumbbell(0,np.array([0.,0.,0.1]),np.array([0,0,0]))
        db2 = dumbbell(0,np.array([0.,0.,0.1]),np.array([1,0,0]))
        jmp = jump(db1,db2,1,1)
        check = collsion_self(cube,0,jmp,0.001,0.001)
        self.assertTrue(check)

        db3 = dumbbell(0,np.array([-0.1,0.,0.]),np.array([0,0,0]))
        db4 = dumbbell(0,np.array([-0.1,0.,0.]),np.array([1,0,0]))
        jmp2 = jump(db3,db4,1,1)
        check = collsion_self(cube,0,jmp2,0.001,0.001)
        self.assertFalse(check)
