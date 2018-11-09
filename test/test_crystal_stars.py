import numpy as np
from jumpnet3 import *
from stars import *
from test_structs import *
from states import *
from gensets import *
import unittest

class test_StarSet(unittest.TestCase):

    def setUp(self):
        famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
        family = [famp0]
        pairs_pure = genpuresets(cube,0,family)
        jset = purejumps(cube,0,pairs_pure,0.3,0.01,0.01)
        self.crys_stars = StarSet(cube,0,jset,pairs_pure,2)

    def test_generate(self):
        #test if the starset is generated correctly
        o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
        db = dumbbell(0,o,np.array([1,0,0]))
        for l in self.crys_stars.starset:
            for state in l:
                if state.db==db or state.db==-db:
                    test_list = l.copy()
        self.assertEqual(len(test_list),6)

    def test_jumpnetworks(self):
        #See the example file. Provides much clearer understanding.
