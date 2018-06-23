import numpy as np
import onsager.crystal as crystal
from states import *
from representations import *
from test_structs import *
from gensets import *
import unittest

class test_sets(unittest.TestCase):
    def setUp(self):
        famp0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        famp12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        family = [famp0,famp12]
        self.crys = tet2
        self.pairs_pure = genpuresets(tet2,0,family)

    def test_dbStates(self):
        dbstates=dbStates(self.crys,0,self.pairs_pure)
        self.assertEqual(len(dbstates.symorlist),4)
