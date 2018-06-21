import numpy as np
import onsager.crystal as crystal
from gensets import *
from test_structs import *
import unittest

class test_sets(unittest.TestCase):

    def test_genpurestates(self):
        famp0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        famp12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        family = [famp0,famp12]
        pairs_pure = genpuresets(tet2,0,family)

        self.assertEqual(len(pairs_pure),len(tet2.sitelist(0)))
        self.assertEqual(len(pairs_pure[0]),4)
        self.assertEqual(len(pairs_pure[1]),12)
