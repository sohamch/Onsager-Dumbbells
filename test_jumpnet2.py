import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple
from representations import *
from jumpnet2 import *
import unittest

class test_gen_orsets(unittest.TestCase):

    def test_gen_orsets(self):
        #test that the correct number of orientations are captured
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        fam_p = np.array([1.,1.,0.])
        fam_m = np.array([1.,1.,0.])
        plist, mlist = gen_orsets(crys,0,[fam_p],[fam_m])
        self.assertEqual(len(plist),1)
        self.assertEqual(len(mlist),1)
        self.assertEqual(len(plist[0]),2)
        self.assertEqual(len(mlist[0]),4)
