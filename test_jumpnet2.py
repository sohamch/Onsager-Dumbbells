import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple
from representations import *
from jumpnet2 import *
from test_structs import *
import unittest

class test_gen_orsets(unittest.TestCase):

    def test_gen_orsets(self):
        #test that the correct number of orientations are captured
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        fam_p0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        purelist=[fam_p0]
        fam_m0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        mixlist=[fam_m0]
        plist, mlist = gen_orsets(crys,0,purelist,mixlist)
        self.assertEqual(len(plist),1)
        self.assertEqual(len(mlist),1)
        self.assertEqual(len(plist[0]),4)
        self.assertEqual(len(mlist[0]),8)

        #For a more complicated omega-Ti lattice
        fam_p0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        fam_p12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        purelist=[fam_p0,fam_p12]
        fam_m0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        fam_m12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        mixlist=[fam_m0,fam_m12]
        plist, mlist = gen_orsets(omega_Ti,0,purelist,mixlist)
        self.assertEqual(len(plist),2)
        self.assertEqual(len(mlist),2)
        # self.assertEqual(len(plist[0]),4)
        # self.assertEqual(len(mlist[0]),8)
    def test_gensets(self):
        fam_p0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        fam_p12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        purelist=[fam_p0,fam_p12]
        fam_m0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        fam_m12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        mixlist=[fam_m0,fam_m12]
        plist, mlist = gen_orsets(omega_Ti,0,purelist,mixlist)
        pairs_pure,pairs_mixed = gensets(omega_Ti,0,plist,mlist)
        self.assertEqual(len(pairs_pure),3)
        self.assertEqual(len(pairs_mixed),3)
