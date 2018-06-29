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
        self.pairs_mixed = genmixedsets(tet2,0,family)

    def test_dbStates(self):
        dbstates=dbStates(self.crys,0,self.pairs_pure)
        self.assertEqual(len(dbstates.symorlist),4)

    def test_mStates(self):
        dbstates=dbStates(self.crys,0,self.pairs_pure)
        mstates = mStates(self.crys,0,self.pairs_mixed)
        self.assertEqual(len(mstates.symorlist),4)
        for i in range(4):
            self.assertEqual(len(dbstates.symorlist[i])/len(mstates.symorlist[i]),0.5)

        mstates = mStates(self.crys,0,self.pairs_pure) #negative orientations are missing
        self.assertEqual(len(mstates.symorlist),4)
        for i in range(4):
            self.assertEqual(len(dbstates.symorlist[i])/len(mstates.symorlist[i]),0.5)

    def test_pairstates(self):
        famp0 = [np.array([1.,0.,0.])]
        famp12 = [np.array([1.,1.,0.])]
        family = [famp0,famp12]
        pairs_pure = genpuresets(tet2,0,family)
        pairset = genPairSets(tet2,0,pairs_pure,1)
        pset = Pairstates(tet2,0,pairset)
        count=0
        for plis in pset.sympairlist:
            for pair in plis:
                if pair.db.i==0 and np.allclose(pair.db.R,np.array([1,1,0]),atol=1e-8):
                    count=1
                    lis=plis.copy()
                    break
            if count == 1:
                break
        self.assertEqual(len(lis),8)
        sm=0
        for i in pset.sympairlist:
            sm += len(i)
        self.assertEqual(len(pairset),sm)
