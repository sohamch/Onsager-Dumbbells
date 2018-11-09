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
        self.family = [famp0,famp12]
        self.crys = tet2
        # self.pairs_pure = genpuresets(tet2,0,family)
        # self.pairs_mixed = genmixedsets(tet2,0,family)
        #generating pairs from families is now taken care of within states

    def test_dbStates(self):
        #check that symmetry analysis is correct
        dbstates=dbStates(self.crys,0,self.family)
        self.assertEqual(len(dbstates.symorlist),4)
        #check that every (i,or) set is accounted for
        sm=0
        for i in dbstates.symorlist:
            sm += len(i)
        self.assertEqual(sm,len(dbstates.iorlist))

        #test group operations
        db=dumbbell(1, np.array([1.,1.,0.]), np.array([1,1,0]))
        Glist = list(self.crys.G)
        x = np.random.randint(0,len(Glist))
        g = Glist[x] #select random groupop
        newdb_test = db.gop(self.crys,0,g)
        newdb, p = dbstates.gdumb(g,db)
        count=0
        if(newdb_test==newdb):
            self.assertEqual(p,1)
            count=1
        elif(newdb_test==-newdb):
            self.assertEqual(p,-1)
            count=1
        self.assertEqual(count,1)

    def test_mStates(self):
        dbstates=dbStates(self.crys,0,self.family)
        mstates1 = mStates(self.crys,0,self.family)

        #check that symmetry analysis is correct
        self.assertEqual(len(mstates1.symorlist),4)

        #check that negative orientations are accounted for
        for i in range(4):
            self.assertEqual(len(mstates1.symorlist[i])/len(dbstates.symorlist[i]),2)

        #check that every (i,or) set is accounted for
        sm=0
        for i in mstates1.symorlist:
            sm += len(i)
        self.assertEqual(sm,len(mstates1.iorlist))
