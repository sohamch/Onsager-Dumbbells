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
        #check that symmetry analysis is correct
        dbstates=dbStates(self.crys,0,self.pairs_pure)
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
        print(newdb_test)
        print(newdb,p)
        if(newdb_test==newdb):
            self.assertEqual(p,1)
            count=1
        elif(newdb_test==-newdb):
            self.assertEqual(p,-1)
            count=1
        self.assertEqual(count,1)

    def test_mStates(self):
        dbstates=dbStates(self.crys,0,self.pairs_pure)
        mstates1 = mStates(self.crys,0,self.pairs_mixed)

        #check that symmetry analysis is correct
        self.assertEqual(len(mstates1.symorlist),4)

        #check that negative orientations are accounted for
        for i in range(4):
            self.assertEqual(len(dbstates.symorlist[i])/len(mstates1.symorlist[i]),0.5)

        #check that every (i,or) set is accounted for
        sm=0
        for i in mstates1.symorlist:
            sm += len(i)
        self.assertEqual(sm,len(mstates1.iorlist))

        #check without supplying negative orientations
        mstates = mStates(self.crys,0,self.pairs_pure)
        self.assertEqual(len(mstates.symorlist),4)
        for i in range(4):
            self.assertEqual(len(dbstates.symorlist[i])/len(mstates.symorlist[i]),0.5)

        #check validity of group operations
        Glist = list(self.crys.G)
        x = np.random.randint(0,len(Glist))
        g = Glist[x] #select random groupop
        db=dumbbell(1, np.array([1.,1.,0.]), np.array([1,1,0]))
        mdb = SdPair(db.i,db.R,db)
        newmdb = mstates.gdumb(g,mdb)
        self.assertTrue(newmdb.i_s==newmdb.db.i and np.allclose(newmdb.R_s,newmdb.db.R,atol=self.crys.threshold))
        self.assertTrue(any(newmdb.db.i==tup[0] and np.allclose(newmdb.db.o,tup[1],atol=self.crys.threshold) for lis in mstates.symorlist for tup in lis))

    def test_pairstates(self):
        famp0 = [np.array([1.,0.,0.])]
        famp12 = [np.array([1.,1.,0.])]
        family = [famp0,famp12]
        pairs_pure = genpuresets(tet2,0,family)
        pairset = genPairSets(tet2,0,pairs_pure,0.6)
        print("flat pair list done")
        pset = Pairstates(tet2,0,pairs_pure,0.6)
        print("Pairstates object done")
        count=0

        #first extract the test set
        for plis in pset.sympairlist:
            for pair in plis:
                if pair.db.i==0 and pair.i_s == 0 and np.allclose(abs(pair.db.R),np.array([1,1,0]),atol=1e-8):
                    count=1
                    lis=plis.copy()
                    break
            if count == 1:
                break
        #check that symmetry grouping is correct
        #check members of the test set manually once this passes
        self.assertEqual(len(lis),8)

        #check that all of the pairs are accounted for
        sm=0
        for i in pset.sympairlist:
            sm += len(i)
        self.assertEqual(len(pairset),sm)

        #test group operations
        Glist = list(self.crys.G)
        for t in range(5):
            x = np.random.randint(0,len(Glist))
            g = Glist[x] #select random groupop
            db = dumbbell(0,np.array([1.,0.,0.]),np.array([1,1,1]))
            pair = SdPair(0,np.array([0,0,0]),db)
            pairnewtest = pair.gop(self.crys,0,g)
            pairnew,p = pset.gpair(g,pair)
            count=0
            print(pairnewtest)
            print(pairnew,p)
            print()
            if pairnewtest==pairnew:
                self.assertEqual(p,1)
                count=1
            elif pairnewtest==-pairnew:
                self.assertEqual(p,-1)
                count=1
            self.assertEqual(count,1)
            #check that pairnew exists in the states list
            check = False
            for lis in pset.sympairlist:
                for p in lis:
                    if (p==pairnew):
                        check = True
                        break
            self.assertTrue(check)

        #check that all of the pairs are within cutoff distance
        check = False
        for lis in pset.sympairlist:
            for p in lis:
                dx = self.crys.unit2cart(p.db.R,self.crys.basis[0][p.db.i]) - self.crys.unit2cart(p.R_s,self.crys.basis[0][p.i_s])
                if (np.dot(dx,dx) > 0.6**2):
                    check = True
                    break
        self.assertFalse(check)
