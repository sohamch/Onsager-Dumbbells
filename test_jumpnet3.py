import numpy as np
from representations import *
from jumpnet3 import *
from test_structs import *
from states import *
# from gensets import *
import unittest

class test_jumpnetwork(unittest.TestCase):
    def setUp(self):
         famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
         self.family = [famp0]
         # self.pairs_pure = genpuresets(cube,0,family)

    def test_purejumps(self):
        pdbcontainer = dbStates(cube,0,self.family)
        jset,jind = purejumps(pdbcontainer,0.3,0.01,0.01)
        test_dbi = dumbbell(0, np.array([0.126,0.,0.]),np.array([0,0,0]))
        test_dbf = dumbbell(0, np.array([0.126,0.,0.]),np.array([0,1,0]))
        count=0
        for i,jlist in enumerate(jset):
            for q,j in enumerate(jlist):
                if j.state1 == test_dbi or j.state1 == -test_dbi:
                    if j.state2 == test_dbf or j.state2 == -test_dbf:
                        if j.c1 == j.c2 == -1:
                           count += 1
                           jtest = jlist
        self.assertEqual(count,1) #see that this jump has been taken only once into account
        self.assertEqual(len(jtest),24)

        #test_indices
        #First check if they have the same number of lists and elements
        self.assertEqual(len(jind),len(jset))
        #now check if all the elements are correctly correspondent
        for lindex in range(len(jind)):
            self.assertEqual(len(jind[lindex]),len(jset[lindex]))
            for jindex in range(len(jind[lindex])):
                (i1,o1) = pdbcontainer.iorlist[jind[lindex][jindex][0]]
                (i2,o2) = pdbcontainer.iorlist[jind[lindex][jindex][1]]
                self.assertEqual(jset[lindex][jindex].state1.i,i1)
                self.assertEqual(jset[lindex][jindex].state2.i,i2)
                self.assertTrue(np.allclose(jset[lindex][jindex].state1.o,o1))
                self.assertTrue(np.allclose(jset[lindex][jindex].state2.o,o2))


    def test_mixedjumps(self):
        mdbcontainer = mStates(cube,0,self.family)
        #check for the correct number of states
        jset,jind = mixedjumps(mdbcontainer,0.3,0.01,0.01)
        test_dbi = dumbbell(0, np.array([0.126,0.,0.]),np.array([0,0,0]))
        test_dbf = dumbbell(0, np.array([0.126,0.,0.]),np.array([0,1,0]))
        count=0
        for i,jlist in enumerate(jset):
            for q,j in enumerate(jlist):
                if j.state1.db == test_dbi:
                    if j.state2.db == test_dbf:
                        if j.c1 == j.c2 == 1:
                           count += 1
                           jtest = jlist
        self.assertEqual(count,1) #see that this jump has been taken only once into account
        self.assertEqual(len(jtest),24)

        #check if conditions for mixed dumbbell transitions are satisfied
        count=0
        for jl in jset:
            for j in jl:
                if j.c1==-1 or j.c2==-1:
                    count+=1
                    break
                if not (j.state1.i_s==j.state1.db.i and j.state2.i_s==j.state2.db.i and np.allclose(j.state1.R_s,j.state1.db.R) and np.allclose(j.state2.R_s,j.state2.db.R)):
                    count+=1
                    break
            if count==1:
                break
        self.assertEqual(count,0)

        #test_indices
        #First check if they have the same number of lists and elements
        self.assertEqual(len(jind),len(jset))
        #now check if all the elements are correctly correspondent
        for lindex in range(len(jind)):
            self.assertEqual(len(jind[lindex]),len(jset[lindex]))
            for jindex in range(len(jind[lindex])):
                (i1,o1) = mdbcontainer.iorlist[jind[lindex][jindex][0]]
                (i2,o2) = mdbcontainer.iorlist[jind[lindex][jindex][1]]
                self.assertEqual(jset[lindex][jindex].state1.db.i,i1)
                self.assertEqual(jset[lindex][jindex].state2.db.i,i2)
                self.assertTrue(np.allclose(jset[lindex][jindex].state1.db.o,o1))
                self.assertTrue(np.allclose(jset[lindex][jindex].state2.db.o,o2))
