import numpy as np
import onsager.crystal as crystal
from states import *
from representations import *
from test_structs import *
import unittest

class test_sets(unittest.TestCase):

    def test_dbStates(self):
        famp0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        famp12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        family = [famp0,famp12]
        dbStates_tet2 = dbStates(tet2,0,family)
        pairs_pure = dbStates_tet2.iorlist
        #test set creation
        self.assertEqual(len(pairs_pure),len(tet2.sitelist(0)))
        self.assertEqual(len(pairs_pure[0]),4)
        self.assertEqual(len(pairs_pure[1]),12)

        #test group operation on jump
        o1 = np.array([-1.,-1.,0])
        o2 = np.array([1.,0.,0])
        db1 = dumbbell(0,o1,np.array([1,0,0]))
        db2 = dumbbell(0,o2,np.array([2,0,0]))
        jmp = jump(db1,db2,1,1)
        listG = list(tet.G)
        op = 1#np.random.randint(0,len(listG))
        g = listG[op]
        jmp_new,(p1,p2) = dbStates_tet2.gdumb(g,jmp)
        self.assertEqual(p1,1)
        self.assertEqual(p2,-1)
