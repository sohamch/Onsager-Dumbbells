import numpy as np
import onsager.crystal as crystal
from states import *
from representations import *
from test_structs import *
import unittest

class test_sets(unittest.TestCase):
    def setUp(self):
        rot = np.array([[ 1,  0,  0],[ 0,  1,  0],[ 0,  0, -1]])
        trans=np.array([0., 0., 0.])
        cartrot=np.array([[ 1.,  0.,  0.],[ 0.,  1.,  0.],[ 0.,  0., -1.]])
        indexmap=((0, 1, 2),)
        self.g = crystal.GroupOp(rot,trans,cartrot,indexmap)

        famp0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        famp12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        family = [famp0,famp12]
        dbStates_tet2 = dbStates(tet2,0,family)
        self.dbStates_tet2 = dbStates_tet2
        self.pairs_pure = dbStates_tet2.iorlist

    def test_dbStates(self):

        #test set creation
        self.assertEqual(len(self.pairs_pure),len(tet2.sitelist(0)))
        self.assertEqual(len(self.pairs_pure[0]),4)
        self.assertEqual(len(self.pairs_pure[1]),12)

        #test group operation on jump
        o1 = np.array([-1.,-1.,0])
        o2 = np.array([1.,0.,0])
        db1 = dumbbell(0,o1,np.array([1,0,0]))
        db2 = dumbbell(0,o2,np.array([2,0,0]))
        jmp = jump(db1,db2,1,1)
        jmp_new,(p1,p2) = self.dbStates_tet2.gdumb(self.g,jmp)
        jmp_new2 = jmp.gop(tet2,0,self.g)
        self.assertEqual(jmp_new,jmp_new2)
        if any(x[0]==0 and np.allclose(np.array([-1.,-1.,0]),x[1]) for lis in self.pairs_pure for x in lis):
            self.assertEqual(p1,1)
        else:
            self.assertEqual(p1,-1)

        if any(x[0]==0 and np.allclose(np.array([1.,0.,0]),x[1]) for lis in self.pairs_pure for x in lis):
            self.assertEqual(p2,1)
        else:
            self.assertEqual(p2,-1)
