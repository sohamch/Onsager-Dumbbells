import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
import unittest
from representations import *
from collections import namedtuple


# Single dumbbell defect in a lattice. The representation should satisfy the follwing tests.
# 1. Equality testing - Test whether the \__eq__ function performs as expected.
# 2. Addition of a jump object to a state - Test whether the \__add__ function performs as expected.
# 3. Application of group operations - Test whether correct number of states are produced that are symmetry equivalent.

class DB_Tests(unittest.TestCase):
    """
    Tests related to a single dumbell diffusing in a lattice.
    Test case - tetragonal lattice.
    Format of a dumbbell object - "i o R (+-)1" -> basis site, orientation, unit cell location, jumping
                                   atom indicator respectively.
    """
    def test_equality(self):
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))#orientation 1
        or_2 = np.array([0.,1.,1.])/la.norm(np.array([0.,1.,1.]))#orientation 2
        db1 = dumbbell(0,or_1,np.array([0.,0.,0.]))#create a dumbbell state
        db2 = dumbbell(0,or_2,np.array([1.,0.,0.]))#db2=db3 != db1 deliberately
        db3 = dumbbell(0,or_2,np.array([1.,0.,0.]))
        self.assertEqual(db3,db2) #test equality and inequality operators
        self.assertNotEqual(db1,db3)

    #Test Application of group operations
    def test_gop(self):
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        or_1 = np.array([0.,1.,1.])
        db1 = dumbbell(0,or_1,np.array([1.,0.,0.]))
        dblist=[]
        dblist.append(db1)
        for g in crys.G:
            dbnew = db1.gop(crys,0,g)
            if not any(db2==dbnew for db2 in dblist):
                dblist.append(dbnew)
        self.assertEqual(len(dblist),8)

# Test For solute-dumbbell pairs
# 1. Equality testing - Test whether the \__eq__ function performs as expected.
# 2. Adding a jump - Test whether \__add__ performs as expected
# 3. Test whether group operations generate the correct number of symmetrically equivalent pairs

class SdPair_Tests(unittest.TestCase):
    """
    Tests related to a single dumbell diffusing in a lattice.
    Test case - tetragonal lattice.
    Format of a pair object - "i_s R_s db" -> solute location (state), dumbbell state
    """
    #test equality comparison
    def test_equality(self):
        or_1 = np.array([0.,0.,1.])
        or_2 = np.array([0.,1.,1.])
        or_x = np.array([1.,0.,0.])

        db1 = dumbbell(0,or_1,np.array([0.,0.,0.]))#create a dumbbell state
        db1n = dumbbell(0,or_1,np.array([0.,0.,0.]))#create a dumbbell state
        db2 = dumbbell(0,or_2,np.array([1.,0.,0.]))

        pair1 = SdPair(0,np.array([1.,0.,0.]),db1)
        pair2 = SdPair(0,np.array([1.,0.,0.]),db2)
        pair3 = SdPair(0,np.array([1.,0.,0.]),db2)
        self.assertEqual(pair3,pair2) #test equality and inequality
        self.assertNotEqual(pair1,pair3)

    #test addition with a jump object

    def test_gop(self):
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        or_1 = np.array([0.,0.,1.])
        db1 = dumbbell(0,or_1,np.array([1.,0.,0.]),1)
        pair1 = SdPair(0,np.array([1.,0.,0.]),db1)
        pair_list=[]
        for g in crys.G:
            pairn = pair1.gop(crys,0,g)
            if not any(pair==pairn for pair in pair_list):
                pair_list.append(pairn)
        self.assertEqual(len(pair_list),8)
        c_neg=0
        c_pos=0
        for i in range(8):
            if pair_list[i].db.c == 1:
                c_pos+=1
            if pair_list[i].db.c == -1:
                c_neg+=1
        self.assertEqual(c_pos,8)
        self.assertEqual(c_neg,0)

# For jump objects, need to check the following two for now:
# 1. Addition of jump objects to produce third jump object
# 2. Applying group operations to produce symmetry equivalent jump objects.

class jump_Tests(unittest.TestCase):

    def test_jump_validation(self):
        #Test that invalid jumps are caught for mixed dumbbells
        #Check that solute does not jump in a seperated solute-dumbbell pair.
        or_1 = np.array([0.,0.,1.])
        R1 = np.array([1.,0.,0.])
        db1 = dumbbell(0,or_1,R1,1)
        db2 = dumbbell(0,or_1,R1,-1)
        pair1 = SdPair(0,np.array([0.,1.,0.]),db1)
        pair2 = SdPair(0,np.array([1.,0.,0.]),db2)
        with self.assertRaises(ArithmeticError):
            j=jump(pair1,pair2)

        #Check that no other change except flipping of 'c' is caught
        or_1 = np.array([0.,0.,1.])
        R1 = np.array([1.,0.,0.])
        db1 = dumbbell(0,or_1,R1,1)
        db2 = dumbbell(0,or_1,R1,-1)
        pair1 = SdPair(0,np.array([1.,0.,0.]),db1)
        pair2 = SdPair(0,np.array([1.,0.,0.]),db2)
        with self.assertRaises(ArithmeticError):
            j=jump(pair1,pair2)

        #Check that if solute atom jumps from a mixed dumbbell, then not generating another mixed dumbbell is caught
        R2 = np.array([0.,0.,0.])
        or_2 = np.array([0.,1.,1.])/la.norm(np.array([0.,1.,1.]))
        db3 = dumbbell(0,or_2,R2,-1)
        pair3 = SdPair(0,np.array([1.,0.,0.]),db2)
        with self.assertRaises(ArithmeticError):
            j=jump(pair1,pair3)

        #Check that if solvent atom jumps from a mixed dumbbell, then generating another mixed dumbbell is caught
        db1 = dumbbell(0,or_1,np.array([2.,0.,0.]),-1)
        pair1 = SdPair(0,np.array([2.,0.,0.]),db1)
        db2 = dumbbell(0,or_1,R1,1)
        pair2 = SdPair(0,np.array([1.,0.,0.]),db2)
        with self.assertRaises(ArithmeticError):
            j=jump(pair1,pair2)

    def test_add(self):
        #Addition of jumps
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        or_2 = np.array([0.,1.,1.])/la.norm(np.array([0.,1.,1.]))
        or_3 = np.array([1.,0.,1.])/la.norm(np.array([1.,0.,1.]))

        R1 = np.array([0.,1.,0.])
        R2 = np.array([0.,0.,0.])
        R3 = np.array([0.,0.,1.])

        db1 = dumbbell(0,or_1,R1,1)
        db2 = dumbbell(0,or_2,R2,1)
        db3 = dumbbell(0,or_2,R2,1)

        j1 = jump(db1,db2)
        j2 = jump(db2,db3)
        j3 = jump(db1,db3)
        with self.assertRaises(ArithmeticError):
            j3+j1
        self.assertEqual(j3,j1+j2)

        #Addition of jumps and dumbbells.
        with self.assertRaises(ArithmeticError):
            db2+j1
        self.assertEqual(db1+j1,db2)
        self.assertEqual(j1+db1,db2)

    def test_gop(self):
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        or_1 = np.array([1.,0.,0.])
        or_2 = np.array([0.,1.,0.])
        db1 = dumbbell(0,or_1,np.array([0,1,0]),1)
        db2 = dumbbell(0,or_1,np.array([1,0,0]),-1)
        j = jump(db1,db2)
        j_list=[]
        for g in crys.G:
            j_new = j.gop(crys,0,g)
            if not any (j2==j_new for j2 in j_list):
                j_list.append(j_new)
        self.assertEqual(len(j_list),4)
