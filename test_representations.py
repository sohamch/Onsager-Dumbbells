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
        db1 = dumbbell(0,or_1,np.array([1.,0.,0.]))
        pair1 = SdPair(0,np.array([1.,0.,0.]),db1)
        pair_list=[]
        pair_list.append(pair1)
        for g in crys.G:
            pairn = pair1.gop(crys,0,g)
            if not any(pair==pairn for pair in pair_list):
                pair_list.append(pairn)
        self.assertEqual(len(pair_list),8)

# For jump objects, need to check the following two for now:
# 1. Addition of jump objects to produce third jump object
# 2. Applying group operations to produce symmetry equivalent jump objects.

class jump_Tests(unittest.TestCase):

    def test_jump_validation(self):
        #Test that invalid jumps are caught for mixed dumbbells
        #Check that solute does not jump in a seperated solute-dumbbell pair.
        or_1 = np.array([0.,0.,1.])
        R1 = np.array([1,0,0])
        db1 = dumbbell(0,or_1,R1)
        db2 = dumbbell(0,or_1,R1)
        db3 = dumbbell(0,or_1,np.array([0,1,0]))
        pair1 = SdPair(0,np.array([0,1,0]),db1)
        pair2 = SdPair(0,np.array([1,0,0]),db2)
        with self.assertRaises(ArithmeticError):
            j=jump(pair1,pair2,1,1)

        #check that mixed dumbbell jumps are created properly
        pair3 = SdPair(0,np.array([0,2,0]),db3)
        with self.assertRaises(ArithmeticError):
            j=jump(pair2,pair3,1,1)
        #Check that if solute atom jumps from a mixed dumbbell, then not generating another mixed dumbbell is caught
        with self.assertRaises(ArithmeticError):
            j=jump(pair2,pair1,1,1)
        pair3 = SdPair(0,np.array([0.,1.,0.]),db3)
        #check that if solvent atom jumps, not another mixed dumbbell is created
        with self.assertRaises(ArithmeticError):
            j=jump(pair2,pair3,-1,-1)
        #check that not indicating the same active atom in intial and final atom for mixed dumbbells is caught.
        with self.assertRaises(ArithmeticError):
            j=jump(pair2,pair3,1,-1)

    def test_add(self):
        #Addition of jumps
        #1. Test with dumbbells
        or_1 = np.array([0,0,1])
        or_2 = np.array([0,1,0])
        db = dumbbell(0,or_1,np.array([1,0,0]))
        db1 = dumbbell(0,or_1,np.array([0,0,0]))
        db2 = dumbbell(0,or_2,np.array([0,1,0]))
        j = jump(db1,db2,1,1)
        dbf = db + j
        dbftrue = dumbbell(0,or_2,np.array([1,1,0]))
        self.assertEqual(dbf,dbftrue)
        dbffalse = dumbbell(0,or_1,np.array([1,1,0]))
        self.assertNotEqual(dbf,dbffalse)
        dbffalse = dumbbell(0,or_2,np.array([0,1,0]))
        self.assertNotEqual(dbf,dbffalse)

        #Test addition for separated solute-dumbell pair - translation of jumps
        or_1 = np.array([1,0,0])
        or_2 = np.array([1,1,0])
        db1 = dumbbell(0,or_1,np.array([-1,0,0]))
        db2 = dumbbell(0,or_2,np.array([-1,-1,0]))
        pair1 = SdPair(0,np.array([0,0,0]),db1)
        db1shift = dumbbell(db1.i,db1.o,db1.R + np.array([0,1,0]))
        pair1_shift = SdPair(0,np.array([0,1,0]),db1shift)
        pair2 = SdPair(0,np.array([0,0,0]),db2)
        j = jump(pair1,pair2,1,1)
        pair3 = pair1_shift + j
        db3 = dumbbell(db2.i,db2.o,np.array([-1,0,0]))
        pair3true = SdPair(0,np.array([0,1,0]),db3)
        self.assertEqual(pair3,pair3true)

        #Test addition for mixed dumbbell - translation of jumps
        or_1 = np.array([-1,0,0])
        or_2 = np.array([-1,0,0])
        db1 = dumbbell(0,or_1,np.array([0,0,0]))
        db2 = dumbbell(0,or_2,np.array([0,-1,0]))
        pair1 = SdPair(0,np.array([0,0,0]),db1)
        pair2 = SdPair(0,np.array([0,-1,0]),db2)
        db1shift = dumbbell(db1.i,db1.o,db1.R + np.array([0,1,0]))
        pair1_shift = SdPair(0,np.array([0,1,0]),db1shift)
        j = jump(pair1,pair2,1,1)
        pair3 = pair1_shift + j
        db3 = dumbbell(db2.i,db2.o,np.array([0,0,0]))
        pair3true = SdPair(0,np.array([0,0,0]),db3)
        self.assertEqual(pair3,pair3true)
        self.assertNotEqual(pair3,pair2)


    def test_gop(self):
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        or_1 = np.array([1.,0.,0.])
        or_2 = np.array([0.,1.,0.])
        db1 = dumbbell(0,or_1,np.array([0,1,0]))
        db2 = dumbbell(0,or_1,np.array([1,0,0]))
        j = jump(db1,db2,1,1)
        j_list=[]
        for g in crys.G:
            j_new = j.gop(crys,0,g)
            if not any (j2==j_new for j2 in j_list):
                j_list.append(j_new)
        self.assertEqual(len(j_list),8)
