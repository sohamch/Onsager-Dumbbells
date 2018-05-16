
# coding: utf-8

# In[1]:


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

# In[4]:


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
        db1 = dumbbell(0,or_1,np.array([0.,0.,0.]),1)#create a dumbbell state
        db2 = dumbbell(0,or_2,np.array([1.,0.,0.]),1)#db2=db3 != db1 deliberately
        db3 = dumbbell(0,or_2,np.array([1.,0.,0.]),1)
        self.assertEqual(db3,db2) #test equality and inequality operators
        self.assertNotEqual(db1,db3)

    #test addition with a jump object
    def test_jump(self):
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        db1 = dumbbell(0,or_1,np.array([0.,0.,0.]),1)
        or_2 = np.array([0.,1.,1.])/la.norm(np.array([0.,1.,1.]))
        j1 = jump(0,0,or_1,or_2,np.array([0.,0.,0.]),np.array([1.,0.,0.]),1,1)
        db_f = dumbbell(0,or_2,np.array([1.,0.,0.]),1)
        with self.assertRaises(ArithmeticError):
            db_f+j1
        self.assertEqual(db_f,db1+j1)

    #Test Application of group operations
    def test_gop(self):
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        db1 = dumbbell(0,or_1,np.array([1.,0.,0.]),1)
        db_list=[]
        for g in crys.G:
            db2 = db1.gop(crys,0,g)
            if not any(db2==db for db in db_list):
                db_list.append(db2)
        self.assertEqual(len(db_list),8)
        #should have 4 equivalent positions, and 2 states each for +- vectors.


# Test For solute-dumbbell pairs
# 1. Equality testing - Test whether the \__eq__ function performs as expected.
# 2. Adding a jump - Test whether \__add__ performs as expected
# 3. Test whether group operations generate the correct number of symmetrically equivalent pairs

# In[6]:


class SdPair_Tests(unittest.TestCase):
    """
    Tests related to a single dumbell diffusing in a lattice.
    Test case - tetragonal lattice.
    Format of a pair object - "i_s R_s db" -> solute location (state), dumbbell state
    """
    #test equality comparison
    def test_equality(self):
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))#orientation 1
        or_2 = np.array([0.,1.,1.])/la.norm(np.array([0.,1.,1.]))#orientation 2
        or_x = np.array([1.,0.,0.])/la.norm(np.array([1.,0.,0.]))#orientation along x

        db1 = dumbbell(0,or_1,np.array([0.,0.,0.]),1)#create a dumbbell state
        db1n = dumbbell(0,or_1,np.array([0.,0.,0.]),-1)#create a dumbbell state
        db2 = dumbbell(0,or_2,np.array([1.,0.,0.]),1)

        pair1 = SdPair(0,np.array([1.,0.,0.]),db1)
        pair2 = SdPair(0,np.array([1.,0.,0.]),db2)
        pair3 = SdPair(0,np.array([1.,0.,0.]),db2)
        self.assertEqual(pair3,pair2) #test equality and inequality
        self.assertNotEqual(pair1,pair3)

    #test addition with a jump object
    def test_jump(self):
        """
        Should contain three tests:
            1a. Put solute and dumbbell in same location - execute solute atom jump
            1b. Put solute and dumbbell in same location - execute solvent atom jump.
            2. Put solute and dumbbell in different locations - execute dumbbell jumps.
        """
        #1a. - solute jumps, c=+1
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        or_2 = np.array([0.,1.,1.])/la.norm(np.array([0.,1.,1.]))

        db1 = dumbbell(0,or_1,np.array([0.,0.,0.]),1)

        pair1 = SdPair(0,np.array([0.,0.,0.]),db1)

        j1 = jump(0,0,or_1,or_2,np.array([0.,0.,0.]),np.array([1.,0.,0.]),1,1)

        db2 = dumbbell(0,or_2,np.array([1.,0.,0.]),1)

        pair2w = SdPair(0,np.array([0.,0.,0.]),db2)#solute doesn't move - wrong
        pair2r = SdPair(0,np.array([1.,0.,0.]),db2)#solute moves - right

        with self.assertRaises(ArithmeticError):
            pair2r+j1

        self.assertEqual(pair1+j1,pair2r)
        self.assertNotEqual(pair1+j1,pair2w)

            #1b - solute stays fixed, c=-1
        db1n = dumbbell(0,or_1,np.array([0.,0.,0.]),-1)
        pair1 = SdPair(0,np.array([0.,0.,0.]),db1n) #mixed dumbbell with solvent as active atom.
        j1 = jump(0,0,or_1,or_2,np.array([0.,0.,0.]),np.array([1.,0.,0.]),-1,1)
        db2 = dumbbell(0,or_2,np.array([1.,0.,0.]),1)
        pair2r = SdPair(0,np.array([0.,0.,0.]),db2)#solute doesn't move - right
        pair2w = SdPair(0,np.array([1.,0.,0.]),db2)#solute moves - wrong
        self.assertEqual(pair1+j1,pair2r)
        self.assertNotEqual(pair1+j1,pair2w)

        #2. isolated dumbbell jump - jumps onto solute site
        pair1 = SdPair(0,np.array([1.,0.,0.]),db1)
        j1 = jump(0,0,or_1,or_2,np.array([0.,0.,0.]),np.array([1.,0.,0.]),1,1)
        db2 = dumbbell(0,or_2,np.array([1.,0.,0.]),1)#dumbbell at the solute site
        pair2 = SdPair(0,np.array([1.,0.,0.]),db2)
        self.assertEqual(pair1+j1,pair2)


    #Test Application of group operations
    def test_gop(self):
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        db1 = dumbbell(0,or_1,np.array([0.,0.,0.]),1)
        pair1 = SdPair(0,np.array([1.,0.,0.]),db1)

        or_x = np.array([1.,0.,0.])/la.norm(np.array([1.,0.,0.]))#orientation along x
        db2 = dumbbell(0,or_x,np.array([0.,0.,1.]),1)
        pair2 = SdPair(0,np.array([0.,0.,0.]),db2)

        pair_list1=[]
        pair_list2=[]
        for g in crys.G:
            pairn = pair1.gop(crys,0,g)
            if not any(pair==pairn for pair in pair_list1):
                pair_list1.append(pairn)

        for g in crys.G:
            pairn = pair2.gop(crys,0,g)
            if not any(pair==pairn for pair in pair_list2):
                pair_list2.append(pairn)
        self.assertEqual(len(pair_list1),8)
        self.assertEqual(len(pair_list2),8) #4 fold symmetry about z, 2fold about x and y.
        #should have 2 equivalent positions, and 4 states each.


# For jump objects, need to check the following two for now:
# 1. Addition of jump objects to produce third jump object
# 2. Applying group operations to produce symmetry equivalent jump objects.

# In[7]:


class jump_Tests(unittest.TestCase):

    def test_dx(self):
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        or_2 = np.array([0.,1.,1.])/la.norm(np.array([0.,1.,1.]))
        j = jump(0,0,or_1,or_2,np.array([0.,1.,0.]),np.array([0.,0.,0.]),1,1)
        self.assertTrue(np.allclose(j.dx,j.R2-j.R1)
                       )
    def test_add(self):
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        or_2 = np.array([0.,1.,1.])/la.norm(np.array([0.,1.,1.]))
        or_3 = np.array([1.,0.,1.])/la.norm(np.array([1.,0.,1.]))
        j1 = jump(0,0,or_1,or_2,np.array([0.,1.,0.]),np.array([0.,0.,0.]),1,1)
        j2 = jump(0,0,or_2,or_3,np.array([0.,0.,0.]),np.array([0.,0.,1.]),1,1)
        j3 = jump(0,0,or_1,or_3,np.array([0.,1.,0.]),np.array([0.,0.,1.]),1,1)
        with self.assertRaises(ArithmeticError):
            j3+j2
        self.assertEqual(j3,j1+j2)

    def test_gop(self):
        crys = crystal.Crystal(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
        or_1 = np.array([0.,0.,1.])/la.norm(np.array([0.,0.,1.]))
        j = jump(0,0,or_1,or_1,np.array([0.,1.,0.]),np.array([0.,0.,1.]),1,1)
        symm_jump_list=[]
        for g in crys.G:
            j1 = j.gop(crys,0,g)
            if not any(j1==j2 for j2 in symm_jump_list):
                symm_jump_list.append(j1)
        self.assertEqual(len(symm_jump_list),8)
