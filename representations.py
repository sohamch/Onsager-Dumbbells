
# coding: utf-8

# import modules

# In[3]:


import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple


# Single dumbbell state representer class.
# 1. Format - 'i o R c' -> basis index, orientation, lattice vector, active atom indicator
# 2. Should be able to check if two dumbbell states are identical
# 3. Should be able to add a jump to a dumbbell state.
# 4. Should be able to apply a given group operation (crystal specified) to a dumbbell.

# In[5]:


class dumbbell(namedtuple('dumbbell','i o R c')):

    def __eq__(self,other):
        true_class = isinstance(other,self.__class__)
        if (self.i==other.i and np.allclose(self.R,other.R) and np.allclose(self.o,other.o) and self.c==other.c):
            return (True and true_class)
    def __ne__(self,other):
        return not self.__eq__(other)

    def __add__(self,other):
        if not isinstance(other,jump):
            raise TypeError("Can only add a jump to a state.")

        if not (self.i==other.db1.i and np.allclose(self.o,other.db1.o)
                and np.allclose(self.R,other.db1.R) and self.c==other.db1.c):
            raise ArithmeticError("Initial state of Jump object must match current dumbbell state")

        return other.db2

    def gop(self,crys,chem,g):
        r1, (ch,i1) = crys.g_pos(g,self.R,(chem,self.i))
        o1 = np.dot(g.cartrot,self.o)
        return self.__class__(i1,o1,r1,self.c)


# A Pair object (that represents a dumbbell-solute state) should have the following attributes:
# 1. It should have the locations of the solute and dumbbell.
# 2. A pair object contains information regarding which atom in the dumbbell is going to jump.
# 3. We should be able to apply Group operations to it to generate new pairs.
# 4. We should be able to add a Jump object to a Pair object to create another pair object.
# 5. We should be able to test for equality.

# In[10]:


class SdPair(namedtuple('SdPair',"i_s R_s db")):

    def __eq__(self, other):
        true_class = isinstance(other,self.__class__)
        true_solute = self.i_s == other.i_s and np.allclose(self.R_s,other.R_s)
        true_db = self.db == other.db
        return (true_class and true_solute and true_db)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __add__(self,other):
        if not isinstance(other,jump):
            raise TypeError("Can only add a jump to a state.")
        if not (self.db.i==other.db1.i and self.db.c==other.db1.c and np.allclose(self.db.R,other.db1.R) and np.allclose(self.db.o,other.db1.o)):
            raise ArithmeticError("Initial state of the jump must match current dumbbell state")

        #check if mixed dumbbell jump
        if(self.i_s==self.db.i and np.allclose(self.R_s,self.db.R) and self.db.c==1):
            return self.__class__(other.db2.i,other.db2.R,other.db2)

        return self.__class__(self.i_s,self.R_s,other.db2)

        #return self.__class__(self.i_s,self.R_s,other.R2,self.dx+other.dx,self.c)


    def gop(self,crys,chem,g):#apply group operation
        R_s_new, (ch,i_s_new) = crys.g_pos(g,self.R_s,(chem,self.i_s))
        dbnew = self.db.gop(crys,chem,g)
        return self.__class__(i_s_new,R_s_new,dbnew)



# Jump objects are rather simple, contain just initial and final orientations, location and pointers towards jumping/active atom (+1 for head of orientation vector, -1 for tail of orientation vector).

# In[12]:


class jump(namedtuple('jump','db1,db2')):

    def __init__(self,db1,db2):#How to go about doing this in a better way?
        self.dx = self.db2.R-self.db1.R

    def __eq__(self,other):
        return(self.db1==other.db1 and self.db2==other.db2)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __add__(self,other):
        if not isinstance(other,self.__class__):
            raise TypeError("Can only add two jumps.")

        if not (self.db2.i==other.db1.i and np.allclose(self.db2.o,other.db1.o) and np.allclose(self.db2.R,other.db1.R)
                and self.db2.c==other.db1.c):
            raise ArithmeticError("Starting point of second jump does not match end point of first")

        return self.__class__(self.db1,other.db2)

    def gop(self,crys,chem,g): #Find symmetry equivalent jumps - required when making composite jumps.
        db1new=self.db1.gop(crys,chem,g)
        db2new=self.db2.gop(crys,chem,g)
        return self.__class__(db1new,db2new)
