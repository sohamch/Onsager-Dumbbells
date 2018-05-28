import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple


# Single dumbbell state representer class.
# 1. Format - 'i o R c' -> basis index, orientation, lattice vector, active atom indicator
# 2. Should be able to check if two dumbbell states are identical
# 3. Should be able to add a jump to a dumbbell state.
# 4. Should be able to apply a given group operation (crystal specified) to a dumbbell.

class dumbbell(namedtuple('dumbbell','i o R c')):

    def costheta(self,other):
        return(np.dot(self.o,other.o)/(la.norm(self.o)*la.norm(other.o)))

    def __eq__(self,other):
        zero=np.zeros(len(self.o))
        true_class = isinstance(other,self.__class__)
        return true_class and (self.i==other.i and np.allclose(self.o,other.o) and np.allclose(self.R,other.R) and self.c==other.c)
    def __ne__(self,other):
        return not self.__eq__(other)

    def gop(self,crys,chem,g):
        zero=np.zeros(len(self.o))
        r1, (ch,i1) = crys.g_pos(g,self.R,(chem,self.i))
        o1 = np.dot(g.cartrot,self.o)
        if np.allclose(self.o + o1, zero,atol=1e-8):#zero is not exactly representable. Add tolerance for safety.
            return self.__class__(i1,self.o,r1,self.c*(-1))
        return self.__class__(i1,o1,r1,self.c)


# A Pair dbect (that represents a dumbbell-solute state) should have the following attributes:
# 1. It should have the locations of the solute and dumbbell.
# 2. A pair dbect contains information regarding which atom in the dumbbell is going to jump.
# 3. We should be able to apply Group operations to it to generate new pairs.
# 4. We should be able to add a Jump dbect to a Pair dbect to create another pair dbect.
# 5. We should be able to test for equality.
# 6. Applying group operations should be able to return the correct results for seperated and mixed dumbbell pairs.

class SdPair(namedtuple('SdPair',"i_s R_s db")):

    def __eq__(self, other):
        true_class = isinstance(other,self.__class__)
        true_solute = self.i_s == other.i_s and np.allclose(self.R_s,other.R_s)
        true_db = self.db == other.db
        return (true_class and true_solute and true_db)

    def __ne__(self,other):
        return not self.__eq__(other)

    def gop(self,crys,chem,g):#apply group operation
        zero = np.zeros(len(self.db.o))
        R_s_new, (ch,i_s_new) = crys.g_pos(g,self.R_s,(chem,self.i_s))
        dbnew = self.db.gop(crys,chem,g)
        if (np.allclose(self.R_s,self.db.R) and self.i_s==self.db.i):#if mixed dumbbell
             if np.allclose(self.db.o,dbnew.o) and self.db.c*dbnew.c+1.<1e-8:#meaning dumbbell has been rotated by 180 degrees
                 dbnew2 = dumbbell(dbnew.i,-1*dbnew.o,dbnew.R,self.db.c)
                 return self.__class__(i_s_new,R_s_new,dbnew2)
        return self.__class__(i_s_new,R_s_new,dbnew)

# Jump dbects are rather simple, contain just initial and final orientations
# Also adding a jump to a dumbbell is now done here.
# dumbell dbects are not aware of jump dbects.
class jump(namedtuple('jump','crys,state1 state2')):
    #Crystal object is required to calculate the displacement dx
    def __init__(self,crys,state1,state2):
        #Do Type checking of input stateects
        if not isinstance(self.state2,self.state1.__class__):
            raise TypeError("Incompatible Initial and final states. They must be of the same type.")
        if isinstance(self.state1,dumbbell):
            dR = self.state2.R - self.state1.R
            du = self.state2.i - self.state1.i
            self.dx = self.crys.unit2cart(dR,du)
        if isinstance(self.state1,SdPair):
           # First, Check for invalid jumps involving mixed dumbbell SdPair objects
           #Check that we have a mixed dumbbell as initial state:
           if (self.state1.i_s==self.state1.db.i and np.allclose(self.state1.R_s,self.state1.db.R)):

               #check that if solute atom jumps from a mixed dumbbell, it leads to another mixed dumbbell
               if(self.state1.db.c==1):#solute is the active atom
                   if not(self.state2.i_s==self.state2.db.i and np.allclose(self.state2.R_s,self.state2.db.R)):
                       raise ArithmeticError("Invalid Transition - solute atom jumping from mixed dumbbell must lead to another mixed dumbbell")

               #Check that if solvent atom jumps, it does not lead to another mixed dumbbell.
               if(self.state1.db.c==-1):#solvent is the active atom
                   if (self.state2.i_s==self.state2.db.i and np.allclose(self.state2.R_s,self.state2.db.R)):
                       raise ArithmeticError("Invalid Transition - solvent atom jumping from mixed dumbbell cannot lead to another mixed dumbbell")

              #Check that the active atom is not changed when everything else between the initial and final states are the same.
              #This is an edge case, might have to find a better way to deal with this
               if (self.state1.i_s==self.state2.i_s and np.allclose(self.state1.R_s,self.state2.R_s) and np.allclose(self.state1.db.o,self.state2.db.o)):
                   if(self.state1.db.c != self.state2.db.c):
                       raise ArithmeticError("Invalid transition - Rotation in fixed site, but active atom changes - unphysical.")
               #Now calculate dx based on dumbbell displacement:
               dR = self.state2.db.R - self.state1.db.R
               du = self.state2.db.i - self.state1.db.i
               self.dx = self.crys.unit2cart(dR,du)
               
    def __eq__(self,other):
        return(self.state1==other.state1 and self.state2==other.state2)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __add__(self,other):
        if not (isinstance(other,self.__class__) or isinstance(other,dumbbell) or isinstance(other,SdPair)):
            raise TypeError("Can add a jump only to a dumbbell or a pair or another jump")
        #Add a jump to another jump.
        if isinstance(other,jump):
            if not (self.state2==other.state1):
                raise ArithmeticError("Final state of first jump operand must equal the initial state of the second jump operand.")
            return self.__class__(self.state1,other.state2)

        #Add a jump to a dumbbell.
        if isinstance(other,dumbbell):
            if not(self.state1==other):
                raise ArithmeticError("Initial state of the jump operand must be the same as the dumbbell operand in the sum.")
            return self.state2
        if isinstance(other,SdPair):
            if not(self.state1==other):
                raise ArithmeticError("Initial state of the jump operand must be the same as the dumbbell of the pair operand in the sum.")
            return self.state2

    def __radd__(self,other):
        return self.__add__(other)

    def gop(self,crys,chem,g): #Find symmetry equivalent jumps - required when making composite jumps.
        state1new=self.state1.gop(crys,chem,g)
        state2new=self.state2.gop(crys,chem,g)
        return self.__class__(state1new,state2new)
