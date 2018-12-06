import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple


# Single dumbbell state representer class.
# 1. Format - 'i o R c' -> basis index, orientation, lattice vector, active atom indicator
# 2. Should be able to check if two dumbbell states are identical
# 3. Should be able to add a jump to a dumbbell state.
# 4. Should be able to apply a given group operation (crystal specified) to a dumbbell.

class dumbbell(namedtuple('dumbbell','i o R')):

    def costheta(self,other):
        return(np.dot(self.o,other.o)/(la.norm(self.o)*la.norm(other.o)))

    def __eq__(self,other):
        # zero=np.zeros(len(self.o))
        true_class = isinstance(other,self.__class__)
        c1 = true_class and (self.i==other.i and np.allclose(self.o,other.o,atol=1e-6) and np.allclose(self.R,other.R,atol=1e-8))
        return c1
    def __ne__(self,other):
        return not self.__eq__(other)

    def __neg__(self):
        #negation is used to flip the orientation vector
        return self.__class__(self.i,-self.o+0.,self.R)

    def __hash__(self):
        o = np.round(self.o,3)
        return hash((self.i,o[0],o[1]*5,o[2],self.R[0],self.R[1],self.R[2]))

    def gop(self,crys,chem,g):
        r1, (ch,i1) = crys.g_pos(g,self.R,(chem,self.i))
        o1 = crys.g_direc(g,self.o)
        return self.__class__(i1,o1,r1)

    def __add__(self,other):
        if not isinstance(other,np.ndarray):
            raise TypeError("Can only add a lattice translation to a dumbbell")

        if not len(other)==3:
            raise TypeError("Can add only a lattice translation (vector) to a dumbbell")
        for x in other:
            if not isinstance(x, np.dtype(int).type):
                raise TypeError("Can add only a lattice translation vector (integer components) to a dumbbell")

        return self.__class__(self.i,self.o,self.R+other)

    def __sub__(self,other):
        return self.__add__(-other)

# A Pair obect (that represents a dumbbell-solute state) should have the following attributes:
# 1. It should have the locations of the solute and dumbbell.
# 2. A pair dbect contains information regarding which atom in the dumbbell is going to jump.
# 3. We should be able to apply Group operations to it to generate new pairs.
# 4. We should be able to add a Jump dbect to a Pair dbect to create another pair dbect.
# 5. We should be able to test for equality.
# 6. Applying group operations should be able to return the correct results for seperated and mixed dumbbell pairs.

class SdPair(namedtuple('SdPair',"i_s R_s db")):

    def __eq__(self, other):
        true_class = isinstance(other,self.__class__)
        true_solute = self.i_s == other.i_s and np.allclose(self.R_s,other.R_s,atol=1e-8)
        true_db = self.db == other.db
        return (true_class and true_solute and true_db)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __neg__(self):
        #negation is used to flip the orientation vector
        return self.__class__(self.i_s,self.R_s,-self.db)

    def __hash__(self):
        return hash((self.i_s, self.R_s[0],hash(self.db)))
        # return id(self)

    def gop(self,crys,chem,g):#apply group operation
        zero = np.zeros(len(self.db.o))
        R_s_new, (ch,i_s_new) = crys.g_pos(g,self.R_s,(chem,self.i_s))
        dbnew = self.db.gop(crys,chem,g)
        return self.__class__(i_s_new,R_s_new,dbnew)

    def is_zero(self):
        """
        To check if solute and dumbbell are at the same site
        """
        return self.i_s==self.db.i and np.allclose(self.R_s,self.db.R)

    def addjump(self,j,mixed=False):
        if mixed==False and isinstance(j.state1,SdPair):
            raise TypeError("Only dumbbell -> dumbbell transitions can be added to complexes, not pair -> pair")
        elif mixed==True and not isinstance(j.state1,SdPair):
            raise TypeError("Only pair -> pair transitions can be added to mixed dumbbells")

        if mixed==False:
            if not (self.db.i==j.state1.i and np.allclose(self.db.o,j.state1.o)):
                raise ArithmeticError("Incompatible starting dumbbell configurations")
            db2 = dumbbell(j.state2.i,j.state2.o,self.db.R+j.state2.R-j.state1.R)
            return self.__class__(self.i_s,self.R_s,db2)

        if mixed==True:
            if not self.is_zero():
                raise TypeError("Was indicated that the complex is a mixed dumbbell.")
            if not (self.db.i==j.state1.db.i and np.allclose(self.db.o,j.state1.db.o)):
                raise ArithmeticError("Incompatible starting dumbbell configurations")
            #see if solute atom must move
            db2 = dumbbell(j.state2.db.i,j.state2.db.o,j.state2.db.R + self.db.R-j.state1.db.R)
            soluteshift = j.state2.R_s - j.state1.R_s
            return SdPair(j.state2.i_s,soluteshift+self.R_s,db2)

    def __add__(self,other):

        """
        Adding a translation to a solute-dumbbell pair shifts both the solute and the dumbbell
        by the same translation vector.
        """
        if not isinstance(other,np.ndarray):
            raise TypeError("Can only add a lattice translation to a dumbbell")

        if not len(other)==3:
            raise TypeError("Can add only a lattice translation (vector) to a dumbbell")
        for x in other:
            if not isinstance(x, np.dtype(int).type):
                raise TypeError("Can add only a lattice translation vector (integer components) to a dumbbell")

        return self.__class__(self.i_s,self.R_s+other,self.db+other)

    def __sub__(self,other):
        return self.__add__(-other)

# Jump obects are rather simple, contain just initial and final orientations
# Also adding a jump to a dumbbell is now done here.
# dumbell/pair obects are not aware of jump dbects.
class jump(namedtuple('jump','state1 state2 c1 c2')):
    def __init__(self,state1,state2,c1,c2):
        #Do Type checking of input stateects
        if not isinstance(self.state2,self.state1.__class__):
            raise TypeError("Incompatible Initial and final states. They must be of the same type.")
        if isinstance(self.state1,SdPair):
         #If not a mixed dumbbell, solute cannot move
            if not (self.state1.i_s==self.state1.db.i and np.allclose(self.state1.R_s,self.state1.db.R)):
                if not (self.state1.i_s==self.state2.i_s and np.allclose(self.state1.R_s,self.state2.R_s)):
                    raise ArithmeticError("Solute atom cannot jump unless part of mixed dumbell")
            #If a mixed dumbbell, then initial and final states must indicate the position of the same atom.
            # elif not np.allclose(self.state1.db.R,self.state2.db.R):
                #Rotations can involve pure as well as mixed dumbbells.
                #For now, in mixed dumbbell rotations, c1=1->c2=1 will have to be verified in star generation manually.
                # if not self.c1==self.c2:
                #     raise ArithmeticError("The same active atom must transition between state1 and state2 (must have same c values) for a mixed dumbbell")
                # if self.c1==1:
                #     if not (self.state2.i_s==self.state2.db.i and np.allclose(self.state2.R_s,self.state2.db.R) and self.c2==1):
                #         print(self)
                #         raise ArithmeticError("Solute atom jumping from mixed dumbbell must lead to another mixed dumbbell")
                # if self.c1==-1:
                #     if not (self.state2.i_s==self.state1.i_s and np.allclose(self.state1.R_s,self.state2.R_s)):
                #         raise ArithmeticError("Solvent atom jumping from mixed dumbbell means solute must remain fixed.")

    def __eq__(self,other):
        return(self.state1==other.state1 and self.state2==other.state2 and self.c1==other.c1 and self.c2==other.c2)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __hash__(self):
        # return hash((self.state1,self.state2,self.c1,self.c2))
        return hash((hash(self.state1),hash(self.state2),self.c1,self.c2))
        # return id(self)
    def __add__(self,other):
          #Do type checking of input operands and jump states
          if not isinstance(other,dumbbell):
              raise TypeError("For now jumps can only be added to dumbbell objects.")
          if not (self.state1.i==other.i and np.allclose(self.state1.o,other.o)):
              raise ArithmeticError("Operand dumbbell and initial dumbbell of jump must have same configuration.")
          return dumbbell(self.state2.i,self.state2.o,self.state2.R+other.R)
        # if not (isinstance(self.state1,other.__class__) and isinstance(self.state2,other.__class__)):
        #     raise TypeError("Incompatible operand and jump states.")
        # if isinstance(other,dumbbell):
        #   #check that the initial dumbbell state of the jump and the input dumbbell
        #   #have the same configuration
        #     if not (np.allclose(self.state1.o,other.o) and self.state1.i==other.i):
        #         raise ArithmeticError("Operand dumbbell and initial dumbbell of jump must have same configuration (basis site coordinate and orientation")
        #     return (dumbbell(self.state2.i,self.state2.o,other.R+self.state2.R))
        # if isinstance(other,SdPair):
        #   #Check arithmetic compatibility of dumbbells
        #     if not (np.allclose(self.state1.db.o,other.db.o) and self.state1.db.i==other.db.i):
        #         raise ArithmeticError("Operand pair's dumbbell and initial pair's dumbell of jump must have same configuration (basis site coordinate and orientation")
        #       #check that the pair is translated with the same pair configuration.
        #     if not (self.state1.i_s == other.i_s and np.allclose(-self.state1.R_s+self.state1.db.R,-other.R_s+other.db.R)):
        #         raise ArithmeticError("Operand pair and initial pair of jump must have same relative configurations")
        #       #if not a mixed dumbbell, solute stays fixed
        #     if not (other.i_s==other.db.i and np.allclose(other.R_s,other.db.R)):
        #         dbnew = dumbbell(self.state2.db.i,self.state2.db.o,self.state2.db.R+other.R_s-self.state1.R_s)
        #         return SdPair(other.i_s,other.R_s,dbnew)
        #      #if a mixed dumbbell, check that the correct final state has been reached
        #     if self.c1==1:
        #         dbnew = dumbbell(self.state2.db.i,self.state2.db.o,self.state2.db.R+other.R_s-self.state1.R_s)
        #         return SdPair(dbnew.i,dbnew.R,dbnew)
        #     if self.c1==-1:
        #         dbnew = dumbbell(self.state2.db.i,self.state2.db.o,self.state2.db.R+other.R_s-self.state1.R_s)
        #         return SdPair(dbnew.i,dbnew.R,dbnew)
        # if isinstance(other,self.__class__):
        #     #Add two jumps
        #     if not(self.state2==other.state1 and self.c2==other.c1):
        #         raise ArithmeticError("Initial state of second jump must be the same as the final state of the first jump")
        #         return self.__class__(self.state1,other.state2,self.c1,other.c2)

    def __radd__(self,other):
        return self.__add__(other)

    def __neg__(self):
        #negation is used to flip the transition in the opposite direction
        return self.__class__(self.state2,self.state1,self.c2,self.c1)
    def __str__(self):
        if isinstance(self.state1,SdPair):
            strrep = "Jump object:\nInitial state:\n\t"
            strrep += "Solute loctation :basis index = {}, lattice vector = {}\n\t".format(self.state1.i_s,self.state1.R_s)
            strrep += "dumbbell :basis index = {}, lattice vector = {}, orientation = {}\n".format(self.state1.db.i,self.state1.db.R,self.state1.db.o)
            strrep += "Final state:\n\t"
            strrep += "Solute loctation :basis index = {}, lattice vector = {}\n\t".format(self.state2.i_s,self.state2.R_s)
            strrep += "dumbbell :basis index = {}, lattice vector = {}, orientation = {}\n".format(self.state2.db.i,self.state2.db.R,self.state2.db.o)
            strrep += "\nJumping from c = {} to c= {}".format(self.c1,self.c2)
        if isinstance(self.state1,dumbbell):
            strrep = "Jump object:\nInitial state:\n\t"
            strrep += "dumbbell :basis index = {}, lattice vector = {}, orientation = {}\n".format(self.state1.i,self.state1.R,self.state1.o)
            strrep += "Final state:\n\t"
            strrep += "dumbbell :basis index = {}, lattice vector = {}, orientation = {}\n".format(self.state2.i,self.state2.R,self.state2.o)
            strrep += "Jumping from c = {} to c= {}\n".format(self.c1,self.c2)
        return(strrep)

    def gop(self,crys,chem,g): #Find symmetry equivalent jumps - required when making composite jumps.
        state1new=self.state1.gop(crys,chem,g)
        state2new=self.state2.gop(crys,chem,g)
        return self.__class__(state1new,state2new,self.c1,self.c2)
