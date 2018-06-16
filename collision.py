import numpy as np
import onsager.crystal as crystal
from representations import *
from test_structs import *

def collsion_self(crys,chem,jump,cutoff12,cutoff13):
    """
    Check if the three atoms involved in a dumbbell jumping from one site to the next
    are colliding or not.

    params:
        crys - crystal structure under consideration
        jump - the jump object representing the transitions
        cutoff12 - minimum allowed distance between the two atoms in the initial dumbbell.
        cutoff13 - minimum allowed distance between the two atoms in the final dumbbell.
    Returns:
        True if no atoms collide. False otherwise
    """
    def isnotcolliding(a0i,a1i,a0j,a1j,cutoff):
        num = np.dot((a1i-a1j),(a0i-a0j))
        den = np.dot((a1i-a1j),(a1i-a1j))
        tmin = -num/den
        # print("tmin = ",tmin)
        mindist2 = np.dot(((a0i+a1i*tmin)-(a0j+a1j*tmin)),((a0i+a1i*tmin)-(a0j+a1j*tmin)))
        # print ("mindist^2 = ",mindist2)
        if tmin < 0. or tmin > 1.:
            return True #no atoms collide within transition time.
        elif (mindist2-cutoff**2)>1e-8:
            return True #no atoms collide
        else:
            return False #atoms collide
    #create the initial and final locations of the atoms
    if isinstance(jump.state1,dumbbell):
        R1i = crys.unit2cart(jump.state1.R,crys.basis[chem][jump.state1.i])+(jump.c1/2.)*jump.state1.o
        R2i = crys.unit2cart(jump.state1.R,crys.basis[chem][jump.state1.i])-(jump.c1/2.)*jump.state1.o
        R3i = crys.unit2cart(jump.state2.R,crys.basis[chem][jump.state2.i])
        R1f = crys.unit2cart(jump.state2.R,crys.basis[chem][jump.state2.i])+(jump.c2/2.)*jump.state2.o
        R2f = crys.unit2cart(jump.state1.R,crys.basis[chem][jump.state1.i])
        R3f = crys.unit2cart(jump.state2.R,crys.basis[chem][jump.state2.i])-(jump.c2/2.)*jump.state2.o
    elif isinstance(jump.state1,SdPair):
        R1i = crys.unit2cart(jump.state1.db.R,crys.basis[chem][jump.state1.db.i])+(jump.c1/2.)*jump.state1.db.o
        R2i = crys.unit2cart(jump.state1.db.R,crys.basis[chem][jump.state1.db.i])-(jump.c1/2.)*jump.state1.db.o
        R3i = crys.unit2cart(jump.state2.db.R,crys.basis[chem][jump.state2.db.i])
        R1f = crys.unit2cart(jump.state2.db.R,crys.basis[chem][jump.state2.db.i])+(jump.c2/2.)*jump.state2.db.o
        R2f = crys.unit2cart(jump.state1.db.R,crys.basis[chem][jump.state1.db.i])
        R3f = crys.unit2cart(jump.state2.db.R,crys.basis[chem][jump.state2.db.i])-(jump.c2/2.)*jump.state2.db.o

    if np.allclose(R1i,R1f):
        return True #not considering rotations(yet).
    a01=R1i.copy()
    a11=(R1f-R1i)
    a02=R2i.copy()
    a12=(R2f-R2i)
    a03=R3i.copy()
    a13=(R3f-R3i)
    #check the booleans for each pair
    c12 = isnotcolliding(a01,a11,a02,a12,cutoff12)
    c13 = isnotcolliding(a01,a11,a03,a13,cutoff13)
    # c23 = isnotcolliding(a02,a12,a03,a13,cutoff)
    return not(c12 and c13)