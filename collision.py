import numpy as np
import onsager.crystal as crystal
from representations import *
from test_structs import *

def collsion_self(crys,chem,jump,cutoff12,cutoff13=None):
    """
    Check if the three atoms involved in a dumbbell jumping from one site to the next
    are colliding or not.

    params:
        crys - crystal structure under consideration
        jump - the jump object representing the transitions
        cutoff12 - minimum allowed distance between the two atoms in the initial dumbbell.
        cutoff13 - minimum allowed distance between the two atoms in the final dumbbell.
    Returns:
        True if atoms collide. False otherwise
    """
    if cutoff13==None:
        cutoff13=cutoff12

    def iscolliding(a0i,a1i,a0j,a1j,cutoff):
        """
        Returns True if the given atom pair comes closer than cutoff within t=0 or t=1.
        False otherwise.
        The position of an atom 'i' as a function of time is given by - R(t) = a0i + a1i * t
        Then the minimum squared distance between atoms 'i' and 'j' is minimized as a function of time.
        """

        num = np.dot((a1i-a1j),(a0i-a0j))
        den = np.dot((a1i-a1j),(a1i-a1j))
        tmin = np.round(-num/den,decimals=6)
        # print(tmin)
        mindist2 = np.round(np.dot(((a0i+a1i*tmin)-(a0j+a1j*tmin)),((a0i+a1i*tmin)-(a0j+a1j*tmin))),decimals=4)
        # print ("mindist^2 = ",mindist2)
        # print ("cutoff^2 = ",np.round(cutoff**2,decimals=6))
        # print()
        if tmin <= 0 or tmin >= 1:
            return False #no atoms collide within transition time.
        elif (mindist2>=np.round(cutoff**2,decimals=6)):
            return False #no atoms collide
        else:
            return True #atoms collide
    #create the initial and final locations of the atoms
    if isinstance(jump.state1,dumbbell):
        #Shorten this part with a function later on
        R1i = crys.unit2cart(jump.state1.R,crys.basis[chem][jump.state1.i])+(jump.c1/2.)*jump.state1.o
        R2i = crys.unit2cart(jump.state1.R,crys.basis[chem][jump.state1.i])-(jump.c1/2.)*jump.state1.o
        R3i = crys.unit2cart(jump.state2.R,crys.basis[chem][jump.state2.i])
        R1f = crys.unit2cart(jump.state2.R,crys.basis[chem][jump.state2.i])+(jump.c2/2.)*jump.state2.o
        R2f = crys.unit2cart(jump.state1.R,crys.basis[chem][jump.state1.i])
        R3f = crys.unit2cart(jump.state2.R,crys.basis[chem][jump.state2.i])-(jump.c2/2.)*jump.state2.o
        # print(R1i,R2i,R3i,R1f,R2f,R3f)
    elif isinstance(jump.state1,SdPair):
        R1i = crys.unit2cart(jump.state1.db.R,crys.basis[chem][jump.state1.db.i])+(jump.c1/2.)*jump.state1.db.o
        R2i = crys.unit2cart(jump.state1.db.R,crys.basis[chem][jump.state1.db.i])-(jump.c1/2.)*jump.state1.db.o
        R3i = crys.unit2cart(jump.state2.db.R,crys.basis[chem][jump.state2.db.i])
        R1f = crys.unit2cart(jump.state2.db.R,crys.basis[chem][jump.state2.db.i])+(jump.c2/2.)*jump.state2.db.o
        R2f = crys.unit2cart(jump.state1.db.R,crys.basis[chem][jump.state1.db.i])
        R3f = crys.unit2cart(jump.state2.db.R,crys.basis[chem][jump.state2.db.i])-(jump.c2/2.)*jump.state2.db.o

    if np.allclose(R1i,R1f):
        return False #not considering rotations(yet).
    a01=R1i.copy()
    a11=(R1f-R1i)
    a02=R2i.copy()
    a12=(R2f-R2i)
    a03=R3i.copy()
    a13=(R3f-R3i)
    #check the booleans for each pair
    c12 = iscolliding(a01,a11,a02,a12,cutoff12)
    # print(c12)
    c13 = iscolliding(a01,a11,a03,a13,cutoff13)
    # print(c13)
    # c23 = isnotcolliding(a02,a12,a03,a13,cutoff)
    return (c12 or c13)


def collision_others(crys,chem,jmp,supervect,closestdistance):
    """
    Takes a jump and sees if the moving atom of the dumbbell collides with any other atom.
    params:
        crys,chem - the crystal and the sublattice under consideration.
        jmp - the jump object to test.
        supervect - The lattice vectors upto which the jumps extend.
        closestdistance - (A list or a number) minimum allowable distance to other atoms
    Returns:
        True if atoms collide. False otherwise.
    """
    #Format closestdistance appropriately
    if isinstance(closestdistance,list):
        closest2list = [x**2 for c,x in enumerate(closestdistance)]
    else:
        closest2list = [x**2 for c in range(crys.Nchem)]
    #First extract the necessary parameters for calculating the transport vector
    (i1,i2) = (jmp.state1.i,jmp.state2.i) if isinstance(jmps.state1,dumbbell) else (jmp.state1.db.i,jmp.state2.db.i)
    (R1,R2) = (jmp.state1.R,jmp.state2.R) if isinstance(jmps.state1,dumbbell) else (jmp.state1.db.R,jmp.state2.db.R)
    (o1,o2) = (jmp.state1.o,jmp.state2.o) if isinstance(jmps.state1,dumbbell) else (jmp.state1.db.o,jmp.state2.db.o)
    c1,c2 =jmp.c2,jmp.c2
    #Now construct the transport vector
    dvec = (c2/2.)*o2 - (c1/2.)*o1
    dR = crys.unit2cart(R2-R1,crys.basis[chem][i2]-crys.basis[chem][i1])
    dx = dR+dvec
    #now test against other atoms, treating the initial atom as the origin
    for c,mindist2 in enumerate(closest2list)
