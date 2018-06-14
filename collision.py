import numpy as np
import onsager.crystal as crystal
from representations import *
from test_structs import *
def collsion_self(crys,jump,cutoff):
    #1. a0 and a1 for the first atoms
    #create the initial atom positions from the dumbbells
    def isnotcolliding(a0i,a1i,a0j,a1j,cutoff):
        num = np.dot((a1i-a1j),(a0i-a0j))
        den = np.dot((a1i-a1j),(a1i-a1j))
        tmin = -num/den
        if tmin < 0 or tmin > 1:
            return True
        elif np.dot(((a0i+a1i*tmin)-(a0j+a1j*tmin)),((a0i+a1i*tmin)-(a0j+a1j*tmin)))>cutoff*cutoff:
            return True
        else:
            return False
    if isinstance(jump.state1,db):
        R1i = crys.unit2cart(jump.state1.R,jump.state1.u)+(jump.c1/2.)*jump.state1.o
        R2i = crys.unit2cart(jump.state1.R,jump.state1.u)-(jump.c1/2.)*jump.state1.o
        R3i = crys.unit2cart(jump.state2.R,jump.state2.u)
    elif isinstance(jump.state1,SdPair):
        R1i = crys.unit2cart(jump.state1.db.R,jump.state1.db.u)+(jump.c1/2.)*jump.state1.db.o
        R2i = crys.unit2cart(jump.state1.db.R,jump.state1.db.u)-(jump.c1/2.)*jump.state1.db.o
        R3i = crys.unit2cart(jump.state2.db.R,jump.state2.db.u)
    #create the final atom positions from the dumbbells
    if isinstance(jump.state1,db):
        R1f = crys.unit2cart(jump.state2.R,jump.state2.u)+(jump.c2/2.)*jump.state2.o
        R2f = crys.unit2cart(jump.state1.R,jump.state1.u)
        R3f = crys.unit2cart(jump.state2.R,jump.state2.u)-(jump.c2/2.)*jump.state2.o
    elif isinstance(jump.state1,SdPair):
        R1f = crys.unit2cart(jump.state2.db.R,jump.state2.db.u)+(jump.c2/2.)*jump.state2.db.o
        R2f = crys.unit2cart(jump.state1.db.R,jump.state1.db.u)
        R3f = crys.unit2cart(jump.state2.db.R,jump.state2.db.u)-(jump.c2/2.)*jump.state2.db.o
    a01=R1i.copy()
    a11=(R1f-R1i)
    a02=R2i.copy()
    a12=(R2f-R2i)
    a03=R3i.copy()
    a13=(R3f-R3i)
