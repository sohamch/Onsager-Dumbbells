import numpy as np
import onsager.crystal as crystal
from representations import *

class dbStates(object):
    """
    Class to generate all possible dumbbell configurations for given basis sites
    Functionalities - 1. take in a list of (i,or) pairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an (i,or) pair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,iorlist):
        #Do some type checking
        if not isinstance(iorlist,list):
            raise TypeError("The entered orientation families must be in a list")
        for i in iorlist:
            if not isinstance(i,np.ndarray):
                raise TypeError("Please provide the orientation families as numpy arrays")
        if not isinstance(crys,crystal.Crystal):
            raise TypeError("Unrecognized crystal object type")
        self.crys = crys
        self.chem = chem
        self.iorlist = iorlist

    def gdumb(self,g,jmp):
        """
        Takes as an argument a jump that occurs in the given sublattice and
        return the valid result of a group operation
        """
        def inlist(tup,lis):
            return any(tup[0]==x[0] and tup[1]==x[1] for x in lis)
        if not isinstance(jmp.state1,dumbbell):
            raise TypeError("This grop operation is only valid for dumbbells, not pairs.")
        jmp_new = jmp.gop(self.crys,self.chem,g)
        parity=[]
        for db in [jmp_new.state1,jmp_new.state2]:
            i = db.i
            o = db.o
            if inlist((i,o),self.iorlist):
                parity.append(1)
            elif neginlist((i,o),self.iorlist):
                parity.append(-1)
        return (jmp_new,tuple(parity))
