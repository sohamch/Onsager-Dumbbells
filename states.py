import numpy as np
import onsager.crystal as crystal
from representations import *

class dbStates(object):
    """
    Class to generate all possible dumbbell configurations for given basis sites
    """
    def __init__(self,crys,iorlist):
        self.crys = crys
        # self.orlist = orlist
        self.stategroups = self.group(iorlist)
        #This produces a symmetry-grouped list of (i,or) pairs like sitelist.

    def gdumb(self,chem,g,db):
        #operates a group operation and return parity value
        z = np.zeros(3)
        db_new = db.gop(self.crys,chem,g)
        i_new = db_new.i
        o_new = db_new.o
        if any(j==i_new and np.allclose(o,o_new) for states in self.stategroups for j,o in states):
            return tuple(db_new,1)
        if any(j==i_new and np.allclose(o+o_new,) for states in self.stategroups for j,o in states):
            return tuple(db_new,-1)
