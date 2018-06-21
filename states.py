import numpy as np
import onsager.crystal as crystal
from representations import *

class dbStates(object):
    """
    Class to generate all possible dumbbell configurations for given basis sites
    This is mainly to automate group operations on jumps (to return correct dumbbell states)
    Functionalities - 1. take in a list of (i,or) pairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an (i,or) pair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,famlist):
        if not isinstance(famlist,list):
            raise TypeError("Enter the families as a list of lists")
        for i in famlist:
            if not isinstance(i,list):
                raise TypeError("Enter the families for each site as a list of np arrays")
            for j in i:
                if not isinstance(j,np.ndarray):
                    raise TypeError("Enter individual orientation families as numpy arrays")
        self.crys = crys
        self.chem = chem
        self.famlist = famlist
        self.iorlist = self.genpuresets()

    def gdumb(self,g,jmp):
        """
        Takes as an argument a jump that occurs in the given sublattice and
        return the valid result of a group operation
        """
        def inlist(tup):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for lis in self.iorlist for x in lis)

        if not isinstance(jmp.state1,dumbbell):
            raise TypeError("This group operation is only valid for dumbbells, not pairs.")
        jmp_new = jmp.gop(self.crys,self.chem,g)
        parity=[]
        for db in [jmp_new.state1,jmp_new.state2]:
            i = db.i
            o = db.o
            if inlist((i,o)):
                parity.append(1)
            elif inlist((i,-o)):
                parity.append(-1)
        return (jmp_new,tuple(parity))

    def genpuresets(self):
        crys,chem,family = self.crys,self.chem,self.famlist
        def inlist(tup,lis):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for x in lis) #for x in l)

        def negOrInList(o,lis):
            z=np.zeros(3)
            return any(np.allclose(o+tup[1],z,atol=1e-8) for tup in lis)
        sitelist = crys.sitelist(chem)
        #Get the Wyckoff sets
        pairlist=[]
        for i,wycksites in enumerate(sitelist):
            orlist = family[i]
            symmlist=[]
            site=wycksites[0]
            newlist=[]
            for o in orlist:
                for g in crys.G:
                    R, (ch,i_new) = crys.g_pos(g,np.zeros(3),(chem,site))
                    o_new = crys.g_direc(g,o)
                    if not (inlist((i_new,o_new),newlist) or inlist((i_new,-o_new),newlist)):
                            if negOrInList(o_new,newlist):
                                o_new = -o_new
                            newlist.append((i_new,o_new))
            pairlist.append(newlist)
        return pairlist
