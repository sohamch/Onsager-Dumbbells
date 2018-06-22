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
    def __init__(self,crys,chem,iorlist):
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
        self.iorlist = iorlist
        self.symorlist = self.gensymset(crys,chem,iorlist)
        #Store both iorlist and symorlist so that we can compare them later if needed.
        self.threshold = crys.threshold

    def gdumb(self,g,db):
        """
        Takes as an argument a dumbbell and return the result of a group operation on that dumbbell
        param: g - group operation
               db - dumbbell object to operate upon
        returns - (db_new,p) - the dumbbell produced by the symmetry operation and the parity value depending on existence
                  within symorlist.
                  Example - 1. If the new orientation produced is [-100] but the one present in symorlist is [100], then returns
                    ("db object with orientation [100]", -1)
                            2. If the new orientation is [100] instead, returns ("db object with orientation [100]", 1)
        """
        def inlist(tup):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=self.threshold) for lis in self.symorlist for x in lis)

        db_new = db.gop(self.crys,self.chem,g)
        i = db_new.i
        o = db_new.o
        tup = None
        if inlist((i,o)):
            tup = tuple([db_new,1])
        elif inlist((i,-o)):
            db_new = dumbbell(db_new.i,-db_new.o,db_new.R)
            tup = tuple([db_new,-1])
        if tup == None:
            #This will be used only during the testing phase, can remove it later when not needed.
            #Ideally, if the production of symorlist is correct, then it should catch either of the
            #above two cases.
            raise RuntimeError("The group operation does not produce an (i,or) set in symorlist")
        return tup

    def gensymset(crys,chem,iorlist):

        def inlist(tup):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for x in iorlist)

        #first make a set of the unique orientations supplied - each taken only once
        #That way we won't need to do redundant group operations
        z = np.zeros(3)
        orset=[]
        for ior in iorlist:
            if not any(i==x[0] and (np.allclose(tup[1],x[1],atol=crys.threshold) or np.allclose(tup[1]+x[1],z,atol=crys.threshold)) for x in orset):
                orset.append(tup[1])

        #Now, start with the first pair in orset
        ior = orset[0]
        symlist=[]
        newlist=[]
        newlist.append(tup)
        for g in crys.G:
            ior_new = tuple([g.indexmap[chem][ior[0]],crys.g_direc(g,ior[1])])
