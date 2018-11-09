import numpy as np
import onsager.crystal as crystal
from representations import *
from shell import *
from gensets import *

class dbStates(object):
    """
    Class to generate all possible dumbbell configurations for given basis sites
    This is mainly to automate group operations on jumps (to return correct dumbbell states)
    Functionalities - 1. take in a list of (i,or) pairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an (i,or) pair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,family):
        if not isinstance(family,list):
            raise TypeError("Enter the families as a list of lists")

        self.crys = crys
        self.chem = chem
        self.family=family
        self.iorlist = self.genpuresets()
        self.symorlist = self.gensymset()
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
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=self.threshold) for x in self.iorlist)

        db_new = db.gop(self.crys,self.chem,g)
        i = db_new.i
        o = db_new.o
        tup = None
        if inlist((i,o)):
            tup = (db_new,1)
        elif inlist((i,-o)):
            # db_new = dumbbell(db_new.i,-db_new.o,db_new.R)
            tup = (-db_new,-1)
        if tup == None:
            #This will be used only during the testing phase, can remove it later when not needed.
            #Ideally, if the production of symorlist is correct, then it should catch either of the
            #above two cases.
            raise RuntimeError("The group operation does not produce an (i,or) pair in the given (i,or) list")
        return tup

    def genpuresets(self):
        """
        generates complete (i,or) set from given family of orientations, neglects negatives, since pure
        """
        if not isinstance(self.family,list):
            raise TypeError("Enter the families as a list of lists")
        for i in self.family:
            if not isinstance(i,list):
                raise TypeError("Enter the families for each site as a list of np arrays")
            for j in i:
                if not isinstance(j,np.ndarray):
                    raise TypeError("Enter individual orientation families as numpy arrays")

        def inlist(tup,lis):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for x in lis)

        def negOrInList(o,lis):
            return any(np.allclose(o+tup[1],0,atol=1e-8) for tup in lis)

        sitelist = self.crys.sitelist(self.chem)
        #Get the Wyckoff sets
        pairlist=[]
        for i,wycksites in enumerate(sitelist):
            orlist = self.family[i]
            site=wycksites[0]
            newlist=[]
            for o in orlist:
                for g in self.crys.G:
                    R, (ch,i_new) = self.crys.g_pos(g,np.zeros(3),(self.chem,site))
                    o_new = self.crys.g_direc(g,o)
                    if not (inlist((i_new,o_new),pairlist) or inlist((i_new,-o_new),pairlist)):
                            if negOrInList(o_new,pairlist):
                                o_new = -o_new
                            pairlist.append((i_new,o_new))
        return pairlist

    def gensymset(self):#crys,chem,iorlist
        """
        Takes in a flat list of (i,or) pairs and groups them according to symmetry
        params:
            crys - the working crystal object
            chem - the sublattice under consideration
            iorlist - flat list of (i,or) pairs
        Returns:
            symorlist - a list of lists which contain symmetry related (i,or) pairs
        """

        #some helper functions
        def matchvec(vec1,vec2):
            return np.allclose(vec1,vec2,atol=self.crys.threshold) or np.allclose(vec1+vec2,0,atol=self.crys.threshold)

        def insymlist(ior,symlist):
            return any(ior[0]==x[0] and matchvec(ior[1],x[1]) for lis in symlist for x in lis)

        def inset(ior,set):
            return any(ior[0]==x[0] and matchvec(ior[1],x[1]) for x in set)

        #first make a set of the unique pairs supplied - each taken only once
        #That way we won't need to do redundant group operations
        orset = []
        for ior in self.iorlist:
            if not inset(ior,orset):
                orset.append(ior)

        #Now check for valid group transitions within orset.
        #If the result of a group operation (j,o1) or it's negative (j,-o1) is not there is orset, append it to orset.
        ior = orset[0]
        newlist=[]
        symlist=[]
        newlist.append(ior)
        for g in self.crys.G:
            R, (ch,inew) = self.crys.g_pos(g,np.array([0,0,0]),(self.chem,ior[0]))
            onew  = np.dot(g.cartrot,ior[1])
            if not inset((inew,onew),newlist):
                newlist.append((inew,onew))
        symlist.append(newlist)
        for ior in orset[1:]:
            if insymlist(ior,symlist):
                continue
            newlist=[]
            newlist.append(ior)
            for g in self.crys.G:
                R, (ch,inew) = self.crys.g_pos(g,np.array([0,0,0]),(self.chem,ior[0]))
                onew  = np.dot(g.cartrot,ior[1])
                if not inset((inew,onew),newlist):
                    newlist.append((inew,onew))
            symlist.append(newlist)
        return symlist

class mStates(object):
    """
    Class to generate all possible mixed dumbbell configurations for given basis sites
    This is mainly to automate group operations on jumps (to return correct dumbbell states)
    Functionalities - 1. take in a list of (i,or) pairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an (i,or) pair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,family):
        if not isinstance(family,list):
            raise TypeError("Enter the families as a list of lists")

        self.crys = crys
        self.chem = chem
        self.family = family
        self.iorlist = self.genmixedsets()
        self.symorlist = self.gensymset()
        #Store both iorlist and symorlist so that we can compare them later if needed.
        self.threshold = crys.threshold

    #Is this function really necessary for dumbbells? This is not a static method so it does occupy space every time mdb is instatiated

    def checkinlist(self,mdb):
        """
        Takes as an argument a dumbbell and return the result of a group operation on that dumbbell
        param: g - group operation
               mdb - pair object to operate upon
        returns - mdb_new -> the result of the group operation
        """
        #Type check to see if a mixed dumbbell is passed
        if not isinstance(mdb,SdPair):
            raise TypeError("Mixed dumbbell must be an SdPair object.")
        if not(mdb.i_s==mdb.db.i and np.allclose(mdb.R_s,mdb.db.R,atol=self.threshold)):
            raise TypeError("Passed in pair is not a mixed dumbbell")

        i = mdb.db.i
        o = mdb.db.o
        #Place a check for test purposes
        return any(i==x[0] and np.allclose(o,x[1],atol=self.threshold) for x in self.iorlist)


    def genmixedsets(self):
        crys,chem,family = self.crys,self.chem,self.family
        if not isinstance(family,list):
            raise TypeError("Enter the families as a list of lists")
        for i in family:
            if not isinstance(i,list):
                raise TypeError("Enter the families for each site as a list of numpy arrays")
            for j in i:
                if not isinstance(j,np.ndarray):
                    raise TypeError("Enter individual orientation families as numpy arrays")

        def inlist(tup,lis):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for x in lis)

        sitelist = crys.sitelist(chem)
        #Get the Wyckoff sets
        pairlist=[]
        for i,wycksites in enumerate(sitelist):
            orlist = family[i]
            site = wycksites[0]
            newlist=[]
            for o in orlist:
                for g in crys.G:
                    R, (ch,i_new) = crys.g_pos(g,np.zeros(3),(chem,site))
                    o_new = crys.g_direc(g,o)
                    if not inlist((i_new,o_new),pairlist):
                        pairlist.append((i_new,o_new))
        return pairlist

    def gensymset(self):
        """
        Takes in a flat list of (i,or) pairs and groups them according to symmetry
        params:
            crys - the working crystal object
            chem - the sublattice under consideration
            iorlist - flat list of (i,or) pairs
        Returns:
            symorlist - a list of lists which contain symmetry related (i,or) pairs
        """
        crys,chem,iorlist = self.crys,self.chem,self.iorlist
        def matchvec(vec1,vec2):
            return np.allclose(vec1,vec2,atol=crys.threshold)

        def insymlist(ior,symlist):
            return any(ior[0]==x[0] and matchvec(ior[1],x[1]) for lis in symlist for x in lis)

        def inset(ior,set):
            return any(ior[0]==x[0] and matchvec(ior[1],x[1]) for x in set)
        #first make a set of the unique pairs supplied - each taken only once
        #That way we won't need to do redundant group operations
        orset = []
        for ior in iorlist:
            if not inset(ior,orset):
                orset.append(ior)

        #Now check for valid group transitions within orset.
        #If the result of a group operation (j,o1) or it's negative (j,-o1) is not there is orset, append it to orset.
        ior = orset[0]
        newlist=[]
        symlist=[]
        newlist.append(ior)
        for g in crys.G:
            R, (ch,inew) = crys.g_pos(g,np.array([0,0,0]),(chem,ior[0]))
            onew  = np.dot(g.cartrot,ior[1])
            if not inset((inew,onew),newlist):
                newlist.append((inew,onew))
        symlist.append(newlist)
        for ior in orset[1:]:
            if insymlist(ior,symlist):
                continue
            newlist=[]
            newlist.append(ior)
            for g in crys.G:
                R, (ch,inew) = crys.g_pos(g,np.array([0,0,0]),(chem,ior[0]))
                onew  = np.dot(g.cartrot,ior[1])
                if not inset((inew,onew),newlist):
                    newlist.append((inew,onew))
            symlist.append(newlist)
        return symlist
