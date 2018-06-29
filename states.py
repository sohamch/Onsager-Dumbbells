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
        if not isinstance(iorlist,list):
            raise TypeError("Enter the families as a list of lists")

        self.crys = crys
        self.chem = chem
        self.iorlist = iorlist
        self.symorlist = self.__class__.gensymset(crys,chem,iorlist)
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
        """
        Takes in a flat list of (i,or) pairs and groups them according to symmetry
        params:
            crys - the working crystal object
            chem - the sublattice under consideration
            iorlist - flat list of (i,or) pairs
        Returns:
            symorlist - a list of lists which contain symmetry related (i,or) pairs
        """
        def matchvec(vec1,vec2):
            return np.allclose(vec1,vec2,atol=crys.threshold) or np.allclose(vec1+vec2,0,atol=crys.threshold)

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

class mStates(object):
    """
    Class to generate all possible mixed dumbbell configurations for given basis sites
    This is mainly to automate group operations on jumps (to return correct dumbbell states)
    Functionalities - 1. take in a list of (i,or) pairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an (i,or) pair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,iorlist):
        if not isinstance(iorlist,list):
            raise TypeError("Enter the families as a list of lists")

        self.crys = crys
        self.chem = chem
        self.iorlist = iorlist
        self.symorlist = self.__class__.gensymset(crys,chem,iorlist)
        #Store both iorlist and symorlist so that we can compare them later if needed.
        self.threshold = crys.threshold

    def gdumb(self,g,mdb):
        """
        Takes as an argument a dumbbell and return the result of a group operation on that dumbbell
        param: g - group operation
               mdb - dumbbell object to operate upon
        returns - mdb_new -> the result of the group operation
        """
        #Type check to see if a mixed dumbbell is passed
        if not isinstance(mdb,SdPair):
            raise TypeError("Mixed dumbbell must be an SdPair object.")
        if not(mdb.i_s==mdb.db.i and np.allclose(mdb.R_s,mdb.db.R,atol=self.threshold)):
            raise TypeError("Passed in pair is not a mixed dumbbell")
        def inlist(tup):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=self.threshold) for lis in self.symorlist for x in lis)

        mdb_new = mdb.gop(self.crys,self.chem,g)
        i = mdb_new.db.i
        o = mdb_new.db.o
        #Place a check for test purposes
        if not(mdb_new.i_s == mdb_new.db.i and np.allclose(mdb_new.R_s,mdb_new.db.R,atol=self.threshold)):
            return("Group operation does not return a mixed dumbbell.")
        if not inlist((i,o)):
            mdb_new=None
        return mdb_new

    def gensymset(crys,chem,iorlist):
        """
        Takes in a flat list of (i,or) pairs and groups them according to symmetry
        params:
            crys - the working crystal object
            chem - the sublattice under consideration
            iorlist - flat list of (i,or) pairs
        Returns:
            symorlist - a list of lists which contain symmetry related (i,or) pairs
        """
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

class Pairstates(object):
    """
    Class to generate all possible solute-dumbbell pair configurations for given basis sites

    Functionalities - 1. take in a list of SdPairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an SdPair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,pairlist):
        if not isinstance(pairlist,list):
            raise TypeError("Enter the solute-dumbbell pairs as a list")
        for i in pairlist:
            if not isinstance(i,SdPair):
                raise TypeError("The list must contain solute-dumbbell pair objects")

        self.crys = crys
        self.chem = chem
        self.pairlist = pairlist
        self.sympairlist = self.__class__.gensympairs(crys,chem,self.pairlist)

    def gensympairs(crys,chem,pairlist):
        """
        Takes in a flat list of SdPair objects and groups them according to symmetry
        params:
            crys - the working crystal object
            chem - the sublattice under consideration
            pairlist - flat list of (i,or) SdPair objects.
        Returns:
            symorlist - a list of lists which contain symmetry related (i,or) pairs
        """
        def withinlist(db):
            """
            checks if a dumbbell orientation is within the input pairlist.
            If it is reversed with respected to one that is there, return the correct one.
            """
            if any(np.allclose(-db.o,pair.db.o,atol=crys.threshold) for pair in pairlist):
                return pair.db
            if any(db==pair.db for pair in pairlist):
                return db

        def inset(pair,lis):
            return any(pair==pair1 for x in lis for pair1 in x)

        def inlist(pair,lis):
            return any(pair==pair1 for pair1 in lis)

        symlist=[]

        for pair in pairlist:
            if inset(pair,symlist):
                continue
            newlist=[]
            for g in crys.G:
                pair_new = pair.gop(crys,chem,g)
                # db = withinlist(pair_new.db)
                pair_new=SdPair(pair_new.i_s,pair_new.R_s,withinlist(pair_new.db))
                if not inlist(pair_new,newlist):
                    newlist.append(pair_new)
            symlist.append(newlist)
        return symlist
