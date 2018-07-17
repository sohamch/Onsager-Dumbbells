import numpy as np
import onsager.crystal as crystal
from jumpnet3 import *
from representations import *

class Pairstates(object):
    """
    Class to generate all possible solute-dumbbell pair configurations for given basis sites

    Functionalities - 1. take in a list of SdPairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an SdPair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,iorlist,thrange):

        self.crys = crys
        self.chem = chem
        self.iorlist = iorlist
        self.sympairlist = self.__class__.gensympairs(crys,chem,iorlist,thrange)

    def gpair(self,g,pair):
        """
        Takes as an argument a pair and returns the result of a group operation on that dumbbell
        param: g - group operation
               pair - pair object to operate upon
        returns - (pair_new,parity) -> the result of the group operation and the parity value associated.
        """
        def inlist(pair):
            return any(pair==pair1 or -pair==pair1 for lis in self.sympairlist for pair1 in lis)

        def inorset(pair_new):
            tup = (pair_new.db.i,pair_new.db.o)
            return any(tup[0]==t[0] and np.allclose(tup[1],t[1],atol=self.crys.threshold) for t in self.iorlist)

        pair_new = pair.gop(self.crys,self.chem,g)

        if inorset(pair_new):
            return tuple([pair_new,1])
        if inorset(-pair_new):
            return tuple([-pair_new,-1])
        if not inlist(pair_new):
            return None

    def gensympairs(crys,chem,iorlist,thshell):
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
            "returns a dumbbell that is within the iorlist by negating a vector if it has been reversed."

            if any(db.i==j and np.allclose(db.o,o1,atol=crys.threshold) for j,o1 in iorlist):
                return db
            if any(db.i==j and np.allclose(db.o+o1,0,atol=crys.threshold) for j,o1 in iorlist):
                return -db

        def inset(pair,lis):
            return any(pair==pair1 for x in lis for pair1 in x)

        def inlist(pair,lis):
            return any(pair==pair1 for pair1 in lis)

        orlist=[]
        #group allowed orientations according to sites
        for c in range(len(crys.basis[chem])):
            orlis=[]
            for tup in iorlist:
                if (tup[0]==c):
                    if not any(np.allclose(tup[1],o1)for o1 in orlis):
                        orlis.append(tup[1])
            orlist.append(orlis)

        z=np.zeros(3).astype(int)
        sympairlist=[]

        for i_s in range(len(crys.basis[chem])):
            shell = buildshell(crys,chem,i_s,thshell)
            for tup in shell:
                for o in orlist[tup[1]]:
                    db = dumbbell(tup[1],o,tup[0])
                    pair = SdPair(i_s,z,db)
                    if inset(pair,sympairlist):
                        continue
                    newlist=[]
                    newlist.append(pair)
                    for g in crys.G:
                        newpair = pair.gop(crys,chem,g)
                        db = withinlist(newpair.db)
                        newpair=SdPair(newpair.i_s,newpair.R_s,db)
                        if not inlist(newpair,newlist):
                            newlist.append(newpair)
                    sympairlist.append(newlist)
        return sympairlist

class StarSet(object):
    """
    class to form the crystal stars, with shells indicated by the number of jumps.
    Almost exactly similar to CrystalStars.StarSet except now includes orientations.
    The minimum shell (Nshells=0) is composed of dumbbells situated atleast one jump away.
    """
    def __init__(crys,chem,jumpnetwork,Nshells):
        """
        Parameters:
        crys and chem - Respectively, the crystal and sublattice we are working on
        jumpnetwork - pure dumbbell jumpnetwork with which the star set is to be made.
        Nshells - the number of shells to be constructed. minimum is zero.
        """
        self.crys = crys
        self.chem = chem
        self.jumplist = [j for j in l for l in jumpnetwork]
        self.jumpindices = []
        count=0
        for l in jumpnetwork:
            self.jumpindices.append([])
            for j in l:
                self.jumpindices[-1].append(count)
                count+=1
        self.generate(Nshells,originstates)

    def generate(self,Nshells,originstates=False):
        if Nshells<1:
            Nshells = 0
        lastshell=[]
        for l in self.jumplist:
            for j in jumplist:
