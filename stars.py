import numpy as np
import onsager.crystal as crystal
from jumpnet3 import *
from representations import *

class StarSet(object):
    """
    class to form the crystal stars, with shells indicated by the number of jumps.
    Almost exactly similar to CrystalStars.StarSet except now includes orientations.
    The minimum shell (Nshells=0) is composed of dumbbells situated atleast one jump away.
    """
    def __init__(self,crys,chem,jumpnetwork,Nshells,originstates=False):
        """
        Parameters:
        crys and chem - Respectively, the crystal and sublattice we are working on
        jumpnetwork - pure dumbbell jumpnetwork with which the star set is to be made.
        Nshells - the number of shells to be constructed. minimum is zero.
        """
        self.crys = crys
        self.chem = chem
        self.jumplist = [j for l in jumpnetwork for j in l]
        self.jumpindices = []
        count=0
        for l in jumpnetwork:
            self.jumpindices.append([])
            for j in l:
                self.jumpindices[-1].append(count)
                count+=1
        self.generate(Nshells,originstates)

    def generate(self,Nshells,originstates):

        if Nshells<1:
            Nshells = 0
        startshell=set([])
        stateset=set([])
        #build the starting shell
        for j in self.jumplist:
            pair = SdPair(j.state1.i,j.state1.R,j.state2)
            if not originstates and pair.is_zero():
                continue
            startshell.add(pair)
            stateset.add(pair)
        lastshell=startshell
        nextshell=set([])
        #Now build the next shells:
        for step in range(Nshells):
            for j in self.jumplist:
                for pair in lastshell:
                    try:
                        pairnew = pair.addjump(j)
                    except:
                        continue
                    if not originstates:
                        if pairnew.is_zero():
                            continue
                    nextshell.add(pairnew)
                    stateset.add(pairnew)
            lastshell = nextshell
            nextshell=set([])
        self.stateset = stateset
