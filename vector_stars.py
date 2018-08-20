import numpy as np
import onsager.crystal as crystal
from jumpnet3 import *
from states import *
from representations import *
from stars import *

class vectorStars(object):
    def __init__(self,crys_star=None):
        self.starset = None
        self.Nvstars = 0
        if starset is not None:
            if starset.Nshells > 0:
                self.generate(crys_star)

    def generate(starset):
        """
        Follows almost the same as that for solute-vacancy case. Only generalized to keep the state
        under consideration unchanged.
        """
        if starset.Nshells == 0: return
        if starset == self.starset: return
        self.starset = starset
        self.vecpos = []
        self.vecvec = []
        #first do it for the complexes
        for s in starset.stars[:starset.mixedstartindex]:
            pair0 = states[s[0]]
            glist=[]
            #Find group operations that leave state unchanged
            for g in starset.crys.G:
                pairnew = pair0.gop(starset.crys,starset.chem,g)
                if pairnew == pair0 or pairnew==-pair0:
                    glist.append(g)
            #Find the intersected vector basis for these group operations
            vb=reduce(CombineVectorBasis,[VectorBasis(*g.eigen()) for g in glist])
            #Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)
            for v in vlist:
                        self.vecpos.append(s.copy())
                        veclist = []
                        for pairI in [pair for pair in s]:
                            for g in starset.crys.G:
                                if pair0.g(starset.crys, starset.chem, g) == pairI or pair0.g(starset.crys, starset.chem, g) == -pairI:
                                    veclist.append(starset.crys.g_direc(g, v))
                                    break
                        self.vecvec.append(veclist)
            self.mixedstartindex = len(vecpos)
            #Now do it for the mixed dumbbells - all negative checks dissappear
            for s in starset.stars[starset.mixedstartindex:]:
                pair0 = states[s[0]]
                glist=[]
                #Find group operations that leave state unchanged
                for g in starset.crys.G:
                    pairnew = pair0.gop(starset.crys,starset.chem,g)
                    if pairnew == pair0:
                        glist.append(g)
                #Find the intersected vector basis for these group operations
                vb=reduce(CombineVectorBasis,[VectorBasis(*g.eigen()) for g in glist])
                #Get orthonormal vectors
                vlist = starset.crys.vectlist(vb)
                for v in vlist:
                            self.vecpos.append(s.copy())
                            veclist = []
                            for pairI in [pair for pair in s]:
                                for g in starset.crys.G:
                                    if pair0.g(starset.crys, starset.chem, g) == pairI:
                                        veclist.append(starset.crys.g_direc(g, v))
                                        break
                            self.vecvec.append(veclist)
