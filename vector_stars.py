import numpy as np
import onsager.crystal as crystal
from jumpnet3 import *
from states import *
from representations import *
from stars import *
from functools import reduce
import itertools
class vectorStars(object):
    def __init__(self,crys_star=None):
        self.starset = None
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
            vb=reduce(starset.crys.CombineVectorBasis,[starset.crys.VectorBasis(*g.eigen()) for g in glist])
            #Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)
            Nvect=len(vlist)
            if Nvect > 0:
                for v in vlist:
                            self.vecpos.append(s.copy())
                            veclist = []
                            for pairI in [pair for pair in s]:
                                for g in starset.crys.G:
                                    if pair0.g(starset.crys, starset.chem, g) == pairI or pair0.g(starset.crys, starset.chem, g) == -pairI:
                                        veclist.append(starset.crys.g_direc(g, v))
                                        break
                            self.vecvec.append(veclist)
            self.Nvstars_pure = len(vecpos)
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
                Nvect = len(vlist)
                if Nvect > 0:
                    for v in vlist:
                                self.vecpos.append(s.copy())
                                veclist = []
                                for pairI in [pair for pair in s]:
                                    for g in starset.crys.G:
                                        if pair0.g(starset.crys, starset.chem, g) == pairI:
                                            veclist.append(starset.crys.g_direc(g, v))
                                            break
                                self.vecvec.append(veclist)
            self.Nvstars = len(vecpos)

    def biasexpansion(self,jumpnetwork_omega1,jumptype,jumpnetwork_omega4,jumpnetwork_omega3):
        """
        Returns an expansion of the bias vector in terms of the displacements produced by jumps.
        Parameters:
            jumpnetwork_omega* - the jumpnetwork for the "*" kind of jumps (1,2,3 or 4)
            jumptype - the omega_0 jump type that gives rise to a omega_1 jump type (see jumpnetwork_omega1 function
            in stars.py module)
        Returns:
            bias0, bias1, bias2, bias4 and bias3 expansions
        """
        def disp(jump):
            if isinstance(jump.state1,dumbbell):
                dx = self.starset.crys.unit2cart(jump.state2.R,jump.state2.i) - self.starset.crys.unit2cart(jump.state1.R,jump.state1.i)
            else:
                dx = self.starset.crys.unit2cart(jump.state2.db.R,jump.state2.db.i) - self.starset.crys.unit2cart(jump.state1.db.R,jump.state1.db.i)
        #First, what would be the shape of bias0expansion
        bias0expansion = np.zeros((self.Nvstars_pure,len(self.starset.jumpindices)))
        bias1expansion = np.zeros((self.Nvstars_pure,len(jumpnetwork_omega1)))

        bias4expansion = np.zeros((self.Nvstars_pure,len(jumpnetwork_omega4)))
        #Expansion of pure dumbbell state bias vectors and comple state bias vectors

        bias2expansion = np.zeros((self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega2)))
        bias3expansion = np.zeros((self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega3)))
        #Expansion of mixed dumbbell state bias vectors.

        for i, states, vectors in zip(itertools.count(),self.vecpos[:Nvstars_pure],self.vecvec[:Nvstars_pure]):
            #First construct bias1expansion and bias0expansion
            #This contains the expansion of omega_0 jumps and omega_1 type jumps
            #omega_0 : pure -> pure
            #omega_1 : complex -> complex
            for k,jumplist,jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
                for j in jumplist:
                    IS=j.state1
                    dx = disp(j)
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if states[0]==IS:
                        geom_bias = np.dot(vectors[0], dx) #I haven't normalized with respect to no. of states.
                        bias1expansion[i, k] += geom_bias
                        bias0expansion[i, jt] += geom_bias
            #Next, omega_4: complex -> mixed
            #The correction delta_bias = bias4 + bias1 - bias0
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega4):
                for j in jumplist:
                    IS=j.state1
                    dx = disp(j)
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if states[0]==IS:
                        geom_bias = np.dot(vectors[0], dx) #I haven't normalized with respect to no. of states.
                        bias4expansion[i, k] += geom_bias

        #Now, construct the bias2expansion and bias4expansion
        for i, states, vectors in zip(itertools.count(),self.vecpos[Nvstars_pure:],self.vecvec[Nvstars_pure:]):
            #First construct bias2expansion
            #omega_2 : mixed -> mixed
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega2):
                for j in jumplist:
                    IS=j.state1
                    dx = disp(j)
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if states[0]==IS:
                        geom_bias = np.dot(vectors[0], dx) #I haven't normalized with respect to no. of states.
                        bias2expansion[i, k] += geom_bias
            #Next, omega_3: mixed -> complex
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega3):
                for j in jumplist:
                    IS=j.state1
                    dx = disp(j)
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if states[0]==IS:
                        geom_bias = np.dot(vectors[0], dx) #I haven't normalized with respect to no. of states.
                        bias3expansion[i, k] += geom_bias
        return bias0expansion,bias1expansion,bias2expansion,bias3expansion,bias4expansion
