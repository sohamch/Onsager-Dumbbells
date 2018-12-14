import numpy as np
import onsager.crystal as crystal
from jumpnet3 import *
from states import *
from representations import *
from stars import *
from functools import reduce
import itertools

class vectorStars(object):
    """
    Stores the vector stars corresponding to a given starset of dumbbell states
    """

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
        for star in starset.stars[:starset.mixedstartindex]:
            pair0 = star[0]
            glist=[]
            #Find group operations that leave state unchanged
            for g in starset.crys.G:
                pairnew = pair0.gop(starset.crys,starset.chem,g)
                if pairnew == pair0 or pairnew==-pair0:
                    glist.append(g)
            #Find the intersected vector basis for these group operations
            vb=reduce(crystal.CombineVectorBasis,[crystal.VectorBasis(*g.eigen()) for g in glist])
            #Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)
            scale = 1./np.sqrt(len(star))
            vlist = [v * scale for v in vlist] # see equation 80 in the paper - (there is a typo, this is correct).
            Nvect=len(vlist)
            if Nvect > 0:
                for v in vlist:
                    self.vecpos.append(star)
                    #implement a copy function like in case of vacancies
                    veclist = []
                    for pairI in star:
                        for g in starset.crys.G:
                            pairnew = pair0.g(starset.crys, starset.chem, g)
                            pairnew = pairnew - pairnew.R_s #translate solute back to origin
                            if pairnew == pairI or pairnew == -pairI:
                                veclist.append(starset.crys.g_direc(g, v))
                                break
                    self.vecvec.append(veclist)
            self.Nvstars_pure = len(vecpos)
        #Now do it for the mixed dumbbells - all negative checks dissappear
        for star in starset.stars[starset.mixedstartindex:]:
            pair0 = star[0]
            glist=[]
            #Find group operations that leave state unchanged
            for g in starset.crys.G:
                pairnew = pair0.gop(starset.crys,starset.chem,g)
                if pairnew == pair0:
                    glist.append(g)
            #Find the intersected vector basis for these group operations
            vb=reduce(crystal.CombineVectorBasis,[crystal.VectorBasis(*g.eigen()) for g in glist])
            #Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)
            scale = 1./np.sqrt(len(star))
            vlist = [v * scale for v in vlist]
            Nvect = len(vlist)
            if Nvect > 0:
                for v in vlist:
                    self.vecpos.append(star) #again, implement copy
                    veclist = []
                    for pairI in star:
                        for g in starset.crys.G:
                            pairnew = pair0.g(starset.crys, starset.chem, g)
                            pairnew = pairnew - pairnew.R_s #translate solute back to origin
                            if pairnew == pairI:
                                veclist.append(starset.crys.g_direc(g, v))
                                break
                    self.vecvec.append(veclist)
            self.Nvstars = len(vecpos)

    #See group meeting update slides of sept 10th to see how this works.
    def biasexpansion(self,jumpnetwork_omega1,jumptype,jumpnetwork_omega34):
        """
        Returns an expansion of the bias vector in terms of the displacements produced by jumps.
        Parameters:
            jumpnetwork_omega* - the jumpnetwork for the "*" kind of jumps (1,2,3 or 4)
            jumptype - the omega_0 jump type that gives rise to a omega_1 jump type (see jumpnetwork_omega1 function
            in stars.py module)
        Returns:
            bias0, bias1, bias2, bias4 and bias3 expansions
        """
        #Expansion of pure dumbbell state bias vectors and complex state bias vectors
        bias0expansion = np.zeros((self.Nvstars_pure,len(self.starset.jumpindices)))
        bias1expansion = np.zeros((self.Nvstars_pure,len(jumpnetwork_omega1)))

        bias4expansion = np.zeros((self.Nvstars_pure,len(jumpnetwork_omega34)))

        #Expansion of mixed dumbbell state bias vectors.
        bias2expansion = np.zeros((self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega2)))
        bias3expansion = np.zeros((self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega34)))

        for i, states, vectors in zip(itertools.count(),self.vecpos[:Nvstars_pure],self.vecvec[:Nvstars_pure]):
            #First construct bias1expansion and bias0expansion
            #This contains the expansion of omega_0 jumps and omega_1 type jumps
            #See slides of Sept. 30 for diagram.
            #omega_0 : pure -> pure
            #omega_1 : complex -> complex
            for k,jumplist,jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
                for j in jumplist:
                    IS=j.state1
                    dx = disp(self.starset.crys,self.starset.chem,j.state1,j.state2)
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if states[0]==IS:
                        geom_bias = np.dot(vectors[0], dx)
                        bias1expansion[i, k] += geom_bias
                        bias0expansion[i, jt] += geom_bias
            #Next, omega_4: complex -> mixed
            #The correction delta_bias = bias4 + bias1 - bias0
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist:
                    IS=j.state1
                    if IS.is_zero(): #check if initial state is mixed dumbbell -> then skip
                        continue
                    dx = disp(self.starset.crys,self.starset.chem,j.state1,j.state2)
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if states[0]==IS:
                        geom_bias = np.dot(vectors[0], dx) #I haven't normalized with respect to no. of states.
                        bias4expansion[i, k] += geom_bias

        #Now, construct the bias2expansion and bias3expansion
        for i, states, vectors in zip(itertools.count(),self.vecpos[Nvstars_pure:],self.vecvec[Nvstars_pure:]):
            #First construct bias2expansion
            #omega_2 : mixed -> mixed
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega2):
                for j in jumplist:
                    IS=j.state1
                    dx = disp(self.starset.crys,self.starset.chem,j.state1,j.state2)
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if states[0]==IS:
                        geom_bias = np.dot(vectors[0], dx) #I haven't normalized with respect to no. of states.
                        bias2expansion[i, k] += geom_bias
            #Next, omega_3: mixed -> complex
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist:
                    if not IS.is_zero(): #check if initial state is not a mixed state -> skip if not mixed
                        continue
                    IS=j.state1
                    dx = disp(self.starset.crys,self.starset.chem,j.state1,j.state2)
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if states[0]==IS:
                        geom_bias = np.dot(vectors[0], dx) #I haven't normalized with respect to no. of states.
                        bias3expansion[i, k] += geom_bias
        return bias0expansion,bias1expansion,bias2expansion,bias3expansion,bias4expansion

    def rateexpansion(self,jumpnetwork_omega1,jumptype,jumpnetwork_omega34):
        """
        Implements expansion of the jump rates in terms of the basis function of the vector stars.
        (Note to self) - Refer to earlier notes for details.
        """
        #See my slides of Sept. 30 for diagram
        rate0expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(self.starset.jumpindices)))
        rate1expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(jumpnetwork)))
        rate0escape = np.zeros((self.Nvstars_pure, len(self.starset.jumpindices)))
        rate1escape = np.zeros((self.Nvstars_pure, len(jumpnetwork)))
        #First, we do the rate1 and rate0 expansions
        for k,jumplist,jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
            for jmp in jumplist:
                for i in range(self.Nvstars_pure): #The first inner sum
                    for chi_i,vi in zip(self.vecpos[i],self.vecvec[i]):
                        if chi_i==jmp.state1:#This is the delta functions of chi_0
                            rate0escape[i, jt] -= np.dot(vi, vi)
                            rate1escape[i, k] -= np.dot(vi, vi)
                            for j in range(self.Nvstars_pure): #The second inner sum
                                for chi_j,vj in zip(self.vecpos[j],self.vecvec[j]):
                                    if chi_j==jmp.state2: #this is the delta function of chi_1
                                        rate1expansion[i,j,k] += np.dot(vi,vj)
                                        rate0expansion[i,j,jt] += np.dot(vi,vj)
        #Next, let's do the rate4expansion -> complex to mixed jumps
        rate4expansion = np.zeros((self.Nvstars_pure,self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega34)))
        #The initial states are complexes, the final states are mixed and there are as many symmetric jumps as in jumpnetwork_omega34
        rate3expansion = np.zeros((self.Nvstars-self.Nvstars_pure,self.Nvstars_pure,len(jumpnetwork_omega34)))
        #The initial states are mixed, the final states are complex and there are as many symmetric jumps as in jumpnetwork_omega34
        rate3escape = np.zeros((self.Nvstars_pure, len(self.starset.jumpindices)))
        rate4escape = np.zeros((self.Nvstars-self.Nvstars_pure, len(jumpnetwork)))

        #We implement the math for omega4 and note that omega3 is the negative jump of omega4
        #This is because the order of the sum over stars in the rate expansion does not matter (see Sept. 30 slides).
        for k,jumplist in zip(itertools.count(), jumpnetwork_omega34):
            for jmp in jumplist:
                if jmp.state1.is_zero(): #the initial state must be a complex
                                         #the negative of this jump is an omega_3 jump anyway
                    continue
                for i in range(self.Nvstars_pure): # iterate over complex states - the first inner sum
                    for chi_i,vi in zip(self.vecpos[i],self.vecvec[i]):
                        #Go through the initial pure states
                        if chi_i==jmp.state1:
                            rate4escape[i,k] -= np.dot(vi,vi)
                            for j in range(self.Nvstars_pure,self.Nvstars): #iterate over mixed states - the second inner sum
                                for chi_j,vj in zip(self.vecpos[j],self.vecvec[j]):
                                    #Go through the final complex states
                                    if chi_j==jmp.state2:
                                        rate3escape[j-self.Nvstars_pure,k] -= np.dot(vj,vj)
                                        rate4expansion[i,j,k] += np.dot(vi,vj)
                                        rate3expansion[j,i,k] += np.dot(vj,vi)
                                        #The jump type remains the same because they have the same transition state

        #Next, we do the rate2expansion for mixed->mixed jumps
        rate2expansion = np.zeros((self.Nvstars-self.Nvstars_pure,self.Nvstars-self.Nvstars_pure, len(jumpnetwork_omega2)))
        rate2escape = np.zeros((self.Nvstars-self.Nvstars_pure, len(jumpnetwork_omega2)))
        for k,jumplist in zip(itertools.count(), jumpnetwork_omega2):
            for jmp in jumplist:
                for i in range(self.Nvstars_pure,self.Nvstars):
                    for chi_i,vi in zip(self.vecpos[i],self.vecvec[i]):
                        if chi_i==jmp.state1:
                            rate2escape[i,k] -= np.dot(vi,vi)
                            for j in range(self.Nvstars_pure,self.Nvstars):
                                for chi_j,vj in zip(self.vecpos[j],self.vecvec[j]):
                                    if chi_j.i_s==jmp.state2.i_s and np.allclose(chi_j.R_s,jmp.state2.R_s) and chi_j.db.i==jmp.state2.db.i and np.allclose(chi_j.db.o,jmp.state2.db.o):
                                        rate2expansion[i,j,k] += np.dot(vi,vj)

        return rate0expansion,rate1expansion,rate2expansion,rate3expansion,rate4expansion
        #One more thing to think about - in our Dyson equation, the diagonal sum of om3 and om4 are added to om2
        #and om0 respectively. How to implement that? See where delta_omega is constructed for vacancies.. we need to do it there


    def bareexpansion(self,jumpnetwork_omega1,jumptype):
        """
        Returns the contributions to the terms of the bare diffusivity term,
        grouped separately for each type of jump.
        """
        D0expansion = np.zeros((3,3,len(self.starset.jumpindices)))
        D1expansion = np.zeros((3,3,len(jumpnetwork_omega1)))
        #The next part should be exactly the same as for the vacancy case
        for k, jt, jumplist in zip(itertools.count(), jumptype, jumpnetwork):
            d0 = np.sum(0.5 * np.outer(dx, dx) for ISFS, dx in jumplist)
            D0expansion[:, :, jt] += d0
            D1expansion[:, :, k] += d0
        return D0expansion, D1expansion
