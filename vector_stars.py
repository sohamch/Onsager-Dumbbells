import numpy as np
import onsager.crystal as crystal
from onsager.crystalStars import zeroclean,VectorStarSet
from states import *
from representations import *
from stars import *
from functools import reduce
import itertools

class vectorStars(VectorStarSet):
    """
    Stores the vector stars corresponding to a given starset of dumbbell states
    """

    def generate(starset):
        """
        Follows almost the same as that for solute-vacancy case. Only generalized to keep the state
        under consideration unchanged.
        """
        self.starset=None
        if starset.Nshells == 0: return
        if starset == self.starset: return
        self.starset = starset
        self.vecpos = []
        self.vecpos_indexed = []
        self.vecvec = []
        #first do it for the complexes
        for star, indstar in zip(starset.stars[:starset.mixedstartindex],starset.starindexed[:starset.mixedstartindex]):
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
                    self.vecpos_indexed.append(indstar)
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
        for star, indstar in zip(starset.stars[starset.mixedstartindex:],starset.starindexed[starset.mixedstartindex:]):
            pair0 = star[0]
            glist=[]
            #Find group operations that leave state unchanged
            for g in starset.crys.G:
                pairnew = pair0.gop(starset.crys,starset.chem,g)
                #what about dumbbell rotations? Does not matter - the state has to remain unchanged
                #Is this valid for origin states too? verify - because we have origin states.
                if pairnew == pair0:
                    glist.append(g)
            #Find the intersected vector basis for these group operations
            vb=reduce(crystal.CombineVectorBasis,[crystal.VectorBasis(*g.eigen()) for g in glist])
            #Get orthonormal vectors
            vlist = starset.crys.vectlist(vb) #This also nomalizes with respect to length of the vectors.
            scale = 1./np.sqrt(len(star))
            vlist = [v * scale for v in vlist]
            Nvect = len(vlist)
            if Nvect > 0:#why did I put this? Makes sense to expand only if Nvects >0, otherwise there is zero bias.
            #verify this
                for v in vlist:
                    self.vecpos.append(star) #again, implement copy
                    self.vecpos_indexed.append(indstar)
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
    # We must produce two expansions. One for pure dumbbell states pointing to pure dumbbell state
    # and the other from mixed dumbbell states to mixed states.

    def genGFstarset(self):
        """
        Makes symmetrically grouped connections between the states in the starset, to be used as GFstarset for the pure and mixed state spaces.
        The connections must lie within the starset and must connect only those states that are connected by omega_0 or omega_2 jumps.
        The GFstarset is to be returned in the form of (i,j),dx. where the indices i and j correspond to the states in the iorset
        """
        def inTotalList(conn,totlist,mixed=False):
            if not mixed:
                ind1 = self.startset.pdbcontainer.iorindex.get(conn.state1)
                ind2 = self.startset.pdbcontainer.iorindex.get(conn.state2)
            else:
                ind1 = self.startset.mdbcontainer.iorindex.get(conn.state1)
                ind2 = self.startset.mdbcontainer.iorindex.get(conn.state2)

            dx = disp(self.startset.crys,self.startset.chem,conn.state1,conn.state2)
            return any(ind1==t[0][0] and ind2==t[0][1] and np.allclose(t[2],dx,threshold=self.kinetic.crys.threshold) for tlist in totlist for t in tlist)

        purestates = self.startset.purestates
        mixedstates = self.startset.mixedstates
        #Connect the states
        GFstarset_pure = []
        GFstarset_pure_starind = {}
        for st1 in purestates:
            for st2 in purestates:
                try:
                    s = s1^s2 #check XOR, if it is the same as the original code - yes, it is, but our sense of the operation is opposite
                except:
                    continue
                if inTotalList(s,GFstarset):
                    continue
                connectlist=[]
                for g in self.startset.crys.G:
                    db1new = self.startset.pdbcontainer.gdumb(s.state1)[0]
                    db2new = self.startset.pdbcontainer.gdumb(s.state2)[0]
                    dx=disp(self.startset.crys,self.kinetic.chem,db1new,db2new)
                    db1new = db1new - db1new.R
                    db2new = db2new - db2new.R
                    ind1 = self.startset.pdbcontainer.iorindex.get(db1new)
                    ind2 = self.startset.pdbcontainer.iorindex.get(db2new)
                    if ind1==None or ind2==None:
                        raise KeyError("dumbbell not found in iorlist")
                    tup = ((ind1,ind2),dx.copy())
                    if not any(t[0][0]==tup[0][0] and t[0][1]==tup[0][1] and np.allclose(tup[1],t[1],threshold=self.startset.crys.threshold) for t in connectlist):
                        connectlist.append(tup)
                for tup in connectlist:
                    GFstarset_pure_starind[tup] = len(GFstarset_pure)
                GFstarset_pure.append(connectlist)

        GFstarset_mixed = []
        GFstarset_mixed_starind = {}
        for st1 in mixedstates:
            for st2 in purestates:
                try:
                    s = s1^s2 #check XOR, if it is the same as the original code - yes, it is, but our sense of the operation is opposite
                except:
                    continue
                if inTotalList(s,GFstarset,mixed=True):
                    continue
                connectlist=[]
                for g in self.startset.crys.G:
                    snew = s.gop(self.starset.crys,self.starset.chem,g)
                    dx=disp(self.starset.crys,self.starset.chem,snew.state1,snew.state2)
                    snew = connector(snew.state1-snew.state1.R,snew.state2-snew.state2.R)
                    ind1 = self.starset.mdbcontainer.iorindex.get(snew.state1)
                    ind2 = self.starset.mdbcontainer.iorindex.get(snew.state2)
                    if ind1==None or ind2==None:
                        raise KeyError("dumbbell not found in iorlist")
                    tup = ((ind1,ind2),dx.copy())
                    if not any(t[0][0]==tup[0][0] and t[0][1]==tup[0][1] and np.allclose(tup[1],t[1],threshold=self.starset.crys.threshold) for t in connectlist):
                        connectlist.append(tup)
                for tup in connectlist:
                    GFstarset_mixed_starind[tup] = len(GFstarset_mixed)
                GFstarset_mixed.append(connectlist)

        return GFstarset_pure,GFstarset_pure_starind,GFstarset_mixed,GFstarset_mixed_starind

    def GFexpansion(self):
        self.GFstarset_pure,self.GFPureStarInd,self.GFstarset_mixed,self.GFMixedStarInd = self.genGFstarset()
        Nvstars_pure = self.Nvstars_pure
        Nvstars_mixed = self.Nvstars - self.Nvstars_pure
        GFexpansion_pure = np.zeros((Nvstars_pure,Nvstars_pure,len(self.GFstarset_pure)))
        GFexpansion_mixed = np.zeros((Nvstars_mixed,Nvstars_mixed,len(self.GFstarset_mixed))

        #build up the pure GFexpansion
        for i in range(Nvstars_pure):
            for si,vi in zip(self.vecpos[i],self.vecvec[i]):
                for j in range(Nvstars_pure):
                    for sj,vj in zip(self.vecpos[j],self.vecvec[j]):
                        try:
                            ds = si^sj
                        except:
                            continue
                            dx=disp(self.starset.crys,self.starset.chem,ds.state1,ds.state2)
                            ind1 = self.starset.pdbcontainer.iorindex.get(ds.state1)
                            ind2 = self.starset.pdbcontainer.iorindex.get(ds.state2)
                            if ind1==None or ind2==None:
                                raise KeyError("enpoint subtraction within starset not found in iorlist")
                            k = GFPureStarInd.get(((ind1,ind2),dx))
                            if k is None:
                                raise ArithmeticError("GF starset not big enough to accomodate state state pair {}".format(tup))
                            GFexpansion_pure[i, j, k] += np.dot(vi, vj)

        #Build up the mixed GF expansion
        for i in range(Nvstars_mixed):
            for si,vi in zip(self.vecpos[Nvstars_pure+i],self.vecvec[Nvstars_pure+i]):
                for j in range(Nvstars_pure):
                    for sj,vj in zip(self.vecpos[Nvstars_pure+j],self.vecvec[Nvstars_pure+j]):
                        try:
                            ds = si^sj
                        except:
                            continue
                            dx=disp(self.starset.crys,self.starset.chem,ds.state1,ds.state2)
                            ind1 = self.starset.pdbcontainer.iorindex.get(ds.state1)
                            ind2 = self.starset.pdbcontainer.iorindex.get(ds.state2)
                            if ind1==None or ind2==None:
                                raise KeyError("enpoint subtraction within starset not found in iorlist")
                            k = GFPureStarInd.get(((ind1,ind2),dx))
                            if k is None:
                                raise ArithmeticError("GF starset not big enough to accomodate state state pair {}".format(tup))
                            GFexpansion_pure[i, j, k] += np.dot(vi, vj)

        #symmetrize
        for i in range(Nvstars_pure):
            for j in range(0,i):
                GFexpansion_pure[i,j,:] = GFexpansion_pure[j,i,:]

        for i in range(Nvstars_mixed):
            for j in range(0,i):
                GFexpansion_mixed[i,j,:] = GFexpansion_pure[j,i,:]

        return (zeroclean(GFexpansion_pure),self.GFstarset_pure,self.GFPureStarInd), (zeroclean(GFexpansion_mixed),self.GFstarset_mixed,self.GFMixedStarInd)

    #See group meeting update slides of sept 10th to see how this works.
    def biasexpansion(self,jumpnetwork_omega1,jumpnetwork_omega2,jumptype,jumpnetwork_omega34):
        """
        Returns an expansion of the bias vector in terms of the displacements produced by jumps.
        Parameters:
            jumpnetwork_omega* - the jumpnetwork for the "*" kind of jumps (1,2,3 or 4)
            jumptype - the omega_0 jump type that gives rise to a omega_1 jump type (see jumpnetwork_omega1 function
            in stars.py module)
        Returns:
            bias0, bias1, bias2, bias4 and bias3 expansions, one each for solute and solvent
            Note - bias0 for solute makes no sense, so we return only for solvent.
        """
        z=np.zeros(3,dtype=float)
        #Expansion of pure dumbbell initial state bias vectors and complex state bias vectors
        bias0expansion = np.zeros((self.Nvstars_pure,len(self.starset.jumpindices)))
        bias1expansion_solvent = np.zeros((self.Nvstars_pure,len(jumpnetwork_omega1)))
        bias1expansion_solute = np.zeros((self.Nvstars_pure,len(jumpnetwork_omega1)))

        bias4expansion_solvent = np.zeros((self.Nvstars_pure,len(jumpnetwork_omega34)))
        bias4expansion_solute = np.zeros((self.Nvstars_pure,len(jumpnetwork_omega34)))

        #Expansion of mixed dumbbell initial state bias vectors.
        bias2expansion_solvent = np.zeros((self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega2)))
        bias2expansion_solute = np.zeros((self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega2)))

        bias3expansion_solvent = np.zeros((self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega34)))
        bias3expansion_solute = np.zeros((self.Nvstars-self.Nvstars_pure,len(jumpnetwork_omega34)))

        for i, purestar, purevstar in zip(itertools.count(),self.vecpos[:Nvstars_pure],self.vecvec[:Nvstars_pure]):
            #iterates over the rows of the matrix
            #First construct bias1expansion and bias0expansion
            #This contains the expansion of omega_0 jumps and omega_1 type jumps
            #See slides of Sept. 10 for diagram.
            #omega_0 : pure -> pure
            #omega_1 : complex -> complex
            for k,jumplist,jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
                #iterates over the columns of the matrix
                for j in jumplist:
                    IS=j.state1
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if purestar[0]==IS:
                        #sees if there is a jump of the kth type with purestar[0] as the initial state.
                        dx = disp(self.starset.crys,self.starset.chem,j.state1,j.state2)
                        dx_solute=z.copy()
                        dx_solvent = dx.copy() #just for clarity that the solvent mass transport is dx itself.

                        geom_bias_solvent = np.dot(vectors[0], dx)*len(purestar)
                        geom_bias_solute = np.dot(vectors[0], dx)*len(purestar)

                        bias1expansion_solvent[i, k] += geom_bias_solvent #this is contribution of kth_type of omega_1 jumps, to the bias
                        bias1expansion_solute[i, k] += geom_bias_solvent
                        #vector along v_i
                        #so to find the total bias along v_i due to omega_1 jumps, we sum over k
                        bias0expansion[i, jt] += geom_bias_solvent #These are the contributions of the omega_0 jumps
                        #to the bias vector along v_i, for bare dumbbells
                        #so to find the total bias along v_i, we sum over k.
            #Next, omega_4: complex -> mixed
            #The correction delta_bias = bias4 + bias1 - bias0
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist:
                    IS=j.state1
                    if IS.is_zero(): #check if initial state is mixed dumbbell -> then skip - it's omega_3
                        continue
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if purestar[0]==IS:
                        dx = disp(self.starset.crys,self.starset.chem,j.state1,j.state2)
                        dx_solute = j.state2.db.o/2.
                        dx_solvent = dx - j.state2.db.o/2.
                        geom_bias_solute = np.dot(vectors[0], dx_solute)*len(purestar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent)*len(purestar)
                        bias4expansion_solute[i, k] += geom_bias_solute #this is contribution of omega_4 jumps, to the bias
                        bias4expansion_solvent[i, k] += geom_bias_solvent
                        #vector along v_i
                        #So, to find the total bias along v_i due to omega_4 jumps, we sum over k.

        #Now, construct the bias2expansion and bias3expansion
        for i, mixedstar, vectors in zip(itertools.count(),self.vecpos[Nvstars_pure:],self.vecvec[Nvstars_pure:]):
            #First construct bias2expansion
            #omega_2 : mixed -> mixed
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega2):
                for j in jumplist:
                    IS=j.state1
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if mixedstar[0]==IS:
                        dx = disp(self.starset.crys,self.starset.chem,j.state1,j.state2)
                        dx_solute = dx + j.state2.db.o/2. - j.state1.db.o/2.
                        dx_solvent = dx - j.state2.db.o/2. + j.state1.db.o/2.
                        geom_bias_solute = np.dot(vectors[0], dx_solute)*len(mixedstar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent)*len(mixedstar)
                        bias2expansion_solute[i, k] += geom_bias
                        bias2expansion_solvent[i, k] += geom_bias
            #Next, omega_3: mixed -> complex
            for k,jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist:
                    if not IS.is_zero(): #check if initial state is not a mixed state -> skip if not mixed
                        continue
                    IS=j.state1
                # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if mixedstar[0]==IS:
                        dx = disp(self.starset.crys,self.starset.chem,j.state1,j.state2)
                        dx_solute = -j.state1.db.o/2.
                        dx_solvent = dx + j.state1.db.o/2.
                        geom_bias_solute = np.dot(vectors[0], dx_solute)*len(mixedstar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent)*len(mixedstar)
                        bias3expansion_solute[i, k] += geom_bias_solute
                        bias3expansion_solvent[i, k] += geom_bias_solvent
        return zeroclean(bias0expansion),(zeroclean(bias1expansion_solute),zeroclean(bias1expansion_solvent)),(zeroclean(bias2expansion_solute),zeroclean(bias2expansion_solvent)),\
               (zeroclean(bias3expansion_solute),zeroclean(bias3expansion_solvent)),(zeroclean(bias4expansion_solute),zeroclean(bias4expansion_solvent))

    def rateexpansion(self,jumpnetwork_omega1,jumptype,jumpnetwork_omega34):
        """
        Implements expansion of the jump rates in terms of the basis function of the vector stars.
        (Note to self) - Refer to earlier notes for details.
        """
        #See my slides of Sept. 10 for diagram
        rate0expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(self.starset.jumpindices)))
        rate1expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(jumpnetwork)))
        rate0escape = np.zeros((self.Nvstars_pure, len(self.starset.jumpindices)))
        rate1escape = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega1)))
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

        return zeroclean(rate0expansion),zeroclean(rate1expansion),zeroclean(rate2expansion),\
               zeroclean(rate3expansion),zeroclean(rate4expansion)
        #One more thing to think about - in our Dyson equation, the diagonal sum of om3 and om4 are added to om2
        #and om0 respectively. How to implement that? See where delta_omega is constructed for vacancies.. we need to do it there

    def bareexpansion(self,jumpnetwork_omega1,jumptype,jumpnetwork_omega2,jumpnetwork_omega3,jumpnetwork_omega4):
        """
        Returns the contributions to the terms of the uncorrelated diffusivity term,
        grouped separately for each type of jump. Intended to be called after displacements have been applied to the displacements.

        Params:
            jumpnetwork_omega* - indexed versions of the jumpnetworks with displacements for a given species. - jumps need to be of the form ((i,j),dx_species)
            jumptype - list that contains the omega_0 jump a given omega_1 jump list corresponds to. - these are the rates to be used to dot into b_0.

        In mixed dumbbell space, both solute and solvent will have uncorrelated contributions.
        The mixed dumbbell space is completely non-local.
        """
        # TODO: What about the cross-species terms - look up later
        D0expansion = np.zeros((3,3,len(self.starset.jumpindices)))
        D1expansion = np.zeros((3,3,len(jumpnetwork_omega1)))
        D2expansion = np.zeros((3,3,len(jumpnetwork_omega2)))
        D3expansion = np.zeros((3,3,len(jumpnetwork_omega3)))
        D4expansion = np.zeros((3,3,len(jumpnetwork_omega4)))
        #Need versions for solute and solvent
        for k, jt, jumplist in zip(itertools.count(), jumptype, jumpnetwork_omega1):
            d0 = np.sum(0.5 * np.outer(dx,dx) for (i,j),dx in jumplist)
            D0expansion_solute[:, :, jt] += d0_solute
            D1expansion_solute[:, :, k] += d0_solute

        for jt,jumplist in enumerate(self.starset.jumpnetwork_omega2):
            d0 = np.sum(0.5 * np.outer(dx, dx) for ISFS, dx in jumplist)
            D2expansion[:, :, jt] += d0

        for jt,jumplist in enumerate(self.starset.jumpnetwork_omega3):
            d0 = np.sum(0.5 * np.outer(dx, dx) for ISFS, dx in jumplist)
            D3expansion[:, :, jt] += d0

        for jt,jumplist in enumerate(self.starset.jumpnetwork_omega4):
            d0 = np.sum(0.5 * np.outer(dx, dx) for ISFS, dx in jumplist)
            D4expansion[:, :, jt] += d0

        return zeroclean(D0expansion), zeroclean(D1expansion), zeroclean(D2expansion), zeroclean(D3expansion), zeroclean(D4expansion)
