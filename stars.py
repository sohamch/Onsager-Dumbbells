import numpy as np
import onsager.crystal as crystal
# from jumpnet3 import *
from states import *
import itertools
from representations import *

def calc_dx_species(crys,jnet,jnet_indexed,type='bare'):
    """
    Return a jumpnetwork for the individual species 'alpha' in the form (i,j,dx_alpha)
    Parameters:
    jnet - jumpnetwork with jumps in terms of states
    jnet_indexed - the indexed jumpnetwork.
    species - indicates which species we are calculating "dx" for.
    pure - True if we are working in pure dumbell space, False if in mixed dumbell space
    Returns:
    symmetry grouped jumps of the form (i,j,dx_species)
    """
    if not(type=='bare' or type=='mixed'):
        raise ValueError('the type can only be bare or mixed')
    if len(jnet_indexed!=jnet):
        raise ValueError("Need the same indexed jumplist as the original jumplist")
    if type=="bare":
        if not isinstance(jnet[0][0].state1,dumbbell):
            raise TypeError("bare dumbell transitions need to be between dumbbell objects")
    else:
        if not isinstance(jnet[0][0].state1,SdPair):
            raise TypeError("mixed dumbbell or complex transitions need to be between SdPair objects")
    #First deal with solute in pure dumbbell space
    jnet_solvent=[]
    jnet_solute=[]
    if pure:
        for i,jlist in enumerate(jnet):
            speclist=[]
            for j,jmp in enumerate(jlist):
                dx=jnet_indexed[i][j][2]
                dx += (jmp.c2*jmp.state2.o/2. - jmp.c1*jmp.state1.o/2.)
                speclist.append(((jnet_indexed[i][j][0][0],jnet_indexed[i][j][0][1]),dx))
            jnet_solvent.append(speclist)
        return jnet_solvent,None
    else:
        for i,jlist in enumerate(jnet):
            speclist_solute=[]
            speclist_solvent=[]
            for j,jmp in enumerate(jlist):
                dx=jnet_indexed[i][j][2]
                dx_solute = dx + (jmp.state2.db.o/2. - jmp.state1.db.o/2.)
                dx_solvent = dx + (-jmp.state2.db.o/2. + jmp.state1.db.o/2.)
                speclist_solute.append(((jnet_indexed[i][j][0][0],jnet_indexed[i][j][0][1]),dx_solute))
                speclist_solvent.append(((jnet_indexed[i][j][0][0],jnet_indexed[i][j][0][1]),dx_solvent))
            jnet_solvent.append(speclist)
        return jnet_solvent,jnet_solute



class StarSet(object):
    """
    class to form the crystal stars, with shells indicated by the number of jumps.
    Almost exactly similar to CrystalStars.StarSet except now includes orientations.
    The minimum shell (Nshells=0) is composed of dumbbells situated atleast one jump away.
    """
    def __init__(self,pdbcontainer,mdbcontainer,jumpnetwork_omega0,jumpnetwork_omega2,Nshells=None):#,originstates=False):
        """
        Parameters:
        pdbcontainer,mdbcontainer:
            -containers containing the pure and mixed dumbbell information respectively
        jumpnetwork_omega0,jumpnetwork_omega2 - jumpnetworks in pure and mixed dumbbell spaces respectively.
            Note - must send in both as pair states and indexed.
        Nshells - number of thermodynamic shells. Minimum - one jump away - corresponds to Nshells=0
        """
        #check that we have the same crystal structures for pdbcontainer and mdbcontainer
        if not np.allclose(pdbcontainer.crys.lattice,mdbcontainer.crys.lattice):
            raise TypeError("pdbcontainer and mdbcontainer have different crystals")

        if not len(pdbcontainer.crys.basis)==len(mdbcontainer.crys.basis):
            raise TypeError("pdbcontainer and mdbcontainer have different basis")

        for atom1,atom2 in zip(pdbcontainer.crys.chemistry,mdbcontainer.crys.chemistry):
            if not atom1==atom2:
                raise TypeError("pdbcontainer and mdbcontainer basis atom types don't match")
        for l1,l2 in zip(pdbcontainer.crys.basis,mdbcontainer.crys.basis):
            if not l1==l2:
                raise TypeError("basis atom types have different numbers in pdbcontainer and mdbcontainer")

        if not pdbcontainer.chem==mdbcontainer.chem:
            raise TypeError("pdbcontainer and mdbcontainer have states on different sublattices")


        self.crys = pdbcontainer.crys
        self.chem = pdbcontainer.chem
        self.pdbcontainer = pdbcontainer
        self.mdbcontainer = mdbcontainer
        self.mixedset = mdbcontainer.iorlist
        self.jumpnetwork = jumpnetwork_omega0[0]

        self.jumpnetwork_indexed = jumpnetwork_omega0[1]
        #get version of the indexed jumpnetwork containing ((i,j),dx_alpha) type jump representations
        #where alpha='solute' means solute displacement, alpha = 'solvent' means solvent displacement
        self.jumpnetwork_indexed_solute, self.jumpnetwork_indexed_solvent = calc_dx_species()

        self.jumpnetwork_omega2 = jumpnetwork_omega2[0]

        self.jumpnetwork_omega2_indexed = jumpnetwork_omega2[1]

        self.jumplist = [j for l in self.jumpnetwork for j in l]
        self.jumpset = set(self.jumplist)
        self.jumpindices = []
        self.Nshells = Nshells
        count=0
        for l in self.jumpnetwork:
            self.jumpindices.append([])
            for j in l:
                self.jumpindices[-1].append(count)
                count+=1
        if not Nshells==None:
            self.generate(Nshells)

    def generate(self,Nshells):
        #Return nothing if Nshells are not specified
        if Nshells==None: return
        z=np.zeros(3).astype(int)
        if Nshells<=1:
            #A minimum of one shell will be produced.
            Nshells = 0
        startshell=set([])
        stateset=set([])
        #build the starting shell
        for j in self.jumplist:
            #Build the first shell from the jumpnetwork- The initial dumbbell is in the origin, so assign it as the solute location.
            pair = SdPair(j.state1.i,j.state1.R,j.state2)
            startshell.add(pair)
            stateset.add(pair)
        lastshell=startshell
        nextshell=set([])
        #Now build the next shells:
        for step in range(Nshells):
            for j in self.jumplist:
                for pair in lastshell:
                    if not np.allclose(pair.R_s,0,atol=self.crys.threshold):
                        raise RuntimeError("The solute is not at the origin")
                    try:
                        pairnew = pair.addjump(j)
                        if not pair.i==pairnew.i and np.allclose(pairnew.R_s,pair.R_s,atol=self.crys.threshold):
                            raise RuntimeError("Solute shifted from a complex!(?)")
                    except:
                        continue
                    nextshell.add(pairnew)
                    stateset.add(pairnew)
            lastshell = nextshell
            nextshell=set([])
        self.stateset = stateset
        #group the states by symmetry - form the stars
        self.stars=[]
        hashset=set([])
        for state in self.stateset:
            if not state in hashset:
                newstar=[]
                for g in self.crys.G:
                    newstate = state.gop(self.crys,self.chem,g)
                    newdb = self.pdbcontainer.gdumb(g,state.db)[0] - newstate.R_s
                    newstate = SdPair(newstate.i_s,np.zeros(3,dtype=int),newdb)
                    if not newstate in hashset and newstate in self.stateset:
                        newstar.append(newstate)
                        hashset.add(newstate)
                self.stars.append(newstar)

        for sl in self.stars:
            for s in sl:
                if not np.allclose(s.R_s,0,atol=self.crys.threshold):
                    raise RuntimeError("Solute not at origin")

        self.mixedstartindex = len(self.stars)
        #Now add in the mixed states
        self.mixedstateset=set([])
        for l in self.mdbcontainer.symorlist:
            #The sites and orientations are already grouped - convert them into SdPairs
            newlist=[]
            for tup in l:
                db=dumbbell(tup[0],tup[1],z)
                mdb = SdPair(tup[0],z,db)
                newlist.append(mdb)
                self.mixedstateset.add(mdb)
            self.stars.append(newlist)
        self.purestates = list(self.stateset)
        self.mixedstates = list(self.mixedstateset)

        #generate an indexed version of the starset - seperate for mixed and pure stars
        starindexed = []
        for star in self.stars[:self.mixedstartindex]:
            indlist=[]
            for state in star:
                for j,st in enumerate(self.purestates):
                    if st==state:
                        indlist.append(j)
            starindexed.append(indlist)

        for star in self.stars[self.mixedstartindex:]:
            indlist=[]
            for state in star:
                for j,st in enumerate(self.mixedstates):
                    if st==state:
                        indlist.append(j)
            starindexed.append(indlist)

        self.starindexed = starindexed
        #self.starindexed -> gives the indices to the states list of the states sotred in the starset

        #now generate the index dicts
        # --starindex -> tells us which star (via it's index in the states list) a state belongs to.
        # --indexdict -> tell us given a pair state, what is its index in the states list and the starset.
        self.pureindex = np.zeros(len(self.purestates),dtype=int)
        self.mixedindex = np.zeros(len(self.mixedstates),dtype=int)
        self.pureindexdict = {}
        self.mixedindexdict = {}

        for si, star, starind in zip(itertools.count(),self.stars[:self.mixedstartindex],\
        self.starindexed[:self.mixedstartindex]):
            for state,ind in zip(star,starind):
                self.pureindex[ind] = si
                self.pureindexdict[state] = (ind, si)

        for si, star, starind in zip(itertools.count(),self.stars[self.mixedstartindex:],\
        self.starindexed[self.mixedstartindex:]):
            for state,ind in zip(star,starind):
                self.mixedindex[ind] = si+self.mixedstartindex
                self.mixedindexdict[state] = (ind, si+self.mixedstartindex)

    def jumpnetwork_omega1(self):
        jumpnetwork=[]
        jumptype=[]
        starpair=[]
        jumpset=set([])#set where newly produced jumps will be stored
        for jt,j_indices in enumerate(self.jumpindices):
            for j in [self.jumplist[q] for q in j_indices]:
            #these contain dumbell->dumbell jumps
                for pair in self.stateset:
                    try:
                        pairnew=pair.addjump(j)
                    except:
                        continue
                    # if pairnew.is_zero():#consider only non to-origin jumps
                    #     continue
                    if not pairnew in self.stateset:
                        continue
                    #convert them to pair jumps
                    jpair = jump(pair,pairnew,j.c1,j.c2)
                    if not jpair in jumpset and not -jpair in jumpset: #see if the jump has not already been considered
                        newlist=[]
                        for g in self.crys.G:
                            jnew = jpair.gop(self.crys,self.chem,g)
                            db1new = self.pdbcontainer.gdumb(g,jpair.state1.db)
                            db2new = self.pdbcontainer.gdumb(g,jpair.state2.db)
                            #The solute must be at the origin unit cell - shift it
                            state1new = SdPair(jnew.state1.i_s,jnew.state1.R_s,db1new[0])-jnew.state1.R_s
                            state2new = SdPair(jnew.state2.i_s,jnew.state2.R_s,db2new[0])-jnew.state2.R_s
                            jnew = jump(state1new,state2new,jnew.c1*db1new[1],jnew.c2*db2new[1])
                            if not jnew in jumpset:
                                if not (np.allclose(jnew.state1.R_s,0.,atol=self.crys.threshold) and np.allclose(jnew.state2.R_s,0.,atol=self.crys.threshold)):
                                    raise RuntimeError("Solute shifted from origin")
                                if not(jnew.state1.i_s==jnew.state1.i_s):
                                    raise RuntimeError("Solute must remain in exactly the same position before and after the jump")
                                newlist.append(jnew)
                                newlist.append(-jnew) #we can add the negative since solute always remain at the origin
                                jumpset.add(jnew)
                                jumpset.add(-jnew)
                        if (newlist[0].state1.is_zero() and newlist[0].state2.is_zero()):
                            #remove redundant rotations
                            newnewlist=set([])
                            for j in newlist:
                                j_equiv = jump(j.state1,j.state2,-1*j.c1,-1*j.c2)
                                if not j_equiv in newnewlist:
                                    newnewlist.add(j)
                            newlist=list(newnewlist)
                        jumpnetwork.append(newlist)
                        jumptype.append(jt)

        return jumpnetwork, jumptype

    def jumpnetwork_omega34(self,cutoff,solv_solv_cut,solt_solv_cut,closestdistance):
        #building omega_4 -> association - c2=-1 -> since solvent movement is tracked
        #cutoff required is solute-solvent as well as solvent solvent
        alljumpset_omega4=set([])
        symjumplist_omega4=[]
        alljumpset_omega3=set([])
        symjumplist_omega3=[]
        symjumplist_omega43_all=[]
        alljumpset_omega43_all=set([])
        for p_pure in self.stateset:
            if p_pure.is_zero(): #Specator rotating into mixed does not make sense.
                continue
            for p_mixed in self.mixedstateset:
                for c1 in [-1,1]:
                    try:
                        j = jump(p_pure,p_mixed,c1,-1)
                    except:
                        continue
                    #The next four lines should be commented out when ready
                    if not (np.allclose(p_pure.R_s,0,atol=self.crys.threshold) and np.allclose(p_mixed.R_s,0,atol=self.crys.threshold)):
                        raise RuntimeError("Solute shifted from origin - cannot happen")
                    # if not(p_pure.i_s==p_mixed.i_s): #The solute must remain in exactly the same position before and after the jump
                    #     raise RuntimeError("Incorrect jump constructed")
                    dx = disp(self.crys,self.chem,j.state1,j.state2)
                    if np.dot(dx,dx)>cutoff**2:continue
                    if not j in alljumpset_omega4:#check if jump already considered
                    #if a jump is in alljumpset_omega4, it's negative will have to be in alljumpset_omega3
                        if not collision_self(self.crys,self.chem,j,solv_solv_cut,solt_solv_cut):
                            if not collision_others(self.crys,self.chem,j,closestdistance):
                                newset=set([])
                                newnegset=set([])
                                new_allset=set([])
                                for g in self.crys.G:
                                    jnew = j.gop(self.crys,self.chem,g)
                                    db1new = self.pdbcontainer.gdumb(g,j.state1.db)
                                    state1new = SdPair(jnew.state1.i_s,jnew.state1.R_s,db1new[0]) - jnew.state1.R_s
                                    jnew = jump(state1new,jnew.state2-jnew.state2.R_s,jnew.c1*db1new[1],-1)
                                    if not jnew in newset:
                                        if jnew.state1.i_s==jnew.state1.db.i and np.allclose(jnew.state1.R_s,jnew.state1.db.R,atol=self.crys.threshold):
                                            raise RuntimeError("Initial state mixed")
                                        newset.add(jnew)
                                        newnegset.add(-jnew)
                                        new_allset.add(jnew)
                                        new_allset.add(-jnew)
                                        alljumpset_omega4.add(jnew)
                                symjumplist_omega4.append(list(newset))
                                symjumplist_omega3.append(list(newnegset))
                                symjumplist_omega43_all.append(list(new_allset))

        return symjumplist_omega43_all,symjumplist_omega4,symjumplist_omega3
