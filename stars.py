import numpy as np
import onsager.crystal as crystal
from jumpnet3 import *
from states import *
from representations import *

class StarSet(object):
    """
    class to form the crystal stars, with shells indicated by the number of jumps.
    Almost exactly similar to CrystalStars.StarSet except now includes orientations.
    The minimum shell (Nshells=0) is composed of dumbbells situated atleast one jump away.
    """
    def __init__(self,crys,chem,jumpnetwork,iormixed,Nshells,originstates=False):
        """
        Parameters:
        crys and chem - Respectively, the crystal and sublattice we are working on
        jumpnetwork - pure dumbbell jumpnetwork with which the star set is to be made.
        Nshells - the number of shells to be constructed. minimum is zero.
        fam_mixed = flat list of (i,or) tuples for mixed dumbbells
        """
        self.crys = crys
        self.chem = chem
        self.iormixed = iormixed
        self.dbstates = jumpnetwork[1]
        self.jumpnetwork = jumpnetwork
        self.jumplist = [j for l in jumpnetwork[0] for j in l]
        self.jumpset = set(self.jumplist)
        self.jumpindices = []
        self.Nshells = Nshells
        count=0
        for l in jumpnetwork[0]:
            self.jumpindices.append([])
            for j in l:
                self.jumpindices[-1].append(count)
                count+=1
        self.generate(Nshells,originstates)
        self.mixedset = [x for l in self.mstates.symorlist for x in l]

    def generate(self,Nshells,originstates):
        z=np.zeros(3).astype(int)
        if Nshells<1:
            Nshells = 0
        startshell=set([])
        stateset=set([])
        #build the starting shell
        for j in self.jumplist:
            #Build the first shell from the jumpnetwork- The initial dumbbell is in the origin, so assign it as the solute location.
            pair = SdPair(j.state1.i,j.state1.R,j.state2)
            # NEED ORIGIN STATES FOR GF-EXPANSION - LOOK AT DALLAS'S NOTES
            # redundant rotation jumps are taken care of in jumpnet3
            # if not originstates and pair.is_zero():
            #     continue
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
                    # if not originstates:
                    #     if pairnew.is_zero():
                    #         continue
                    nextshell.add(pairnew)
                    stateset.add(pairnew)
            lastshell = nextshell
            nextshell=set([])
        self.stateset = stateset
        #group the states by symmetry - form the starset
        self.starset=[]
        hashset=set([])
        for state in self.stateset:
            if not state in hashset:
                newstar=set([])
                for g in self.crys.G:
                    newstate = state.gop(self.crys,self.chem,g)
                    if not newstate in hashset and newstate in self.stateset:
                        newstar.add(newstate)
                        hashset.add(newstate)
                self.starset.append(list(newstar))
        self.mixedstartindex = len(self.starset)
        #Now add in the mixed states
        self.mstates = mStates(self.crys,self.chem,self.iormixed)
        self.mixedstateset=set([])
        for l in self.mstates.symorlist:
            newlist=[]
            for tup in l:
                db=dumbbell(tup[0],tup[1],z)
                mdb = SdPair(tup[0],z,db)
                newlist.append(mdb)
                self.mixedstateset.add(mdb)
            self.starset.append(newlist)

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
                            jnew1 = jpair.gop(self.crys,self.chem,g)
                            db1new = self.dbstates.gdumb(g,jpair.state1.db)
                            db2new = self.dbstates.gdumb(g,jpair.state2.db)
                            state1new = SdPair(jnew1.state1.i_s,jnew1.state1.R_s,db1new[0])
                            state2new = SdPair(jnew1.state2.i_s,jnew1.state2.R_s,db2new[0])
                            jnew = jump(state1new,state2new,jnew1.c1*db1new[1],jnew1.c2*db2new[1])
                            if not jnew in jumpset:
                                newlist.append(jnew)
                                newlist.append(-jnew)
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

    def jumpnetwork_omega2(self,cutoff,solt_solv_cut,closestdistance):
        jumpnetwork = mixedjumps(self.crys,self.chem,self.mixedset,cutoff,solt_solv_cut,closestdistance)
        return jumpnetwork

    def jumpnetwork_omega34(self,cutoff,solv_solv_cut,solt_solv_cut,closestdistance):
        #building omega_4 -> association - c2=-1 -> since solute movement is tracked
        #cutoff required is solute-solvent as well as solvent solvent
        alljumpset_omega4=set([])
        symjumplist_omega4=[]
        alljumpset_omega3=set([])
        symjumplist_omega3=[]
        symjumplist_omega34_all=[]
        alljumpset_omega43_all=set([])
        for p_pure in self.stateset:
            if p_pure.is_zero(): #Specator rotating into mixed does not make sense.
                continue
            for p_mixed in self.mixedstateset:
                for c1 in [-1,1]:
                    j = jump(p_pure,p_mixed,c1,-1)
                    if dx_excess(self.crys,self.chem,j.state1,j.state2,cutoff):continue
                    if not j in alljumpset_omega4:#check if jump already considered
                    #if a jump is in alljumpset_omega4, it's negative will have to be in alljumpset_omega3
                        if not collision_self(self.crys,self.chem,j,solv_solv_cut,solt_solv_cut):
                            if not collision_others(self.crys,self.chem,j,closestdistance):
                                newset=set([])
                                newnegset=set([])
                                new_allset=set([])
                                for g in self.crys.G:
                                    jnew = j.gop(self.crys,self.chem,g)
                                    db1new = self.dbstates.gdumb(g,j.state1.db)
                                    state1new = SdPair(jnew.state1.i_s,jnew.state1.R_s,db1new[0])
                                    jnew = jump(state1new,jnew.state2,jnew.c1*db1new[1],-1)
                                    if not jnew in newset:
                                        newset.add(jnew)
                                        newnegset.add(-jnew)
                                        new_allset.add(jnew)
                                        new_allset.add(-jnew)
                                        alljumpset_omega4.add(jnew)
                                symjumplist_omega4.append(list(newset))
                                symjumplist_omega3.append(list(newnegset))
                                symjumplist_omega43_all.append(list(new_allset))
                                
        return symjumplist_omega43_all,symjumplist_omega3,symjumplist_omega4
