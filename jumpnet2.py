
# coding: utf-8

# # Function to Generate jumpnetworks for solute-dumbbell pairs
import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple
from representations import *


# First, we need to create all possible orientations of a given family
def gen_orsets(crys,or_pure,or_mixed):
    """
        Function to generate symmetry related orientations for a given crystal and orientation family.
        Parameters:
            crys - crystal object for which orientation sets are to be found
            or_pure - family of pure dumbbell orientation directions
            or_mixed - family of directions for mixed dumbbells.
    """
    if not isinstance(or_pure,np.ndarray) or not isinstance(or_mixed,np.ndarray):
        raise TypeError("Orientation vectors must be numpy arrays")
    if not len(or_pure)==len(or_mixed):
        raise TypeError ("Dimensionality of input vectors must be the same")
    zero = np.zeros(len(or_pure))
    or_pure_list = []
    or_pure_list.append(or_pure)
    or_mix_list = []
    or_mix_list.append(or_mixed)
    for g in crys.G:
        vec_pure_new = np.dot(g.cartrot,or_pure)
        if not any(np.allclose(vec,vec_pure_new,atol=1e-8) or np.allclose(vec+vec_pure_new,zero,atol=1e-8) for vec in or_pure_list):
            #For pure dumbbells, don't accept negative orientations
            or_pure_list.append(vec_pure_new)

        vec_mixed_new = np.dot(g.cartrot,or_mixed)
        if not any(np.allclose(vec,vec_mixed_new,atol=1e-8) for vec in or_mix_list):
            or_mix_list.append(vec_mixed_new)

    return or_pure_list,or_mix_list

# Now we construct the jumpnetwork based on these orientations.
# >Solute atom does not always remain at the centre.
# >It remains fixed for seperated pairs but moves for mixed dumbbells.
#
# >Initial try -for every jump other than omega_2 type (mixed dumbbell motion), keep the solute at
# the origin. For omega_2 type jumps, keep the intial location of the dumbbell at the origin. Since all
# jumps with different initial states are related by translational symmetry.

def jumpnet(crys,chem,oListPure,oListMix,cutoff,closestDistance=0):
    #If it is necessary for this to be in crystal module, must replace 'crys' with 'self' later.
    """
    Constructs jump jumpnetworks for each type of jump (omega 0,1,2,3,4) - grouped by symmetry
    Returns a list with all of those jumps and a dictionary that contains the ending indices
    for a given type of jump in that list
    Format of jumpnetwork elements - tuple(jump object,dx -> cartesian jump vector)
    parameters:
        crys - crystal structure (to calculate dx and symmetry-related jumps)
        oList* - family of respective orientations.
        cutoff - maximum allowable jump distance(dx_max).
        closestDistance - for collision detection
    Returns:
        1. lis - a list of jumps, grouped together into lists by symmetry
        2. jt_indices - where each type of jump ends in lis
    """
    def inlist(j,lis):
        return any(j==j_dx for jlist in lis for j_dx in jlist)

    # def rm_neg_or(j_new): put the orientation changing part here

    r2 = cutoff**2
    nmax = [int(np.round(np.sqrt(crys.metric[i,i]))) + 1 for i in range(3)]
    supervect = [np.array([n0,n1,n2]) for n0 in range(-nmax[0],nmax[0]+1)
                                      for n1 in range(-nmax[1],nmax[1]+1)
                                      for n2 in range(-nmax[2],nmax[2]+1)]
    zero = np.zeros(3,dtype=int)
    lis=[]
    jt_indices=[] #stores the ending indices for each type of jump in lis
    #Construct omega0 and omega1 type jumps.
    #Omega0 - only involves pure dumbbells
    #Jumps occurring out of the origin - yes
    for i,u1 in enumerate(crys.basis[chem]):
        for j,u2 in enumerate(crys.basis[chem]):
            for R2 in supervect:
                du = u2-u1
                dR = R2
                dx = crys.unit2cart(dR,du)
                if np.dot(dx,dx)>r2:
                    continue
                for o1 in oListPure:
                    for o2 in oListPure:
                        for c1 in [-1,1]:
                            for c2 in [-1,1]:
                                #catch diagonal case
                                if i==j and np.allclose(o1,o2) and np.allclose(zero,R2) and c1==c2:
                                    continue
                                db1 = dumbbell(i,o1,zero,c1)
                                db2 = dumbbell(j,o2,R2,c2)
                                jmp = jump(db1,db2)
                                if not inlist(jmp,lis):
                                    trans=[]
                                    trans.append(jmp)
                                    for g in crys.G:
                                        j_new = jmp.gop(crys,chem,g)
                                        ost1 = j_new.state1.o.copy()
                                        ost2 = j_new.state2.o.copy()
                                        #if any of the pure dumbbell orientations become negative
                                        #to what has been supplied in orsets, it is to be inverted.
                                        for ot in oListPure:
                                            if np.allclose(ost1+ot,zero,atol=1e-8):
                                                ost1 = -1*ost1
                                                state2 = j_new.state2
                                                state1 = dumbbell(j_new.state1.i,ost1,j_new.state1.R,j_new.state1.c*-1)
                                                j_new = jump(state1,state2)
                                                break
                                        for ot in oListPure:
                                            if np.allclose(ost2+ot,zero,atol=1e-8):
                                                ost2 = -1*ost2
                                                state1 = j_new.state1
                                                state2 = dumbbell(j_new.state2.i,ost2,j_new.state2.R,j_new.state2.c*-1)
                                                j_new = jump(state1,state2)
                                                break
                                        dRnew = j_new.state2.R - j_new.state1.R
                                        dunew = crys.basis[chem][j_new.state2.i] - crys.basis[chem][j_new.state1.i]
                                        dxnew = crys.unit2cart(dRnew,dunew)
                                        if not any(j_dx==j_new for j_dx in trans):
                                            # print(j_new.state1.o)
                                            trans.append(j_new)
                                    lis.append(trans)
    n = len(lis)
    jt_indices.append(n) #jumps involving pure dumbbells moving around

    #jumps with the solute being present at the origin and pure dumbbells jump in to form mixed dumbbells
    for i_s,u_s in enumerate(crys.basis[chem]):
        for j,u2 in enumerate(crys.basis[chem]):
            for R in supervect:
                if np.allclose(R,zero) and i_s==j:
                    continue
                du = u_s - u2
                dR = -R
                dx = crys.unit2cart(dR,du)
                if np.dot(dx,dx) > r2:
                    continue
                for o1 in oListPure:
                    for o2 in oListMix:
                        for c1 in [-1,1]:#get into both the representations
                            for c2 in [-1,1]:
                                #make the pairs
                                db1 = dumbbell(j,o1,R,c1)
                                db2 = dumbbell(i_s,o2,zero,c2)
                                pair1 = SdPair(i_s,zero,db1)
                                pair2 = SdPair(i_s,zero,db2)
                                jmp = jump(pair1,pair2)
                                if not inlist(jmp,lis[n:]):
                                    trans=[]
                                    trans.append(jmp)
                                    for g in crys.G:
                                        j_new = jmp.gop(crys,chem,g)
                                        ost1 = j_new.state1.db.o.copy()
                                        for ot in oListPure:
                                            if np.allclose(ost1+ot,zero,atol=1e-8):
                                                ost1 = -1*ost1
                                                state2 = j_new.state2
                                                db1_1 = dumbbell(j_new.state1.db.i,ost1,j_new.state1.db.R,j_new.state1.db.c*-1)
                                                state1 = SdPair(i_s,zero,db1_1)
                                                j_new = jump(state1,state2)
                                                break
                                        dRnew = j_new.state2.db.R - j_new.state1.db.R
                                        dunew = crys.basis[chem][j_new.state2.db.i] - crys.basis[chem][j_new.state1.db.i]
                                        dxnew = crys.unit2cart(dRnew,dunew)
                                        if not any(j_dx==j_new for j_dx in trans):
                                            # print(j_new.state1.o)
                                            trans.append(j_new)
                                    lis.append(trans)
    n = len(lis)
    jt_indices.append(n)

    #jumps with the mixed dumbbell being present at the origin and pure dumbbells form by solvent jumping out.
    for i_s,u_s in enumerate(crys.basis[chem]):
        for j,u2 in enumerate(crys.basis[chem]):
            for R in supervect:
                if np.allclose(R,zero) and i_s==j:
                    continue
                du = u2 - u_s
                dR = R
                dx = crys.unit2cart(dR,du)
                if np.dot(dx,dx) > r2:
                    continue
                for o1 in oListMix:
                    for o2 in oListPure:
                        for c2 in [-1,1]:
                            #make the pairs
                            db1 = dumbbell(i_s,o1,zero,-1)#solvent must jump out for dissociation
                            db2 = dumbbell(j,o2,R,c2)
                            pair1 = SdPair(i_s,zero,db1)
                            pair2 = SdPair(i_s,zero,db2)
                            jmp = jump(pair1,pair2)
                            if not inlist(jmp,lis[n:]):
                                trans=[]
                                trans.append(jmp)
                                for g in crys.G:
                                    j_new = jmp.gop(crys,chem,g)
                                    ost2 = j_new.state2.db.o.copy()
                                    for ot in oListPure:
                                        if np.allclose(ost1+ot,zero,atol=1e-8):
                                            ost1 = -1*ost1
                                            state1 = j_new.state1
                                            db2_1 = dumbbell(j_new.state2.db.i,ost1,j_new.state2.db.R,j_new.state2.db.c*-1)
                                            state2 = SdPair(i_s,zero,db2_1)
                                            j_new = jump(state1,state2)
                                            break
                                    dRnew = j_new.state2.db.R - j_new.state1.db.R
                                    dunew = crys.basis[chem][j_new.state2.db.i] - crys.basis[chem][j_new.state1.db.i]
                                    dxnew = crys.unit2cart(dRnew,dunew)
                                    if not any(j_dx==j_new for j_dx in trans):
                                        # print(j_new.state1.o)
                                        trans.append(j_new)
                                lis.append(trans)
    n = len(lis)
    jt_indices.append(n)

    #Construct mixed dumbbell jumps - initial solute location at the origin
    for i,u1 in enumerate(crys.basis[chem]):
        for j,u2 in enumerate(crys.basis[chem]):
            for R2 in supervect:
                du = u2-u1
                dR = R2
                dx = crys.unit2cart(dR,du)
                if np.dot(dx,dx)>r2:
                    continue
                for o1 in oListMix:
                    for o2 in oListMix:
                        # for c1 in [-1,1]: C1 is always 1 for mixed dumbbell jumps
                        for c2 in [-1,1]:
                            #catch diagonal case
                            if i==j and np.allclose(o1,o2) and np.allclose(zero,R2) and c1==c2:
                                continue
                            db1 = dumbbell(i,o1,zero,1)
                            pair1 = SdPair(i,zero,db1)
                            db2 = dumbbell(j,o2,R2,c2)
                            pair2 = SdPair(j,R2,db2)
                            try:
                                jmp = jump(pair1,pair2)
                            except:
                                continue
                            if not inlist(jmp,lis[n:]):
                                trans=[]
                                trans.append(jmp)
                                for g in crys.G:
                                    j_new = jmp.gop(crys,chem,g)
                                    dRnew = j_new.state2.db.R - j_new.state1.db.R
                                    dunew = crys.basis[chem][j_new.state2.db.i] - crys.basis[chem][j_new.state1.db.i]
                                    dxnew = crys.unit2cart(dRnew,dunew)
                                    if not any(j_dx==j_new for j_dx in trans):
                                        # print(j_new.state1.o)
                                        trans.append(j_new)
                                lis.append(trans)
    n=len(lis)
    jt_indices.append(n)
    return lis,jt_indices
