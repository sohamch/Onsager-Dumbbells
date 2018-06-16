# coding: utf-8

# # Function to Generate jumpnetworks for solute-dumbbell pairs
import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple
from representations import *
from collision import *

#first generate orientation states
def gen_orsets(crys,chem,p_or_fam,m_or_fam):
    """
    Returns all the members of the families of orientations at given sites
    params:
        crys, chem - working crystal and sublattice respectively.
        p_or_fam: list of vectors that represent orientations at symmetrically unrelated sites for pure dumbbells.
        m_or_fam: list of vectors that represent orientations at symmetrically unrelated sites for mixed dumbbells.

    Returns:
        [purelist,mixedlist] - a list of 2 lists that contain, sitewise, all the members of a family of directions
        for pure and mixed dumbbells respectively.
    """
    z = np.zeros(3)
    n = len(crys.sitelist(chem))
    if not (n==len(p_or_fam)==len(m_or_fam)):
        raise TypeError("Some sites are symmetrically related and so must draw orientations from same family.")
    #generate members of the families
    purelist = []
    mixedlist = []
    for i in range(n):
        oplist=[]
        omlist=[]
        for j in range(len(p_or_fam[i])):
            op = p_or_fam[i][j]
            oplist.append(op)
            om = m_or_fam[i][j]
            omlist.append(om)
            for g in crys.G:
                op_new = np.dot(g.cartrot,op)
                if not (any(np.allclose(o,op_new) for o in oplist) or any(np.allclose(o+op_new,z,atol=1e-8) for o in oplist)):
                    oplist.append(op_new)
                om_new = np.dot(g.cartrot,om)
                if not any(np.allclose(o,om_new) for o in omlist):
                    omlist.append(om_new)
        purelist.append(oplist)
        mixedlist.append(omlist)
    return purelist,mixedlist

def gensets(crys,chem,purelist,mixedlist):
    pairs_pure=[]
    pairs_mixed=[]
    wyck_sites = crys.sitelist(chem)
    n = len(wyck_sites)
    for i in range(n):
        plist = purelist[i]
        mlist = mixedlist[i]
        atomlist=wyck_sites[i]
        for j in range(len(atomlist)):
            site_or_pair=[]
            for op in plist:
                site_or_pair.append(tuple([atomlist[j],op]))
            pairs_pure.append(site_or_pair)
            site_or_pair=[]
            for op in mlist:
                site_or_pair.append(tuple([atomlist[j],op]))
            pairs_mixed.append(site_or_pair)
    return pairs_pure,pairs_mixed

def jumpnetwork(crys,chem,pairs_pure,pairs_mixed,cutoff,cutoff12,cutoff13):
    z=np.zeros(3)
    states_pure = pairs_pure.copy()
    states_mixed = pairs_mixed.copy()
    def makevalid(jmp,first=0,second=0):
        #checks whether a dumbbell state is allowed
        #checks if an orientation has been reversed for a pure dumbbell
        #first extract necessary data
        db1 = jmp.state1 if isinstance(jmp.state1,dumbell) else jmp.state1.db
        db2 = jmp.state2 if isinstance(jmp.state2,dumbell) else jmp.state2.db
        c1 = jmp.c1
        c2=jmp.c2
        jmpl=list(jump)#convert jump to a list to support item assignment

        #see which states in the jump are indicated to have pure and mixed dumbbells
        if first!=0 and second!=0:
            purelist=[(0,db1,c1),(1,db2,c2)]
            mixedlist=[]
        elif first!=0 and second==0:
            purelist=[(0,db1,c1)]
            mixedlist=[(1,db2,c2)]
        elif first==0 and second!=0:
            purelist=[(1,db2,c2)]
            mixedlist=[(0,db1,c1)]
        elif first==0 and second==0:
            purelist=[]
            mixedlist=[(0,db1,c1),(1,db2,c2)]

        #first check if mixed list is sane
        if len(mixedlist)>0:
            for p,db,c in mixedlist:
                i=db.i
                o=db.o
                if not any(j==i and np.allclose(o,o1,atol=1e-8) for j,o1 in states_mixed):
                    return None
        #provided mixed list is sane, then move on to pure list
        if len(purelist)>0:
            for p,db,c in purelist:
                i=db.i
                o=db.o
                if not (any(j==i and np.allclose(o,o1,atol=1e-8) for j,o1 in states_mixed) or any(j==i and np.allclose(o+o1,z,atol=1e-8) for j,o1 in states_mixed)):
                    return None
                for st in states_pure:
                    if (i==st[0] and np.allclose(o,st[1])):
                        jmpl[p]=db
                        break
                    elif (i==st[0] and np.allclose(o+st[1],z,atol=1e-8)):
                        jmpl[p] = -db #negation of dumbbell just means flipping the orientation
                        jmpl[p+2] = int(jmpl[p+2]*-1)
                        break
            jmp_new = jump(*jmpl) #argument unpack and create new jump object
    return jmp_new

    r2=cutoff*cutoff
    nmax = [int(np.round(np.sqrt(crys.metric[i, i]))) + 1 for i in range(3)]
    supervect = [np.array([n0, n1, n2])
                     for n0 in range(-nmax[0], nmax[0] + 1)
                     for n1 in range(-nmax[1], nmax[1] + 1)
                     for n2 in range(-nmax[2], nmax[2] + 1)]
    #Ask Dallas regarding the origin of this
    #First produce pure->pure omega_0 and omega1
    #Initial state at the origin unit cell
    lis0=[]
    for i,o1 in states_pure:
        for j,o2 in states_pure:
            for R in supervect:
                #catch the diagonal case
                if i==j and np.allclose(o1,o2) and np.allclose(R,z,atol=1e-8):
                    continue
                du = crys.basis[chem][j]-crys.basis[chem][i]
                dR = R
                dx = crys.unit2cart(dR,du)
                if np.dot(dx,dx) > r2:
                    continue
                db1 = dumbell(i,o1,z)
                db2 = dumbell(j,o2,R)
                for c1 in [-1,1]:
                    for c2 in [-1,1]:
                        jmp=jump(db1,db2,c1,c2)
                        #see if the jump leads to atoms coming too close to each other
                        if collsion_self(crys,chem,jmp,cutoff12,cutoff13) or collision_others(crys,chem,jmp,cutoff):
                            continue
                        #if passed, apply group operations
                        if not inlist(jmp,lis0):
                            trans=[]
                            trans.append(jmp)
                            for g in crys.G:
                                jmp_new=makevalid(jmp.gop(crys,chem,g),1,1)
                                if jmp_new != None:
                                    if not any(j==jmp_new for j in trans):
                                        trans.append(jmp_new)
                                        trans.append(-jmp_new)
                            lis0.append(trans)

    #Next, we construct dissociative jumps
    #mixed dumbbell to pure dumbbell
    for i,o1 in states_mixed:
        for j,o2 in states_pure:
            for R in supervect:
                #catch transitions within same site
                if (i==j and np.allclose(R,z)):
                    continue
                du = crys.basis[chem][j]-crys.basis[chem][i]
                dR = R
                dx = crys.unit2cart(dR,du)
                if np.dot(dx,dx) > r2:
                    continue
                db1 = dumbell(i,o1,z)
                db2 = dumbell(j,o2,R)
