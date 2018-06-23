# coding: utf-8

# # Function to Generate jumpnetworks for solute-dumbbell pairs
import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple
from representations import *
from collision import *

#first generate orientation states


def jumpnetwork(crys,chem,pairs_pure,pairs_mixed,cutoff,cutoffsoltsolv,cutoffsolvsolv,closestDistance):
    z=np.zeros(3)
    states_pure = pairs_pure.copy()
    states_mixed = pairs_mixed.copy()
    db1 = jmp.state1 if isinstance(jmp.state1,dumbbell) else jmp.state1.db
    db2 = jmp.state2 if isinstance(jmp.state2,dumbbell) else jmp.state2.db
    c1 = jmp.c1
    c2=jmp.c2
    jmpl=list(jmp)#convert jump to a list to support item assignment

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
    # print (purelist,mixedlist)
    #first check if mixed list is sane
    if len(mixedlist)>0:
        for p,db,c in mixedlist:
            i=db.i
            o=db.o
            if not any(j==i and np.allclose(o,o1,atol=1e-8) for j,o1 in states_mixed[i]):
                return None
    #provided mixed list is sane, then move on to pure list
    if len(purelist)>0:
        for p,db,c in purelist:
            i=db.i
            o=db.o
            if not (any(j==i and np.allclose(o,o1,atol=1e-8) for j,o1 in states_pure[i]) or any(j==i and np.allclose(o+o1,z,atol=1e-8) for j,o1 in states_pure[i])):
                return None
            for st in states_pure[i]:
                if (i==st[0] and np.allclose(o+st[1],z,atol=1e-8)):
                    if isinstance(jmpl[p],dumbbell):
                        jmpl[p] = -db #negation of dumbbell just means flipping the orientation
                    else:
                        sdpairl = list(jmpl[p])
                        sdpairl[2] = -db
                        Sdp = SdPair(*sdpairl)
                        jmpl[p] = Sdp
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
                        if collsion_self(crys,chem,jmp,cutoffsolvsolv,cutoffsolvsolv) or collision_others(crys,chem,jmp,closestDistance):
                            continue
                        #if passed, apply group operations
                        if not inlist(jmp,lis0):
                            trans=[]
                            trans.append(jmp)
                            for g in crys.G:
                                jmp_new=makevalid(jmp.gop(crys,chem,g),1,1)
                                if jmp_new != None:
                                    if not any(j==jmp_new for j,d in trans):
                                        trans.append((jmp_new,dx))
                                        trans.append((-jmp_new,-dx))
                            lis0.append(trans)

    #Next, we construct dissociative and associative jumps
    #negative of associative(4) is dissociative(3) jump
    #we first construct a dissociative jump and then take its negative to form
    #associative jump.
    #Mixed state state at the origin
    lis4=[]
    lis3=[]
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
                state1 = SdPair(i,z,db1)
                state2 = SdPair(i,z,db2)
                for c2 in [-1,1]:
                    jmpd = jump(state1,state2,-1,c2)
                    if collsion_self(crys,chem,jmp,cutoffsoltsolv,cutoffsolvsolv) or collision_others(crys,chem,jmp,closestDistance):
                        continue
                    if not inlist(jmpd,lis3):
                        transd=[]
                        transa=[]
                        transd.append(jmpd)
                        transa.append(-jmpd)
                        for g in crys.G:
                            jmpd_new = makevalid(jmpd.gop(crys,chem,g),0,1)
                            if jmpd_new != None:
                                if not any(jd==jmpd_new for jd,d in trans):
                                    transd.append((jmpd_new,dx))
                                    transa.append((-jmpd_new,-dx))
                        lis3.append(transd)
                        lis4.append(transa)
    lis2 = []
    for i,o1 in states_mixed:
        for j,o2 in states_mixed:
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
                state1 = SdPair(i,z,db1)
                state2 = SdPair(j,R,db2)
                jmp = jump(state1,state2,1,1)
                if collsion_self(crys,chem,jmp,cutoffsoltsolv,cutoffsoltsolv) or collision_others(crys,chem,jmp,closestDistance):
                    continue
                if not inlist(jmp,lis2):
                    trans=[]
                    trans.append(jmp)
                    for g in crys.G:
                        jmp_new=makevalid(jmp.gop(crys,chem,g),1,1)
                        if jmp_new != None:
                            if not any(j==jmp_new for j,d in trans):
                                trans.append((jmp_new,dx))
                                trans.append((-jmp_new,-dx))
                    lis2.append(trans)

    return [lis0,lis2,lis3,lis4]
