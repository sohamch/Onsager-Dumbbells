import numpy as np
import onsager.crystal as crystal
from representations import *
from states import *
from collision import *
import time


def flat(lis):
    a=[x for l in lis for x in l]
    return a

def inset(j,s):
    return hash(j) in s

def inlist(j,l):
    return any(hash(j)==hash(j1) for j1 in l)

def disp(crys,chem,obj1,obj2):
    (i1,i2) = (obj1.i,obj2.i) if isinstance(obj1,dumbbell) else (obj1.db.i,obj2.db.i)
    (R1,R2) = (obj1.R,obj2.R) if isinstance(obj1,dumbbell) else (obj1.db.R,obj2.db.R)
    return crys.unit2cart(R2,crys.basis[chem][i2]) - crys.unit2cart(R1,crys.basis[chem][i1])
    # if np.dot(dR,dR) > cutoff**2:
    #     return True

def purejumps(dbobj,cutoff,solv_solv_cut,closestdistance):
    """
    Makes a jumpnetwork of pure dumbbells within a given distance to be used for omega_0
    and to create the solute-dumbbell stars.
    Parameters:
        dbobj - dbStates object. Contains the following info
            -crys,chem - working crystal object and sublattice respectively.
            -iorset - allowed orientations in the given sublattice (negatives excluded).
        cutoff - maximum jump distance
        solv_solv_cut - minimum allowable distance between two solvent atoms - to check for collisions
        closestdistance - minimum allowable distance to check for collisions with other atoms. Can be a single
        number or a list (corresponding to each sublattice)
    Returns:
        jumpnetwork - the symmetrically grouped jumpnetworks (pair1,pair2,c1,c2)
        jumpindices - the jumpnetworks with dbs in pair1 and pair2 indexed to iorset -> (i,j,dx,c1,c2)
    """
    #pointers to necessary parameters from dbobj
    if not isinstance(dbobj,dbStates):
        raise TypeError("the first parameter must be a dbStates object")

    crys,chem,iorset = dbobj.crys,dbobj.chem,dbobj.iorlist

    def indexed(j):
        """
        Takes in a jump and indexes it to the iorset
        Params:
            j - the jump to index
        Return:
            indexj - (i,j,dx,c1,c2)
        """
        (i1,o1) = (j.state1.i,j.state1.o)
        (i2,o2) = (j.state2.i,j.state2.o)
        initindex=None
        finindex=None
        for ind,(i,o) in enumerate(iorset):
            if (i1==i and np.allclose(o,o1,atol=1e-8)):
                initindex=ind
                break
        for ind,(i,o) in enumerate(iorset):
            if (i2==i and np.allclose(o,o2,atol=1e-8)):
                finindex=ind
                break
        if initindex==None or finindex==None:
            raise RuntimeError("The given initial or final dumbbell state in the jump is not in the (i,or) list provided")
        dx = disp(crys,chem,j.state1,j.state2)
        tup = (initindex,finindex,dx,j.c1,j.c2)
        return tup

    nmax = [int(np.round(np.sqrt(cutoff**2/crys.metric[i,i]))) + 1 for i in range(3)]
    Rvects = [np.array([n0,n1,n2]) for n0 in range(-nmax[0],nmax[0]+1)
                                 for n1 in range(-nmax[1],nmax[1]+1)
                                 for n2 in range(-nmax[2],nmax[2]+1)]
    jumplist=[]
    jumpindices=[]
    hashset=set([])
    count=0
    z=np.zeros(3).astype(int)
    for R in Rvects:
        for i in iorset:
            for f in iorset:
                db1 = dumbbell(i[0],i[1],np.array([0,0,0]))
                db2 = dumbbell(f[0],f[1],R)
                if db1==db2:#catch the diagonal case
                    continue
                dx = disp(crys,chem,db1,db2)
                if np.dot(dx,dx)>cutoff*cutoff:
                    continue
                for c1 in[-1,1]:
                    rotcheck = i[0]==f[0] and np.allclose(R,z,atol=crys.threshold)
                    if rotcheck:
                        j = jump(db1,db2,c1,1)

                        if inset(j,hashset): #no point doing anything else if the jump has already been considered
                            continue
                        if not (collision_self(crys,chem,j,solv_solv_cut,solv_solv_cut) or collision_others(crys,chem,j,closestdistance)):
                            #If the jump has not already been considered, check if it leads to collisions.
                            jlist=[]
                            jindlist=[]
                            for g in crys.G:
                                # jnew = j.gop(crys,chem,g)
                                db1new = dbobj.gdumb(g,db1)
                                db2new = dbobj.gdumb(g,db2)
                                jnew = jump(db1new[0],db2new[0],c1*db1new[1],1*db2new[1])
                                if not inset(jnew,hashset):
                                    #create the negative jump
                                    #not the __neg__ in jump because the initial state must be at the origin.
                                    db1newneg = dumbbell(db2new[0].i,db2new[0].o,np.array([0,0,0]))
                                    db2newneg = dumbbell(db1new[0].i,db1new[0].o,-db2new[0].R)
                                    jnewneg = jump(db1newneg,db2newneg,jnew.c2,jnew.c1)
                                    #add both the jump and it's negative
                                    jlist.append(jnew)
                                    jlist.append(jnewneg)
                                    jindlist.append(indexed(jnew))
                                    jindlist.append(indexed(jnewneg))
                                    hashset.add(hash(jnew))
                                    hashset.add(hash(jnewneg))
                            jumplist.append(jlist)
                            jumpindices.append(jindlist)
                            continue
                    for c2 in [-1,1]:
                        j = jump(db1,db2,c1,c2)
                        start = time.time()
                        cond=inset(j,hashset)
                        # tcheck.append(time.time()-start)
                        # tlen.append(len(hashset))
                        if cond: #no point doing anything else if the jump has already been considered
                            continue
                        if not (collision_self(crys,chem,j,solv_solv_cut,solv_solv_cut) or collision_others(crys,chem,j,closestdistance)):
                            #If the jump has not already been considered, check if it leads to collisions.
                            jlist=[]
                            jindlist=[]
                            for g in crys.G:
                                # jnew = j.gop(crys,chem,g)
                                db1new = dbobj.gdumb(g,db1)
                                db2new = dbobj.gdumb(g,db2)
                                jnew = jump(db1new[0],db2new[0],c1*db1new[1],c2*db2new[1])
                                if not inset(jnew,hashset):
                                    #create the negative jump
                                    #not exactly the __neg__ in jump because the initial state must be at the origin.
                                    db1newneg = dumbbell(db2new[0].i,db2new[0].o,np.array([0,0,0]))
                                    db2newneg = dumbbell(db1new[0].i,db1new[0].o,-db2new[0].R)
                                    jnewneg = jump(db1newneg,db2newneg,jnew.c2,jnew.c1)
                                    #add both the jump and it's negative
                                    jlist.append(jnew)
                                    jlist.append(jnewneg)
                                    jindlist.append(indexed(jnew))
                                    jindlist.append(indexed(jnewneg))
                                    hashset.add(hash(jnew))
                                    hashset.add(hash(jnewneg))
                            jumplist.append(jlist)
                            jumpindices.append(jindlist)
    return jumplist,jumpindices

def mixedjumps(mdbobj,cutoff,solt_solv_cut,closestdistance):
    """
    Makes a jumpnetwork of pure dumbbells within a given distance to be used for omega_0
    and to create the solute-dumbbell stars.
    Parameters:
        crys,chem - working crystal object and sublattice respectively.
        iorset - allowed (site,orientation) pairs in the given sublattice for mixed dumbbells, must include negatives
        cutoff - maximum jump distance
        solt_solv_cut - minimum allowable distance between solute and solvent atoms - to check for collisions
        closestdistance - minimum allowable distance to check for collisions with other atoms. Can be a single
        number or a list (corresponding to each sublattice)
    """
    crys,chem,mset = mdbobj.crys,mdbobj.chem,mdbobj.iorlist

    def indexed(j):
        """
        Takes in a jump and indexes it to the iorset
        Params:
            j - the jump to index
        Return:
            indexj - (i,j,dx,c1,c2)
        """
        (i1,o1) = (j.state1.db.i,j.state1.db.o)
        (i2,o2) = (j.state2.db.i,j.state2.db.o)
        initindex=None
        finindex=None
        for ind,(i,o) in enumerate(mset):
            if (i1==i and np.allclose(o,o1,atol=1e-8)):
                initindex=ind
                break
        for ind,(i,o) in enumerate(mset):
            if (i2==i and np.allclose(o,o2,atol=1e-8)):
                finindex=ind
                break
        if initindex==None or finindex==None:
            raise RuntimeError("The given initial or final dumbbell state in the jump is not in the (i,or) list provided")
        dx = disp(crys,chem,j.state1,j.state2)
        tup = (initindex,finindex,dx,j.c1,j.c2)
        return tup

    nmax = [int(np.round(np.sqrt(cutoff**2/crys.metric[i,i]))) + 1 for i in range(3)]
    Rvects = [np.array([n0,n1,n2]) for n0 in range(-nmax[0],nmax[0]+1)
                                 for n1 in range(-nmax[1],nmax[1]+1)
                                 for n2 in range(-nmax[2],nmax[2]+1)]
    jumplist=[]
    jumpindices=[]
    hashset=set([])

    for R in Rvects:
        for i in mset:
            for f in mset:
                db1 = dumbbell(i[0],i[1],np.array([0,0,0]))
                p1 = SdPair(i[0],np.array([0,0,0]),db1)
                db2=dumbbell(f[0],f[1],R)
                p2 = SdPair(f[0],R,db2)
                if p1 == p2:
                    continue
                dx=disp(crys,chem,db1,db2)
                if np.dot(dx,dx)>cutoff**2:
                    continue
                j = jump(p1,p2,1,1)#since only solute moves, both indicators are +1
                if inset(j,hashset):
                    continue
                if not (collision_self(crys,chem,j,solt_solv_cut,solt_solv_cut) or collision_others(crys,chem,j,closestdistance)):
                    jlist=[]
                    jindlist=[]
                    for g in crys.G:
                        jnew = j.gop(crys,chem,g)
                        if not mdbobj.checkinlist(jnew.state1) or not mdbobj.checkinlist(jnew.state2):
                            raise RuntimeError("Unexpected mixed dumbbell with (i,o) not in given set")
                        if not inset(jnew,hashset):
                            #create the negative jump
                            p11 = jnew.state1
                            p21 = jnew.state2
                            p1neg = SdPair(p21.i_s,np.array([0,0,0]),dumbbell(p21.db.i,p21.db.o,np.array([0,0,0])))
                            p2neg = SdPair(p11.i_s,-p21.db.R,dumbbell(p11.db.i,p11.db.o,-p21.db.R))
                            jnewneg = jump(p1neg,p2neg,1,1)
                            #add both the jump and its negative
                            jlist.append(jnew)
                            jlist.append(jnewneg)
                            jindlist.append(indexed(jnew))
                            jindlist.append(indexed(jnewneg))
                            hashset.add(hash(jnew))
                            hashset.add(hash(jnewneg))
                    jumplist.append(jlist)
                    jumpindices.append(jindlist)

    return jumplist,jumpindices
