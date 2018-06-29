import numpy as np
from representations import *

def genpuresets(crys,chem,family):
    if not isinstance(family,list):
        raise TypeError("Enter the families as a list of lists")
    for i in family:
        if not isinstance(i,list):
            raise TypeError("Enter the families for each site as a list of np arrays")
        for j in i:
            if not isinstance(j,np.ndarray):
                raise TypeError("Enter individual orientation families as numpy arrays")

    def inlist(tup,lis):
        return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for x in lis)

    def negOrInList(o,lis):
        return any(np.allclose(o+tup[1],0,atol=1e-8) for tup in lis)

    sitelist = crys.sitelist(chem)
    #Get the Wyckoff sets
    pairlist=[]
    for i,wycksites in enumerate(sitelist):
        orlist = family[i]
        site=wycksites[0]
        newlist=[]
        for o in orlist:
            for g in crys.G:
                R, (ch,i_new) = crys.g_pos(g,np.zeros(3),(chem,site))
                o_new = crys.g_direc(g,o)
                if not (inlist((i_new,o_new),pairlist) or inlist((i_new,-o_new),pairlist)):
                        if negOrInList(o_new,pairlist):
                            o_new = -o_new
                        pairlist.append((i_new,o_new))
    return pairlist

def genmixedsets(crys,chem,family):

    if not isinstance(family,list):
        raise TypeError("Enter the families as a list of lists")
    for i in family:
        if not isinstance(i,list):
            raise TypeError("Enter the families for each site as a list of numpy arrays")
        for j in i:
            if not isinstance(j,np.ndarray):
                raise TypeError("Enter individual orientation families as numpy arrays")

    def inlist(tup,lis):
        return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for x in lis)

    sitelist = crys.sitelist(chem)
    #Get the Wyckoff sets
    pairlist=[]
    for i,wycksites in enumerate(sitelist):
        orlist = family[i]
        site = wycksites[0]
        newlist=[]
        for o in orlist:
            for g in crys.G:
                R, (ch,i_new) = crys.g_pos(g,np.zeros(3),(chem,site))
                o_new = crys.g_direc(g,o)
                if not inlist((i_new,o_new),pairlist):
                    pairlist.append((i_new,o_new))
    return pairlist

def genPairSets(crys,chem,iorlist,thrange):

    def withinlist(db):
        "returns a dumbbell that is within the iorlist by negating a vector if it has been reversed."
        
        if any(db.i==j and np.allclose(db.o,o1,atol=crys.threshold) for j,o1 in iorlist):
            return db
        if any(db.i==j and np.allclose(db.o+o1,0,atol=crys.threshold) for j,o1 in iorlist):
            return -db

    def inlist(pair,lis):
        return any(pair==pair1 for pair1 in lis)

    #first create all the pairs within the thermodynamic range
    Rvects = [np.array([x,y,z]) for x in range(-thrange,thrange+1)
                      for y in range(-thrange,thrange+1)
                      for z in range(-thrange,thrange+1)]
    # print(Rvects)
    pairlist=[]
    z=np.zeros(3).astype(int)
    for i_s in range(len(crys.basis[chem])):
        for i,o in iorlist:
            for R in Rvects:
                if i==i_s and np.allclose(R,z,atol=crys.threshold):
                    continue
                db = dumbbell(i,o,R)
                pair = SdPair(i_s,z,db)
                if inlist(pair,pairlist):
                    continue
                # newlist=[]
                pairlist.append(pair)
                # print(pair)
                for g in crys.G:
                    newpair = pair.gop(crys,chem,g)
                    db = withinlist(newpair.db)
                    newpair=SdPair(newpair.i_s,newpair.R_s,db)
                    if not inlist(newpair,pairlist):
                        pairlist.append(newpair)
                # print()
    return pairlist
