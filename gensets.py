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
        return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for x in lis) #for x in l)

    def negOrInList(o,lis):
        z=np.zeros(3)
        return any(np.allclose(o+tup[1],z,atol=1e-8) for tup in lis)

    sitelist = crys.sitelist(chem)
    #Get the Wyckoff sets
    pairlist=[]
    for i,wycksites in enumerate(sitelist):
        orlist = family[i]
        symmlist=[]
        site=wycksites[0]
        newlist=[]
        for o in orlist:
            for g in crys.G:
                R, (ch,i_new) = crys.g_pos(g,np.zeros(3),(chem,site))
                o_new = crys.g_direc(g,o)
                if not (inlist((i_new,o_new),newlist) or inlist((i_new,-o_new),newlist)):
                        if negOrInList(o_new,newlist):
                            o_new = -o_new
                        newlist.append((i_new,o_new))
        pairlist.append(newlist)
    return pairlist
