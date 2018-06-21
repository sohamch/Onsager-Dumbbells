import numpy as np
from representations import *

def genpuresets(crys,chem,family):
    sitelist = crys.sitelist(chem)
    #Get the Wyckoff sets
    pairlist=[]
    for i,wycksites in enumerate(sitelist):
        orlist = family[i]
        symmlist=[]
        for site in wycksites:
            newlist=[]
            for o in orlist:
                for g in crys.G:
                    i_new = crys.g_pos(g,np.zeros(3),(chem,site))[1][1]
                    o_new = crys.g_direc(o)
                    if not (inlist((i_new,o_new),symmlist) or inlist((i_new,-o_new),symmlist)):
                            newlist.append((i_new,o_new))
            if len(newlist)>0:
                symmlist.append(newlist)
        pairlist.append(symmlist)
