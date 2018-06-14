# coding: utf-8

# # Function to Generate jumpnetworks for solute-dumbbell pairs
import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple
from representations import *

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

def jumpnetwork(crys,chem,pairs_pure,pairs_mixed):
    def check_parity(db,c):
        #checks whether orientations have been reversed for pure dumbbells
        #if it has been reversed, returns the correct dumbbell otherwise the same one
        z = np.zeros(3)
        state = tuple([db.i,db.o])
        for i in purelist:
            if (state[0]==i[0] and np.allclose(state[1]+i[1],z,atol=1e-8)):
                return dumbbell(i[0],i[1],db.R),-c
        return db,c
