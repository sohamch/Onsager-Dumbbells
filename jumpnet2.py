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
            om = m_or_fam[i][j]
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

# def gensets(crys,chem,purelist,mixedlist):
#     wyck_sites = crys.sitelist(chem)
#     n = len(wyck_sites)
#     for i in range(n):
#         plist = purelist[i]
#         atomlist=wyck_sites[i]
#         for j in range(len(atomlist)):
#             for op in plist:
#                 site_or_pair = tuple([atomlist[j],op])
