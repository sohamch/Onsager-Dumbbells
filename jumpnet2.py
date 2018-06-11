# coding: utf-8

# # Function to Generate jumpnetworks for solute-dumbbell pairs
import numpy as np
import numpy.linalg as la
import onsager.crystal as crystal
from collections import namedtuple
from representations import *

# First, need to create all possible orientation-site pairs
def gensets(crys,chem,or_pure_list,or_mixed_list):
    """
        Function to generate symmetry related orientations for a given crystal and orientation family.
        Parameters:
            crys - crystal object for which orientation sets are to be found
            or_pure - family of pure dumbbell orientation directions
            or_mixed - family of directions for mixed dumbbells.
    """
    zero=np.zeros(3)
    if not (isinstance(or_pure,np.ndarray) and isinstance(or_mixed,np.ndarray):
        raise TypeError("Orientations must be entered in cartesian form")
    if not (len(or_pure_list)==crys.basis[chem] and len(or_mixed_list)==crys.basis[chem]):
        raise TypeError("Need as many orientations as atoms in the sublattice")
    or_pure_expand=[]
    or_mixed_expand=[]
    for i in range(len(or_pure_list)):
        opure=or_pure_list[i]
        omix=or_mixed_list[i]
        plist=[]
        mlist=[]
        plist.append(opure)
        mlist.append(omix)
        for g in crys.G:
            or_new=np.dot(g.cartrot,opure)
            if not any(np.allclose(or_new,or1) for or1 in plist):
                if not any(np.allclose(or_new+or_1,zero,atol=1e-8) for or_1 in plist):
                    plist.append(or_new)
            or_new=np.dot(g.cartrot,omix)
            if not any(np.allclose(or_new,or1) for or1 in mlist):
                mlist.append(or_new)
        or_mixed_expand.append(plist)
        or_pure_expand.append(plist)
    #now build the pairset
    purepairs=[]
    mixedpairs=[]
    n=len(self.basis[chem])
    for i in range(n):
        lstp=[]
        lstm=[]
        for j in range(len(or_pure_expand[i])):
            lstp.append(tuple(i,or_pure_expand[i][j]))
        purepairs.append(lst)
        for j in range(len(or_mixed_expand[i])):
            lstm.append(tuple(i,or_mixed_expand[i][j]))
        mixedpairs.append(lstm)
    return [purepairs,mixedpairs]

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
    """
    Dstate=gensets
