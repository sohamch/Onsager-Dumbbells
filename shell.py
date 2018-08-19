import numpy as np
import onsager.crystal as crystal

def buildlist(crys,chem,j,nnrange):
    R=np.zeros(3)
    u=crys.basis[chem][j]
    #build upto fourth nearest neighbor
    n=nnrange+1
    Rvects = [np.array([x,y,z]) for x in range(-n,n) for y in range(-n,n) for z in range(-n,n)]
    Rvects.sort(key = lambda x:np.linalg.norm(np.dot(crys.lattice,x)))
    nndists=[]
    for R in Rvects:
        for idx,uvec in enumerate(crys.basis[chem]):
            if np.allclose(R,0,atol=crys.threshold) and idx==j:
                continue
            dx = crys.unit2cart(R,uvec) - crys.unit2cart(np.zeros(3),u)
            nndists.append(np.linalg.norm(dx))
    nndists=np.sort(list(set(nndists)))[:nnrange]#store upto fourth nearest neighbor
    nnlist=[]
    for dist in nndists:
        nlist=[]
        for idx,uvec in enumerate(crys.basis[chem]):
            for R in Rvects:
                if np.allclose(R,0,atol=crys.threshold) and idx==j:
                    continue
                dx = crys.unit2cart(R,uvec) - crys.unit2cart(np.zeros(3),u)
                if np.allclose(np.linalg.norm(dx),dist,atol=1e-4):
                    nlist.append(tuple([R,idx,uvec]))
        nnlist.append(nlist)
    return nnlist

def buildshell(crys,chem,i,shell):
    """
    function to build thermodynamic shell upto a given range
    Params:
        -> crys,chem - working crystal and sublattice
        -> i - site index in the sublattice around which to build the shell
        -> shell - nearest neighbor range - (1,1) for 1nn, (1,2) for 1nn^2 and so on
    """
    def flat(lis):
        y=[]
        for l in lis:
            for t in l:
                y.append(t)
        return y
    #build the first nearest neighbor shell
    site_list = flat(buildlist(crys,chem,i,shell[0]))
    m=0#starting index for next run, so that sites that have been visited before are not visited again.
    n = len(site_list) #last index, corresponding the site upto for which the next layer is to be built.
    for j in range (shell[1]-1):#how many times to go on
        for site in site_list[m:n]:#Go through existing sites
            R = site[0] #lattice vector with which to translate the generated sites.
            nn_list = flat(buildlist(crys,chem,site[1],shell[0]))
            for tup in nn_list:
                tup_new = (tup[0]+R,tup[1])
                if not any(np.allclose(tup_new[0],t[0],atol=crys.threshold) and t[1]==tup_new[1] for t in site_list):
                    if not (np.allclose(tup_new[0],0,atol=crys.threshold) and tup_new[1]==i):
                        site_list.append(tup_new)
        m = n
        n = len(site_list)
    return site_list
