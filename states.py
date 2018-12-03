import numpy as np
import onsager.crystal as crystal
from representations import *
from collision import *

#Helper functions for jumpnetworks
def inset(j,s):
    return j in s

# def inlist(j,l):
#     return any(hash(j)==hash(j1) for j1 in l)

def disp(crys,chem,obj1,obj2):
    (i1,i2) = (obj1.i,obj2.i) if isinstance(obj1,dumbbell) else (obj1.db.i,obj2.db.i)
    (R1,R2) = (obj1.R,obj2.R) if isinstance(obj1,dumbbell) else (obj1.db.R,obj2.db.R)
    return crys.unit2cart(R2,crys.basis[chem][i2]) - crys.unit2cart(R1,crys.basis[chem][i1])

#Create pure dumbbell states
class dbStates(object):
    """
    Class to generate all possible dumbbell configurations for given basis sites
    This is mainly to automate group operations on jumps (to return correct dumbbell states)
    Functionalities - 1. take in a list of (i,or) pairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an (i,or) pair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,family):
        if not isinstance(family,list):
            raise TypeError("Enter the families as a list of lists")

        self.crys = crys
        self.chem = chem
        self.family=family
        self.iorlist = self.genpuresets()
        self.symorlist = self.gensymset()
        #Store both iorlist and symorlist so that we can compare them later if needed.
        self.threshold = crys.threshold
        self.invmap = self.invmap(self.iorlist,self.symorlist)
        self.indexmap = self.indexmapping()
        self.indsymlist = self.indexedsymlist()

    @staticmethod
    def invmap(iorlist,symorlist):
        for l in symorlist:
            for tup in l:
                if not any(t[0]==tup[0] and np.allclose(t[1],tup[1]) for t in iorlist):
                    raise TypeError("iorlist and symorlist have different states")
        invmap = np.zeros(len(iorlist),dtype=int)
        for ind,l in enumerate(symorlist):
            for i,tup in enumerate(iorlist):
                if any(tup[0]==t[0] and np.allclose(tup[1],t[1]) for t in l):
                    invmap[i]=ind
        return invmap

    def indexedsymlist(self):
        """
        return a version of symorlist indexed to iorlist
        """
        stind = []
        for l in self.symorlist:
            lis=[]
            for tup1 in l:
                for ind,tup in enumerate(self.iorlist):
                    if tup[0]==tup1[0] and np.allclose(tup[1],tup1[1]):
                        lis.append(ind)
                        break
            stind.append(lis)
        return stind

    def gdumb(self,g,db):
        """
        Takes as an argument a dumbbell and return the result of a group operation on that dumbbell
        param: g - group operation
               db - dumbbell object to operate upon
        returns - (db_new,p) - the dumbbell produced by the symmetry operation and the parity value depending on existence
                  within symorlist.
                  Example - 1. If the new orientation produced is [-100] but the one present in symorlist is [100], then returns
                    ("db object with orientation [100]", -1)
                            2. If the new orientation is [100] instead, returns ("db object with orientation [100]", 1)
        """

        def inlist(tup):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=self.threshold) for x in self.iorlist)

        db_new = db.gop(self.crys,self.chem,g)
        i = db_new.i
        o = db_new.o
        tup = None
        if inlist((i,o)):
            tup = (db_new,1)
        elif inlist((i,-o)):
            # db_new = dumbbell(db_new.i,-db_new.o,db_new.R)
            tup = (-db_new,-1)
        if tup == None:
            #This will be used only during the testing phase, can remove it later when not needed.
            #Ideally, if the production of symorlist is correct, then it should catch either of the
            #above two cases.
            raise RuntimeError("The group operation does not produce an (i,or) pair in the given (i,or) list")
        return tup

    def genpuresets(self):
        """
        generates complete (i,or) set from given family of orientations, neglects negatives, since pure
        """
        if not isinstance(self.family,list):
            raise TypeError("Enter the families as a list of lists")
        for i in self.family:
            if not isinstance(i,list):
                raise TypeError("Enter the families for each site as a list of np arrays")
            for j in i:
                if not isinstance(j,np.ndarray):
                    raise TypeError("Enter individual orientation families as numpy arrays")

        def inlist(tup,lis):
            return any(tup[0]==x[0] and np.allclose(tup[1],x[1],atol=1e-8) for x in lis)

        def negOrInList(o,lis):
            return any(np.allclose(o+tup[1],0,atol=1e-8) for tup in lis)

        sitelist = self.crys.sitelist(self.chem)
        #Get the Wyckoff sets
        pairlist=[]
        for i,wycksites in enumerate(sitelist):
            orlist = self.family[i]
            site=wycksites[0]
            newlist=[]
            for o in orlist:
                for g in self.crys.G:
                    R, (ch,i_new) = self.crys.g_pos(g,np.zeros(3),(self.chem,site))
                    o_new = self.crys.g_direc(g,o)
                    if not (inlist((i_new,o_new),pairlist) or inlist((i_new,-o_new),pairlist)):
                            if negOrInList(o_new,pairlist):
                                o_new = -o_new
                            pairlist.append((i_new,o_new))
        return pairlist

    def gensymset(self):#crys,chem,iorlist
        """
        Takes in a flat list of (i,or) pairs and groups them according to symmetry
        params:
            crys - the working crystal object
            chem - the sublattice under consideration
            iorlist - flat list of (i,or) pairs
        Returns:
            symorlist - a list of lists which contain symmetry related (i,or) pairs
        """

        #some helper functions
        def matchvec(vec1,vec2):
            return np.allclose(vec1,vec2,atol=self.crys.threshold) or np.allclose(vec1+vec2,0,atol=self.crys.threshold)

        def insymlist(ior,symlist):
            return any(ior[0]==x[0] and matchvec(ior[1],x[1]) for lis in symlist for x in lis)

        def inset(ior,set):
            return any(ior[0]==x[0] and matchvec(ior[1],x[1]) for x in set)

        newlist=[]
        symlist=[]
        for ior in self.iorlist:
            if insymlist(ior,symlist):
                continue
            newlist=[]
            newlist.append(ior)
            for g in self.crys.G:
                R, (ch,inew) = self.crys.g_pos(g,np.array([0,0,0]),(self.chem,ior[0]))
                onew  = np.dot(g.cartrot,ior[1])
                if any(np.allclose(onew+t[1],0,atol=1.e-8) for t in self.iorlist):
                    onew = -onew
                if not inset((inew,onew),newlist):
                    newlist.append((inew,onew))
            symlist.append(newlist)
        return symlist

    def indexmapping(self):
        ng = len(self.crys.G)
        indexmap = np.zeros((ng,len(self.iorlist)),dtype=int)
        for i,g in enumerate(self.crys.G):
            for st,ior in enumerate(self.iorlist):
                R, (ch,inew) = self.crys.g_pos(g,np.array([0,0,0]),(self.chem,ior[0]))
                onew  = np.dot(g.cartrot,ior[1])
                if any(np.allclose(onew+t[1],0,atol=1.e-8) for t in self.iorlist):
                    onew = -onew
                for j,t in enumerate(self.iorlist):
                    if(t[0]==inew and np.allclose(t[1],onew)):
                        indexmap[i][st]=j
        return indexmap

    def jumpnetwork(self,cutoff,solv_solv_cut,closestdistance):
        """
        Makes a jumpnetwork of pure dumbbells within a given distance to be used for omega_0
        and to create the solute-dumbbell stars.
        Parameters:
            cutoff - maximum jump distance
            solv_solv_cut - minimum allowable distance between two solvent atoms - to check for collisions
            closestdistance - minimum allowable distance to check for collisions with other atoms. Can be a single
            number or a list (corresponding to each sublattice)
        Returns:
            jumpnetwork - the symmetrically grouped jumpnetworks (pair1,pair2,c1,c2)
            jumpindices - the jumpnetworks with dbs in pair1 and pair2 indexed to iorset -> (i,j,dx,c1,c2)
        """
        #pointers to necessary parameters from dbobj
        crys,chem,iorset = self.crys,self.chem,self.iorlist
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

                            if j in hashset: #no point doing anything else if the jump has already been considered
                                continue
                            if not (collision_self(crys,chem,j,solv_solv_cut,solv_solv_cut) or collision_others(crys,chem,j,closestdistance)):
                                #If the jump has not already been considered, check if it leads to collisions.
                                jlist=[]
                                jindlist=[]
                                for g in crys.G:
                                    # jnew = j.gop(crys,chem,g)
                                    db1new = self.gdumb(g,db1)
                                    db2new = self.gdumb(g,db2)
                                    jnew = jump(db1new[0],db2new[0],c1*db1new[1],1*db2new[1])
                                    if not jnew in hashset:
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
                            if inset(j,hashset): #no point doing anything else if the jump has already been considered
                                continue
                            if not (collision_self(crys,chem,j,solv_solv_cut,solv_solv_cut) or collision_others(crys,chem,j,closestdistance)):
                                #If the jump has not already been considered, check if it leads to collisions.
                                jlist=[]
                                jindlist=[]
                                for g in crys.G:
                                    # jnew = j.gop(crys,chem,g)
                                    db1new = self.gdumb(g,db1)
                                    db2new = self.gdumb(g,db2)
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

class mStates(object):
    """
    Class to generate all possible mixed dumbbell configurations for given basis sites
    This is mainly to automate group operations on jumps (to return correct dumbbell states)
    Functionalities - 1. take in a list of (i,or) pairs created using gensets and convert it
    to symmetry grouped pairs.
                      2. Given an (i,or) pair, check which pair in the set a group operation maps it onto.
    """
    def __init__(self,crys,chem,family):
        if not isinstance(family,list):
            raise TypeError("Enter the families as a list of lists")

        self.crys = crys
        self.chem = chem
        self.family = family
        self.iorlist = self.genmixedsets()
        self.symorlist = self.gensymset()
        #Store both iorlist and symorlist so that we can compare them later if needed.
        self.threshold = crys.threshold
        self.invmap = self.invmap(self.iorlist,self.symorlist)
        self.indexmap = self.indexmapping()

    #Is this function really necessary for dumbbells? This is not a static method so it does occupy space every time mdb is instatiated

    def checkinlist(self,mdb):
        """
        Takes as an argument a dumbbell and return the result of a group operation on that dumbbell
        param: g - group operation
               mdb - pair object to operate upon
        returns - mdb_new -> the result of the group operation
        """
        #Type check to see if a mixed dumbbell is passed
        if not isinstance(mdb,SdPair):
            raise TypeError("Mixed dumbbell must be an SdPair object.")
        if not(mdb.i_s==mdb.db.i and np.allclose(mdb.R_s,mdb.db.R,atol=self.threshold)):
            raise TypeError("Passed in pair is not a mixed dumbbell")

        i = mdb.db.i
        o = mdb.db.o
        #Place a check for test purposes
        return any(i==x[0] and np.allclose(o,x[1],atol=self.threshold) for x in self.iorlist)

    def indexedsymlist(self):
        """
        return a version of symorlist indexed to iorlist
        """
        stind = []
        for l in self.symorlist:
            lis=[]
            for tup1 in l:
                for ind,tup in enumerate(self.iorlist):
                    if tup[0]==tup1[0] and np.allclose(tup[1],tup1[1]):
                        lis.append(ind)
                        break
            stind.append(lis)
        return stind

    def genmixedsets(self):
        crys,chem,family = self.crys,self.chem,self.family
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

    def gensymset(self):
        """
        Takes in a flat list of (i,or) pairs and groups them according to symmetry
        params:
            crys - the working crystal object
            chem - the sublattice under consideration
            iorlist - flat list of (i,or) pairs
        Returns:
            symorlist - a list of lists which contain symmetry related (i,or) pairs
        """
        crys,chem,iorlist = self.crys,self.chem,self.iorlist
        def matchvec(vec1,vec2):
            return np.allclose(vec1,vec2,atol=crys.threshold)

        def insymlist(ior,symlist):
            return any(ior[0]==x[0] and matchvec(ior[1],x[1]) for lis in symlist for x in lis)

        def inset(ior,set):
            return any(ior[0]==x[0] and matchvec(ior[1],x[1]) for x in set)
        #first make a set of the unique pairs supplied - each taken only once
        #That way we won't need to do redundant group operations
        orset = []
        for ior in iorlist:
            if not inset(ior,orset):
                orset.append(ior)

        #Now check for valid group transitions within orset.
        #If the result of a group operation (j,o1) or it's negative (j,-o1) is not there is orset, append it to orset.
        ior = orset[0]
        newlist=[]
        symlist=[]
        newlist.append(ior)
        for g in crys.G:
            R, (ch,inew) = crys.g_pos(g,np.array([0,0,0]),(chem,ior[0]))
            onew  = np.dot(g.cartrot,ior[1])
            if not inset((inew,onew),newlist):
                newlist.append((inew,onew))
        symlist.append(newlist)
        for ior in orset[1:]:
            if insymlist(ior,symlist):
                continue
            newlist=[]
            newlist.append(ior)
            for g in crys.G:
                R, (ch,inew) = crys.g_pos(g,np.array([0,0,0]),(chem,ior[0]))
                onew  = np.dot(g.cartrot,ior[1])
                if not inset((inew,onew),newlist):
                    newlist.append((inew,onew))
            symlist.append(newlist)
        return symlist

    @staticmethod
    def invmap(iorlist,symorlist):
        for l in symorlist:
            for tup in l:
                if not any(t[0]==tup[0] and np.allclose(t[1],tup[1]) for t in iorlist):
                    raise TypeError("iorlist and symorlist have different states")
        invmap = np.zeros(len(iorlist),dtype=int)
        for ind,l in enumerate(symorlist):
            for i,tup in enumerate(iorlist):
                if any(tup[0]==t[0] and np.allclose(tup[1],t[1]) for t in l):
                    invmap[i]=ind
        return invmap

    def indexmapping(self):
        ng = len(self.crys.G)
        indexmap = np.zeros((ng,len(self.iorlist)),dtype=int)
        for i,g in enumerate(self.crys.G):
            for st,ior in enumerate(self.iorlist):
                R, (ch,inew) = self.crys.g_pos(g,np.array([0,0,0]),(self.chem,ior[0]))
                onew  = np.dot(g.cartrot,ior[1])
                for j,t in enumerate(self.iorlist):
                    if(t[0]==inew and np.allclose(t[1],onew)):
                        indexmap[i][st]=j
        return indexmap

    def jumpnetwork(self,cutoff,solt_solv_cut,closestdistance):
        """
        Makes a jumpnetwork of pure dumbbells within a given distance to be used for omega_0
        and to create the solute-dumbbell stars.
        Parameters:
            cutoff - maximum jump distance
            solt_solv_cut - minimum allowable distance between solute and solvent atoms - to check for collisions
            closestdistance - minimum allowable distance to check for collisions with other atoms. Can be a single
            number or a list (corresponding to each sublattice)
        """
        crys,chem,mset = self.crys,self.chem,self.iorlist

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
                            if not self.checkinlist(jnew.state1) or not self.checkinlist(jnew.state2):
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
