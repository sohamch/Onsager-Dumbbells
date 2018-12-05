import numpy as np
import onsager.crystal as crystal
from states import *
from representations import *
from test_structs import *
# from gensets import *
import unittest

class test_statemaking(unittest.TestCase):
    def setUp(self):
        famp0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
        famp12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
        self.family = [famp0,famp12]
        self.crys = tet2
        # self.pairs_pure = genpuresets(tet2,0,family)
        # self.pairs_mixed = genmixedsets(tet2,0,family)
        #generating pairs from families is now taken care of within states

    def test_dbStates(self):
        #check that symmetry analysis is correct
        dbstates=dbStates(self.crys,0,self.family)
        self.assertEqual(len(dbstates.symorlist),4)
        #check that every (i,or) set is accounted for
        sm=0
        for i in dbstates.symorlist:
            sm += len(i)
        self.assertEqual(sm,len(dbstates.iorlist))

        #test group operations
        # db=dumbbell(1, np.array([1.,1.,0.]), np.array([1,1,0]))
        # Glist = list(dbstates.crys.G)
        # x = np.random.randint(0,len(Glist))
        # g = Glist[x] #select random groupop
        # newdb_test = db.gop(self.crys,0,g)
        # newdb, p = dbstates.gdumb(g,db)
        # count=0
        # if(newdb_test==newdb):
        #     self.assertEqual(p,1)
        #     count=1
        # elif(newdb_test==-newdb):
        #     self.assertEqual(p,-1)
        #     count=1
        # self.assertEqual(count,1)

        #test indexmapping
        Glist = list(dbstates.crys.G)
        x = np.random.randint(0,len(Glist))
        g = Glist[x] #select random groupop
        for stateind,tup in enumerate(dbstates.iorlist):
            i,o = tup[0],tup[1]
            R, (ch,inew) = dbstates.crys.g_pos(g,np.array([0,0,0]),(dbstates.chem,i))
            onew  = np.dot(g.cartrot,o)
            if any(np.allclose(onew+t[1],0,atol=1.e-8) for t in dbstates.iorlist):
                onew = -onew
            count=0
            for j,t in enumerate(dbstates.iorlist):
                if(t[0]==inew and np.allclose(t[1],onew)):
                    foundindex=j
                    count+=1
            self.assertEqual(count,1)
            self.assertEqual(foundindex,dbstates.indexmap[x][stateind])

        #test_indexedsymlist
        i1=np.random.randint(0,len(dbstates.indsymlist))
        l = dbstates.indsymlist[i1]
        i2=np.random.randint(0,len(l))
        tupFromSymor = dbstates.symorlist[i1][i2]
        tupFromIorlist = dbstates.iorlist[dbstates.indsymlist[i1][i2]]
        self.assertTrue(tupFromSymor[0]==tupFromIorlist[0] and np.allclose(tupFromSymor[1],tupFromIorlist[1]))

    #Test jumpnetwork
    def test_purejumps(self):
        #cube
        famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
        family = [famp0]
        pdbcontainer_cube = dbStates(cube,0,family)
        jset_cube,jind_cube = pdbcontainer_cube.jumpnetwork(0.3,0.01,0.01)
        test_dbi = dumbbell(0, np.array([0.126,0.,0.]),np.array([0,0,0]))
        test_dbf = dumbbell(0, np.array([0.126,0.,0.]),np.array([0,1,0]))
        count=0
        indices=[]
        for i,jlist in enumerate(jset_cube):
            for q,j in enumerate(jlist):
                if j.state1 == test_dbi or j.state1 == -test_dbi:
                    if j.state2 == test_dbf or j.state2 == -test_dbf:
                        if j.c1 == j.c2 == -1:
                           count += 1
                           indices.append((i,q))
                           jtest = jlist
        # print (indices)
        self.assertEqual(count,1) #see that this jump has been taken only once into account
        self.assertEqual(len(jtest),24)

        #Next FCC
        #test this out with FCC
        fcc = crystal.Crystal.FCC(0.55)
        famp0 = [np.array([1.,1.,0.])/np.sqrt(2)*0.2]
        family = [famp0]
        pdbcontainer_fcc = dbStates(fcc,0,family)
        jset_fcc,jind_fcc = pdbcontainer_fcc.jumpnetwork(0.4,0.01,0.01)
        o1 = np.array([1.,-1.,0.])*0.2/np.sqrt(2)
        if any(np.allclose(-o1,o) for i,o in pdbcontainer_fcc.iorlist):
            o1 = -o1.copy()
        db1 = dumbbell(0,o1,np.array([0,0,0]))
        db2 = dumbbell(0,o1,np.array([0,0,1]))
        jmp = jump(db1,db2,1,1)
        jtest=[]
        for jl in jset_fcc:
            for j in jl:
                if j==jmp:
                    jtest.append(jl)
        #see that the jump has been accounted just once.
        self.assertEqual(len(jtest),1)
        #See that there 24 jumps. 24 0->0.
        self.assertEqual(len(jtest[0]),24)

        #DC_Si - same symmetry as FCC, except twice the number of jumps, since we have two basis
        #atoms belonging to the same Wyckoff site, in a crystal with the same lattice vectors.
        latt = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])*0.55
        DC_Si = crystal.Crystal(latt,[[np.array([0.,0.,0.]),np.array([0.25,0.25,0.25])]],["Si"])
        famp0 = [np.array([1.,1.,0.])/np.sqrt(2)*0.2]
        family = [famp0]
        pdbcontainer_si = dbStates(DC_Si,0,family)
        jset_si,jind_si = pdbcontainer_si.jumpnetwork(0.4,0.01,0.01)
        o1 = np.array([1.,-1.,0.])*0.2/np.sqrt(2)
        if any(np.allclose(-o1,o) for i,o in pdbcontainer_si.iorlist):
            o1 = -o1.copy()
        db1 = dumbbell(0,o1,np.array([0,0,0]))
        db2 = dumbbell(0,o1,np.array([0,0,1]))
        jmp = jump(db1,db2,1,1)
        jtest=[]
        for jl in jset_si:
            for j in jl:
                if j==jmp:
                    jtest.append(jl)
        #see that the jump has been accounted just once.
        self.assertEqual(len(jtest),1)
        #See that there 48 jumps. 24 0->0 and 24 1->1.
        self.assertEqual(len(jtest[0]),48)

        #HCP
        Mg = crystal.Crystal.HCP(0.3294,chemistry=["Mg"])
        famp0 = [np.array([1.,0.,0.])*0.145]
        family = [famp0]
        pdbcontainer_hcp = dbStates(Mg,0,family)
        jset_hcp,jind_hcp = pdbcontainer_hcp.jumpnetwork(0.45,0.01,0.01)
        o = np.array([0.145,0.,0.])
        if any(np.allclose(o,o1)for i,o1 in pdbcontainer_hcp.iorlist):
            o = -o+0.
        db1 = dumbbell(0,np.array([0.145,0.,0.]),np.array([0,0,0],dtype=int))
        db2 = dumbbell(1,np.array([0.145,0.,0.]),np.array([0,0,0],dtype=int))
        testjump = jump(db1,db2,1,1)
        count=0
        testlist=[]
        for jl in jset_hcp:
            for j in jl:
                if j==testjump:
                    count+=1
                    testlist=jl
        self.assertEqual(len(testlist),24)
        self.assertEqual(count,1)

        #test_indices
        #First check if they have the same number of lists and elements
        jindlist=[jind_cube,jind_fcc,jind_si,jind_hcp]
        jsetlist=[jset_cube,jset_fcc,jset_si,jset_hcp]
        pdbcontainerlist = [pdbcontainer_cube,pdbcontainer_fcc,pdbcontainer_si,pdbcontainer_hcp]
        for pdbcontainer,jset,jind in zip(pdbcontainerlist,jsetlist,jindlist):
            self.assertEqual(len(jind),len(jset))
            #now check if all the elements are correctly correspondent
            for lindex in range(len(jind)):
                self.assertEqual(len(jind[lindex]),len(jset[lindex]))
                for jindex in range(len(jind[lindex])):
                    (i1,o1) = pdbcontainer.iorlist[jind[lindex][jindex][0]]
                    (i2,o2) = pdbcontainer.iorlist[jind[lindex][jindex][1]]
                    self.assertEqual(jset[lindex][jindex].state1.i,i1)
                    self.assertEqual(jset[lindex][jindex].state2.i,i2)
                    self.assertTrue(np.allclose(jset[lindex][jindex].state1.o,o1))
                    self.assertTrue(np.allclose(jset[lindex][jindex].state2.o,o2))

    def test_mStates(self):
        dbstates=dbStates(self.crys,0,self.family)
        mstates1 = mStates(self.crys,0,self.family)

        #check that symmetry analysis is correct
        self.assertEqual(len(mstates1.symorlist),4)

        #check that negative orientations are accounted for
        for i in range(4):
            self.assertEqual(len(mstates1.symorlist[i])/len(dbstates.symorlist[i]),2)

        #check that every (i,or) set is accounted for
        sm=0
        for i in mstates1.symorlist:
            sm += len(i)
        self.assertEqual(sm,len(mstates1.iorlist))

        #check indexmapping
        Glist = list(mstates1.crys.G)
        x = np.random.randint(0,len(Glist))
        g = Glist[x] #select random groupop
        i,o = mstates1.iorlist[0]
        for stateind,tup in enumerate(mstates1.iorlist):
            i,o = tup[0],tup[1]
            R, (ch,inew) = mstates1.crys.g_pos(g,np.array([0,0,0]),(mstates1.chem,i))
            onew  = np.dot(g.cartrot,o)
            count=0
            for j,t in enumerate(mstates1.iorlist):
                if(t[0]==inew and np.allclose(t[1],onew)):
                    foundindex=j
                    count+=1
            self.assertEqual(count,1)
            self.assertEqual(foundindex,mstates1.indexmap[x][stateind])

        #Check indexing of symlist
        i1=np.random.randint(0,len(dbstates.indsymlist))
        l = dbstates.indsymlist[i1]
        i2=np.random.randint(0,len(l))
        tupFromSymor = dbstates.symorlist[i1][i2]
        tupFromIorlist = dbstates.iorlist[dbstates.indsymlist[i1][i2]]
        self.assertTrue(tupFromSymor[0]==tupFromIorlist[0] and np.allclose(tupFromSymor[1],tupFromIorlist[1]))

    def test_mixedjumps(self):
        famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
        family = [famp0]
        mdbcontainer = mStates(cube,0,family)
        #check for the correct number of states
        jset,jind = mdbcontainer.jumpnetwork(0.3,0.01,0.01)
        test_dbi = dumbbell(0, np.array([0.126,0.,0.]),np.array([0,0,0]))
        test_dbf = dumbbell(0, np.array([0.126,0.,0.]),np.array([0,1,0]))
        count=0
        for i,jlist in enumerate(jset):
            for q,j in enumerate(jlist):
                if j.state1.db == test_dbi:
                    if j.state2.db == test_dbf:
                        if j.c1 == j.c2 == 1:
                           count += 1
                           jtest = jlist
        self.assertEqual(count,1) #see that this jump has been taken only once into account
        self.assertEqual(len(jtest),24)

        #check if conditions for mixed dumbbell transitions are satisfied
        count=0
        for jl in jset:
            for j in jl:
                if j.c1==-1 or j.c2==-1:
                    count+=1
                    break
                if not (j.state1.i_s==j.state1.db.i and j.state2.i_s==j.state2.db.i and np.allclose(j.state1.R_s,j.state1.db.R) and np.allclose(j.state2.R_s,j.state2.db.R)):
                    count+=1
                    break
            if count==1:
                break
        self.assertEqual(count,0)

        #test_indices
        #First check if they have the same number of lists and elements
        self.assertEqual(len(jind),len(jset))
        #now check if all the elements are correctly correspondent
        for lindex in range(len(jind)):
            self.assertEqual(len(jind[lindex]),len(jset[lindex]))
            for jindex in range(len(jind[lindex])):
                (i1,o1) = mdbcontainer.iorlist[jind[lindex][jindex][0]]
                (i2,o2) = mdbcontainer.iorlist[jind[lindex][jindex][1]]
                self.assertEqual(jset[lindex][jindex].state1.db.i,i1)
                self.assertEqual(jset[lindex][jindex].state2.db.i,i2)
                self.assertTrue(np.allclose(jset[lindex][jindex].state1.db.o,o1))
                self.assertTrue(np.allclose(jset[lindex][jindex].state2.db.o,o2))
