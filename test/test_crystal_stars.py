import numpy as np
# from jumpnet3 import *
from stars import *
from test_structs import *
from states import *
# from gensets import *
import unittest

class test_StarSet(unittest.TestCase):

    def setUp(self):
        famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
        family = [famp0]
        pdbcontainer = dbStates(cube,0,family)
        mdbcontainer = mStates(cube,0,family)
        jset0 = pdbcontainer.jumpnetwork(0.3,0.01,0.01)
        jset2 = mdbcontainer.jumpnetwork(0.3,0.01,0.01)
        self.crys_stars = StarSet(pdbcontainer,mdbcontainer,jset0,jset2,2)

    def test_generate(self):
        #test if the starset is generated correctly
        o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
        db = dumbbell(0,o,np.array([1,0,0]))
        for l in self.crys_stars.starset:
            for state in l:
                if state.db==db or state.db==-db:
                    test_list = l.copy()
        self.assertEqual(len(test_list),6)

    def test_jumpnetworks(self):
        #See the example file. Provides much clearer understanding.
        #The tests have just one Wyckoff site for now
        def inlist(jmp,jlist):
            return any(j==jmp for j in jlist)
        hcp_Mg=crystal.Crystal.HCP(0.3294,chemistry=["Mg"])
        fcc_Ni = crystal.Crystal.FCC(0.352,chemistry=["Ni"])
        latt = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])*0.55
        DC_Si = crystal.Crystal(latt,[[np.array([0.,0.,0.]),np.array([0.25,0.25,0.25])]],["Si"])
        crys_list=[hcp_Mg,fcc_Ni,DC_Si]
        famp0 = [np.array([1.,0.,0.])*0.145]
        family = [famp0]
        for struct,crys in enumerate(crys_list):
            pdbcontainer = dbStates(crys,0,family)
            mdbcontainer = mStates(crys,0,family)
            jset0 = pdbcontainer.jumpnetwork(0.45,0.01,0.01)
            jset2 = mdbcontainer.jumpnetwork(0.45,0.01,0.01)
            #4.5 angst should cover atleast the nn distance in all the crystals
            #create starset
            crys_stars = StarSet(pdbcontainer,mdbcontainer,jset0,jset2,1)

            ##TEST omega_1
            omega1_network = crys_stars.jumpnetwork_omega1()[0]
            #select a jump list in random
            x = np.random.randint(0,len(omega1_network))
            #select any jump from this list at random
            y = np.random.randint(0,len(omega1_network[x]))
            jmp=omega1_network[x][y]
            jlist=[]
            #reconstruct the list using the selected jump, without using has tables (sets)
            for g in crys.G:
                jnew1 = jmp.gop(crys,0,g)
                db1new = pdbcontainer.gdumb(g,jmp.state1.db)
                db2new = pdbcontainer.gdumb(g,jmp.state2.db)
                #shift the states back to the origin unit cell
                state1new = SdPair(jnew1.state1.i_s,jnew1.state1.R_s,db1new[0])-jnew1.state1.R_s
                state2new = SdPair(jnew1.state2.i_s,jnew1.state2.R_s,db2new[0])-jnew1.state2.R_s
                jnew = jump(state1new,state2new,jnew1.c1*db1new[1],jnew1.c2*db2new[1])
                if not inlist(jnew,jlist):
                    jlist.append(jnew)
                    jlist.append(-jnew)
            self.assertEqual(len(jlist),len(omega1_network[x]))
            count=0
            for j in jlist:
                for j1 in omega1_network[x]:
                    if j==j1:
                        count+=1
                        break
            self.assertEqual(count,len(jlist))

            omega43_all,omega4_network,omega3_network = crys_stars.jumpnetwork_omega34(0.45,0.01,0.01,0.01)
            self.assertEqual(len(omega4_network),len(omega3_network))
            for jl4,jl3 in zip(omega4_network,omega3_network):
                self.assertEqual(len(jl3),len(jl4))
            ##TEST omega3 and omega4
            omeg34list=[omega3_network,omega4_network]
            for i,omega in enumerate(omeg34list):
                x=np.random.randint(0,len(omega))
                y=np.random.randint(0,len(omega[x]))
                jmp=omega[x][y]
                jlist=[]
                if i==0: #then build omega3
                    for g in crys.G:
                        jnew = jmp.gop(crys,0,g)
                        db2new = pdbcontainer.gdumb(g,jmp.state2.db)
                        state2new = SdPair(jnew.state2.i_s,jnew.state2.R_s,db2new[0])-jnew.state2.R_s
                        jnew = jump(jnew.state1-jnew.state1.R_s,state2new,-1,jnew.c2*db2new[1])
                        if not inlist(jnew,jlist):
                            jlist.append(jnew)
                else: #build omega4
                    for g in crys.G:
                        jnew = jmp.gop(crys,0,g)
                        db1new = pdbcontainer.gdumb(g,jmp.state1.db)
                        state1new = SdPair(jnew.state1.i_s,jnew.state1.R_s,db1new[0])-jnew.state1.R_s
                        jnew = jump(state1new,jnew.state2-jnew.state2.R_s,jnew.c1*db1new[1],-1)
                        if not inlist(jnew,jlist):
                            jlist.append(jnew)
                self.assertEqual(len(jlist),len(omega[x]))
                count=0
                for j1 in omega[x]:
                    for j in jlist:
                        if j==j1:
                            count+=1
                            break
                self.assertEqual(count,len(omega[x]))
