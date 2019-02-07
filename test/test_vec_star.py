import numpy as np
import onsager.crystal as crystal
from stars import *
from test_structs import *
from states import *
from representations import *
from functools import reduce
from vector_stars import *
import unittest

class test_vecstars(unittest.TestCase):
    def setUp(self):
        latt = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])*0.55
        self.DC_Si = crystal.Crystal(latt,[[np.array([0.,0.,0.]),np.array([0.25,0.25,0.25])]],["Si"])

        #keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = dbStates(self.DC_Si,0,family)
        self.mdbcontainer_si = mStates(self.DC_Si,0,family)

        self.jset0,self.jset2 = self.pdbcontainer_si.jumpnetwork(0.4,0.01,0.01), self.mdbcontainer_si.jumpnetwork(0.4,0.01,0.01)

        self.crys_stars = StarSet(self.pdbcontainer_si,self.mdbcontainer_si,self.jset0,self.jset2, Nshells=2)
        self.vec_stars = vectorStars(self.crys_stars)

    def test_basis(self):
        self.assertEqual(len(self.vec_stars.vecpos),len(self.vec_stars.vecvec))
        #choose a random vector star
        vecstarind = np.random.randint(0,len(self.vec_stars.vecpos))
        #get the representative state of the star
        testvecstate = self.vec_stars.vecpos[vecstarind][0]
        count=0
        for i in range(len(self.vec_stars.vecpos)):
            if self.vec_stars.vecpos[vecstarind][0] == self.vec_stars.vecpos[i][0]:
                count += 1
                #The number of times the position list is repeated is also the dimensionality of the basis.

        #Next see what is the algaebric multiplicity of the eigenvalue 1.
        glist=[]
        for g in self.crys_stars.crys.G:
            pairnew = testvecstate.gop(self.crys_stars.crys, self.crys_stars.chem, g)
            pairnew = pairnew - pairnew.R_s
            if vecstarind < self.vec_stars.Nvstars_pure:
                if pairnew == testvecstate or pairnew==-testvecstate:
                    glist.append(g)
            else:
                if pairnew == testvecstate:
                    glist.append(g)
        sumg = sum([g.cartrot for g in glist])/len(glist)
        vals,vecs = np.linalg.eig(sumg)
        count_eigs=0
        for val in vals:
            if np.allclose(val,1.0,atol=1e-8):
                count_eigs+=1
        self.assertEqual(count,count_eigs,msg="{}".format(testvecstate))

    def test_expansions(self):
        #first let us test bias_1 expansion - we make all the transitions have unity rate
        #We also check that the velocity vectors using unsymmetrized rates are correct
        (jnet_1,jnet_1_indexed),jt = self.crys_stars.jumpnetwork_omega1()

        W1rates = np.ones(len(jnet_1))
        #select a state at random from complex states
        for i in range(10):
            stateind = np.random.randint(0,len(self.crys_stars.purestates))
            st = self.crys_stars.purestates[stateind]
            #Now, we calculate the total bias vector
            bias_st_solute=np.zeros(3)
            bias_st_solvent=np.zeros(3)
            count=0
            for jlist in jnet_1:
                for j in jlist:
                    if st==j.state1:
                        count+=1
                        dx = disp(self.crys_stars.crys,self.crys_stars.chem,j.state1,j.state1)
                        bias_st_solute += np.zeros(3)
                        bias_st_solvent += dx

            tot_bias = bia
