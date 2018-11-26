import numpy as np
from states import *
from representations import *
import GFcalc
from functools import reduce
import scipy.linalg as la
from onsager.OnsagerCalc import Interstitial

class BareDumbbell(Interstitial):
    """
    class to compute Green's function for a bare interstitial dumbbell
    diffusing through as crystal.
    """
    def __init__(self,container,jumpnetwork, mixed=False):
        """
        param: container - container object for dumbbell states
        param: jumpnetwork - jumpnetwork (either omega_0 or omega_2)
        """
        self.container = container
        self.jumpnetwork = jumpnetwork
        self.N = sum(1 for w in container.symorlist for i in w)
        self.VB,self.VV = self.FullVectorBasis(mixed)
        self.NV = len(self.VB)

        self.omega_invertible = True
        if self.NV > 0:
            self.omega_invertible = any(np.allclose(g.cartrot,-np.eye(3)) for g  in self.container.crys.G)
        #What is this for though?
        if self.omega_invertible:
            self.bias_solver = lambda omega,b : -la.solve(-omega,b,sym_pos=True)
        else:
            # pseudoinverse required:
            self.bias_solver = lambda omega, b: np.dot(pinv2(omega), b)

        self.sitegroupops = self.generateStateGroupOps()
        self.jumpgroupops = self.generateJumpGroupOps()


    def FullVectorBasis(self,mixed):
        crys = self.container.crys
        chem = self.container.chem
        z = np.zeros(3,dtype=int)

        def makeglist(tup):
            glist = []
            for g in self.container.crys.G:
                r1, (ch,i1) = crys.g_pos(g,z,(chem,tup[0]))
                onew = np.dot(g.cartrot,tup[1])
                if mixed==False:
                    if np.allclose(onew+tup[1],z):
                        onew=-onew
                if tup[0]==i1 and np.allclose(r1,z) and np.allclose(onew,tup[1]): #the state remains unchanged
                    glist.append(g)
            return glist
        lis=[]
        for statelistind,statelist in enumerate(self.container.symorlist):
            N = len(self.container.iorlist)
            glist=makeglist(statelist[0])
            vbasis=reduce(starset.crys.CombineVectorBasis,[starset.crys.VectorBasis(*g.eigen()) for g in glist])
            for v in crys.vectlist(vbasis):
                v /= np.sqrt(len(statelist))
                vb = np.zeros((N,3))
                for gind,g in enumerate(crys.G):
                    vb[self.container.indexmap[gind][self.container.indsymlist[statelistind][0]]] = self.g_direc(g,v)
                lis.append(vb)
            VV = np.zeros((3, 3, len(lis), len(lis)))
            for i, vb_i in enumerate(lis):
                for j, vb_j in enumerate(lis):
                    VV[:, :, i, j] = np.dot(vb_i.T, vb_j)

        return np.array(lis),VV


    def generateStateGroupOps(self):
        """
        Returns a list of lists of groupOps that map the first element of each list in symorlist
        to the corresponding elements in the same list.
        """
        glist=[]
        for lind,l in enumerate(self.container.symorlist):
            stind = self.container.indsymlist[lind][0]
            tup0 = l[0]
            lis=[]
            for ind,tup in enumerate(l):
                for gind,g in enumerate(self.container.crys.G):
                    if self.container.indexmap[gind][ind]==stind:
                        lis.append(g)
            glist.append(lis)
        return glist

    def generateJumpGroupOps(self):
        """
        which group operations land the first jump of a jump list to the rest of the jumps in the same
        list.
        """
        glist=[]
        for jlist in jumpnetwork:
            tup=jlist[0]
            lis=[]
            for j in jlist:
                for gind,g in self.container.crys.G:
                    if self.container.indexmap[gind][tup[0]]==j[0]:
                        if self.container.indexmap[gind][tup[1]]==j[1]:
                            if np.allclose(tup[2],self.container.crys.g_direc(g,j[2])):
                                lis.append(g)
            glist.append(lis)
        return glist

    def stateprob(self, pre, betaene):
        """Returns our (i,or) probabilities, normalized, as a vector.
           Straightforward extension from vacancy case.
        """
        # be careful to make sure that we don't under-/over-flow on beta*ene
        minbetaene = min(betaene)
        rho = np.array([pre[w] * np.exp(minbetaene - betaene[w]) for w in self.container.invmap])
        return rho / sum(rho)

    def ratelist(self, pre, betaene, preT, betaeneT):
        """Returns a list of lists of rates, matched to jumpnetwork"""
        stateene = np.array([betaene[w] for w in self.container.invmap])
        statepre = np.array([pre[w] for w in self.container.invmap])
        return [[pT * np.exp(stateene[i] - beT) / statepre[i]
                 for i, j, dx, c1, c2 in t]
                for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]

    def symmratelist(self, pre, betaene, preT, betaeneT):
        """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
        stateene = np.array([betaene[w] for w in self.container.invmap])
        statepre = np.array([pre[w] for w in self.container.invmap])
        return [[pT * np.exp(0.5 * stateene[i] + 0.5 * stateene[j] - beT) / np.sqrt(statepre[i] * statepre[j])
                 for i, j, dx, c1, c2 in t]
                for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]

    def diffusivity(self, pre, betaene, preT, betaeneT):
        """
        Computes bare dumbbell diffusivity - works for mixed as well pure.
        """
        if len(pre) != len(self.container.symorlist):
            raise IndexError("length of prefactor {} doesn't match symorlist".format(pre))
        if len(betaene) != len(self.container.symorlist):
            raise IndexError("length of energies {} doesn't match symorlist".format(betaene))
        if len(preT) != len(self.jumpnetwork):
            raise IndexError("length of prefactor {} doesn't match jump network".format(preT))
        if len(betaeneT) != len(self.jumpnetwork):
            raise IndexError("length of energies {} doesn't match jump network".format(betaeneT))

        rho = self.siteprob(pre, betaene)
        sqrtrho = np.sqrt(rho)
        ratelist = self.ratelist(pre, betaene, preT, betaeneT)
        symmratelist = self.symmratelist(pre, betaene, preT, betaeneT)
        omega_ij = np.zeros((self.N, self.N))
        domega_ij = np.zeros((self.N, self.N))
        bias_i = np.zeros((self.N, 3))
        dbias_i = np.zeros((self.N, 3))
        D0 = np.zeros((3, 3))
        Dcorrection = np.zeros((3, 3))
        Db = np.zeros((3, 3))
        stateene = np.array([betaene[w] for w in self.container.invmap])
        #Boltmann averaged energies of all states
        Eave = np.dot(rho, siteene)

        for jlist, rates, symmrates, beT in zip(self.jumpnetwork,ratelist,symmratelist,betaeneT):
            for (i,j,dx,c1,c2),rate,symmrate in zip(jlist,rates,symmrates):
                omega_ij[i,j] += symmrate
                omega_ij[i,i] -= rate
                domega_ij[i,j] += symmrate * (bET - 0.5 * (stateene[i] + stateene[j]))
                bias_i[i] += sqrtrho[i] * rate * dx
                dbias_i[i] += sqrtrho[i] * rate * dx * (bET - 0.5 * (siteene[i] + Eave))
                #for domega and dbias - read up section 2.2 in the paper.
                #These are to evaluate the derivative of D wrt to beta. Read later.
                D0 += 0.5 * np.outer(dx, dx) * rho[i] * rate
                Db += 0.5 * np.outer(dx, dx) * rho[i] * rate * (bET - Eave)
                #Db - derivative with respect to beta
