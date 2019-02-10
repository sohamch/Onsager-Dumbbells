import numpy as np
from states import *
import onsager.crystal as crystal
from representations import *
import GFcalc
import stars
import vector_stars
from functools import reduce
from scipy.linalg import pinv2
from onsager.OnsagerCalc import Interstitial, VacancyMediated
#Making stateprob, ratelist and symmratelist universal functions so that I can also use them later on in the case of solutes.

def stateprob(self, pre, betaene, invmap):
    """Returns our (i,or) probabilities, normalized, as a vector.
       Straightforward extension from vacancy case.
    """
    # be careful to make sure that we don't under-/over-flow on beta*ene
    minbetaene = min(betaene)
    rho = np.array([pre[w] * np.exp(minbetaene - betaene[w]) for w in self.container.invmap])
    return rho / sum(rho)

#make a static method and reuse later for solute case?
def ratelist(self, pre, betaene, preT, betaeneT, invmap):
    """Returns a list of lists of rates, matched to jumpnetwork"""
    stateene = np.array([betaene[w] for w in self.container.invmap])
    statepre = np.array([pre[w] for w in self.container.invmap])
    return [[pT * np.exp(stateene[i] - beT) / statepre[i]
             for i, j, dx, c1, c2 in t]
            for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]

def symmratelist(self, pre, betaene, preT, betaeneT, invmap):
    """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
    stateene = np.array([betaene[w] for w in self.container.invmap])
    statepre = np.array([pre[w] for w in self.container.invmap])
    return [[pT * np.exp(0.5 * stateene[i] + 0.5 * stateene[j] - beT) / np.sqrt(statepre[i] * statepre[j])
             for i, j, dx, c1, c2 in t]
            for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]

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

        # self.sitegroupops = self.generateStateGroupOps()
        # self.jumpgroupops = self.generateJumpGroupOps()


    def FullVectorBasis(self,mixed):
        crys = self.container.crys
        chem = self.container.chem
        z = np.zeros(3,dtype=int)

        def makeglist(tup):
            """
            Returns a set of gops that leave a state unchanged
            At the least there will be the identity
            """
            glist = set([])
            for g in self.container.crys.G:
                r1, (ch,i1) = crys.g_pos(g,z,(chem,tup[0]))
                onew = np.dot(g.cartrot,tup[1])
                if not mixed:
                    if np.allclose(onew+tup[1],z):
                        onew=-onew
                if tup[0]==i1 and np.allclose(r1,z) and np.allclose(onew,tup[1]): #the state remains unchanged
                    glist.add(g - crys.g_pos(g, z, (chem, tup[0]))[0])
            return glist
        lis=[]
        for statelistind,statelist in enumerate(self.container.symorlist):
            N = len(self.container.iorlist)
            glist=makeglist(statelist[0])
            vbasis=reduce(crystal.CombineVectorBasis,[crystal.VectorBasis(*g.eigen()) for g in glist])
            for v in crys.vectlist(vbasis):
                v /= np.sqrt(len(statelist))
                vb = np.zeros((N,3))
                for gind,g in enumerate(crys.G):
                    vb[self.container.indexmap[gind][self.container.indsymlist[statelistind][0]]] = self.g_direc(g,v)
                    #What if this changes the vector basis for the state itself?
                    #There are several groupos that leave a state unchanged.
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
                    if self.container.indexmap[g][ind]==stind:
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

    # def stateprob(self, pre, betaene):
    #     """Returns our (i,or) probabilities, normalized, as a vector.
    #        Straightforward extension from vacancy case.
    #     """
    #     # be careful to make sure that we don't under-/over-flow on beta*ene
    #     minbetaene = min(betaene)
    #     rho = np.array([pre[w] * np.exp(minbetaene - betaene[w]) for w in self.container.invmap])
    #     return rho / sum(rho)
    #
    # #make a static method and reuse later for solute case?
    # def ratelist(self, pre, betaene, preT, betaeneT):
    #     """Returns a list of lists of rates, matched to jumpnetwork"""
    #     stateene = np.array([betaene[w] for w in self.container.invmap])
    #     statepre = np.array([pre[w] for w in self.container.invmap])
    #     return [[pT * np.exp(stateene[i] - beT) / statepre[i]
    #              for i, j, dx, c1, c2 in t]
    #             for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]
    #
    # def symmratelist(self, pre, betaene, preT, betaeneT):
    #     """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
    #     stateene = np.array([betaene[w] for w in self.container.invmap])
    #     statepre = np.array([pre[w] for w in self.container.invmap])
    #     return [[pT * np.exp(0.5 * stateene[i] + 0.5 * stateene[j] - beT) / np.sqrt(statepre[i] * statepre[j])
    #              for i, j, dx, c1, c2 in t]
    #             for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]

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

        rho = stateprob(pre, betaene, self.container.invmap)
        sqrtrho = np.sqrt(rho)
        ratelist = ratelist(pre, betaene, preT, betaeneT, self.container.invmap)
        symmratelist = symmratelist(pre, betaene, preT, betaeneT, self.container.invmap)
        omega_ij = np.zeros((self.N, self.N))
        domega_ij = np.zeros((self.N, self.N))
        bias_i = np.zeros((self.N, 3))
        dbias_i = np.zeros((self.N, 3))
        D0 = np.zeros((3, 3))
        Dcorrection = np.zeros((3, 3))
        Db = np.zeros((3, 3))
        stateene = np.array([betaene[w] for w in self.container.invmap])
        #Boltmann averaged energies of all states
        Eave = np.dot(rho, stateene)

        for jlist, rates, symmrates, bET in zip(self.jumpnetwork,ratelist,symmratelist,betaeneT):
            for ((i,j),dx),rate,symmrate in zip(jlist,rates,symmrates):
                omega_ij[i,j] += symmrate
                omega_ij[i,i] -= rate
                domega_ij[i,j] += symmrate * (bET - 0.5 * (stateene[i] + stateene[j]))
                bias_i[i] += sqrtrho[i] * rate * dx
                dbias_i[i] += sqrtrho[i] * rate * dx * (bET - 0.5 * (stateene[i] + Eave))
                #for domega and dbias - read up section 2.2 in the paper.
                #These are to evaluate the derivative of D wrt to beta. Read later.
                D0 += 0.5 * np.outer(dx, dx) * rho[i] * rate
                Db += 0.5 * np.outer(dx, dx) * rho[i] * rate * (bET - Eave)
                #Db - derivative with respect to beta
        # gamma_i = np.zeros((self.N, 3))
        gamma_i = np.dot(pinv2(omega_ij), bias_i)
        Dcorr = np.zeros((3,3))
        for i in range(self.N):
            Dcorr += np.outer(bias_i[i],gamma_i[i])

        return D0 + Dcorr

class dumbbellMediated(VacancyMediated):
    """
    class to compute dumbbell mediated solute transport coefficients. We inherit the calculator
    for vacancies from Prof. Trinkle's code for vacancies with changes as and when required.

    Here, unlike vacancies, we must compute the Green's Function by Block inversion
    and Taylor expansion (as in the GFCalc module) for both bare pure (g0)
    and mixed(g2) dumbbells, since our Dyson equation requires so.
    Also, instead of working with crystal and chem, we work with the container objects.
    """
    def __init__(self,pdbcontainer,mdbcontainer,cutoff,solt_solv_cut,solv_solv_cut,closestdistance,NGFmax=4,Nthermo=0):
        #All the required quantities will be extracted from the containers as we move along
        self.pdbcontainer = pdbcontainer
        self.mdbcontainer = mdbcontainer
        self.crys = pdbcontainer.crys #we assume this is the same in both containers
        self.chem = pdbcontainer.chem

        self.GFcalc_pdb = self.GFCalculator(NGFmax)
        self.GFcalc_mdb = self.GFCalculator(NGFmax,pdb=False)
        self.thermo = stars.StarSet(pdbcontainer,mdbcontainer,self.om0_jn_states,self.om2_jn_states)
        self.kinetic = stars.StarSet(pdbcontainer,mdbcontainer,self.om0_jn_states,self.om2_jn_states)

        #Note - even if empty, our starsets go out to atleast the NNstar - later we'll have to keep this in mind
        self.NNstar = stars.StarSet(pdbcontainer,mdbcontainer,self.om0_jn_states,self.om2_jn_states,2)
        self.vkinetic = vector_stars.vectorStars()

        #Generate the initialized crystal and vector stars and the jumpnetworks with the kinetic shell
        self.generate(Nthermo,cutoff,solt_solv_cut,solv_solv_cut,closestdistance)

        #Generate the jumpnetworks using the kinetic shell
        # TODO: Implement generatematrices when required
        # self.generatematrices()

    def GFCalculator(self,NGFmax=0,pdb=True):
        """
        Mostly similar to vacancy case - returns the GF calculator for complex or mixed dumbbell
        states as specified.
        """
        #unlike the vacancy case, it is better to recalculate for now.
        if NGFmax<0: raise ValueError("NGFmax must be greater than 0")
        self.NGFmax=NGFmax
        self.clearcache(pdb)#make all precalculated lists empty
        if pdb:
            return GFcalc_dumbbells.GF_dumbbells(self.pdbcontainer,self.om0_jn,NGFmax)
        return GFcalc_dumbbells.GF_dumbbells(self.mdbcontainer,self.om2_jn,NGFmax)

    def clearcache(self,pdb):
        """
        Makes all pre-computed dicitionaries empty for pure or mixed dumbbell as specified
        """
        if pdb:
            self.GFvalues_pdb, self.LvvValues_pdb, self.etav_pdb = {}, {}, {}
        else:
            self.GFvalues_mdb, self.LvvValues_mdb, self.etav_mdb = {}, {}, {}

    def generate_jnets(self,cutoff,solt_solv_cut,solv_solv_cut,closestdistance):

        #first omega0 and omega2 - indexed to purestates and mixed states
        (self.jnet0,self.jnet0_indexed),(self.jnet2,self.jnet2_indexed)=\
        self.vkinetic.pdbcontainer.jumpnetwork(cutoff,solv_solv_cut,closestdistance),\
        self.vkinetic.mdbcontainer.jumpnetwork(cutoff,solt_solv_cut,closestdistance)

        #Next - omega1 - indexed to purestates
        (self.jnet_1,self.jnet_indexed), self.om1types = self.vkinetic.starset.jumpnetwork_omega1()

        #next, omega3 and omega_4, indexed to pure and mixed states
        (self.symjumplist_omega43_all,self.symjumplist_omega43_all_indexed),(self.symjumplist_omega4,self.symjumplist_omega4_indexed),(self.symjumplist_omega3,self.symjumplist_omega3_indexed)=self.vkinetics.starset.jumpnetwork_omega34(cutoff,solv_solv_cut,solt_solv_cut,closestdistance)

    def generate(self,Nthermo):

        if Nthermo==getattr(self,"Nthermo",0): return
        self.Nthermo = Nthermo
        self.thermo.generate(Nthermo)
        self.kinetic.generate(Nthermo+1)
        self.Nmixedstates = len(self.kinetic.mixedstates)
        self.Npurestates = len(self.kinetic.purestates)
        self.vkinetic.generate(self.kinetic) #we generate the vector star out of the kinetic shell
        #Now generate the pure and mixed dumbbell Green functions expnsions - internalized within vkinetic.
        (self.GFexpansion_pure,self.GFstarset_pure,self.GFPureStarInd), (self.GFexpansion_mixed,self.GFstarset_mixed,self.GFMixedStarInd)\
        = self.vkinetic.GFexpansion()

        #See how the thermo2kin and all of that works later as and when needed
        #Generate the jumpnetworks
        self.generate_jnets(cutoff,solt_solv_cut,solv_solv_cut,closestdistance)

        #clear the cache of GFcalcs
        self.clearcache()

    def Lij(self, pre0, betaene0, pre0T, betaene0T, pre2, betaene2, pre2T, betaene2T):
        """
        Function to calculate the onsager transport coefficients.
        """
        #First, build g2 from omega2 jumpnetwork using pinv
        rate2list = ratelist(pre2, betaene2, pre2T, betaene2T, self.mdbcontainer.invmap)
        rate0list = ratelist(pre0, betaene0, pre0T, betaene0T, self.pdbcontainer.invmap)
        # symmrate2list = symmratelist(pre, betaene, preT, betaeneT, self.container.invmap)

        #get the bias1 and bias2 expansions
        self.biases = self.vkinetic.biasexpansion(jnet_omega1,jnet_omega2,jtype,jnet_43)

        #generate the non-local solute and solvent biases for initial states in pure and mixed stateset of vkinetic.
        self.NlsoluteBias1
        self.NlsolventBias1
        self.NlsoluteBias2
        self.NlsoluteBias1

        #generate the non-local complex-complex block
        oemga1_nonloc = np.zeros((len(self.vkinetic.starset.purestates),len(self.vkinetic.starset.purestates)))
        #use the indexed omega1 to fill up the elements
        #omega1 is indexed to the purestates list in vkinetic.starset


        #invert it with pinv2

        #dot with the cartesian bias to get the eta0 for each state
        eta0=np.zeros((len(self.vkinetic.starset.purestates).3))

        #Now for omega2
        oemga2_nonloc = np.zeros((len(self.vkinetic.starset.mixedstates),len(self.vkinetic.starset.mixedstates)))
        #use the indexed omega2 to fill this up

        #invert it with pinv2

        #dot with the cartesian bias vectors to get the eta0 for each state
