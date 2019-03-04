import numpy as np
from states import *
import onsager.crystal as crystal
from representations import *
import GFcalc_dumbbells
import stars
import vector_stars
from functools import reduce
from scipy.linalg import pinv2
from onsager.OnsagerCalc import Interstitial, VacancyMediated

#Making stateprob, ratelist and symmratelist universal functions so that I can also use them later on in the case of solutes.
def stateprob(pre, betaene, invmap):
    """Returns our (i,or) probabilities, normalized, as a vector.
       Straightforward extension from vacancy case.
    """
    # be careful to make sure that we don't under-/over-flow on beta*ene
    minbetaene = min(betaene)
    rho = np.array([pre[w] * np.exp(minbetaene - betaene[w]) for w in invmap])
    return rho / sum(rho)

#make a static method and reuse later for solute case?
def ratelist(jumpnetwork, pre, betaene, preT, betaeneT, invmap):
    """Returns a list of lists of rates, matched to jumpnetwork"""
    stateene = np.array([betaene[w] for w in invmap])
    statepre = np.array([pre[w] for w in invmap])
    return [[pT * np.exp(stateene[i] - beT) / statepre[i]
             for (i, j), dx in t]
            for t, pT, beT in zip(jumpnetwork, preT, betaeneT)]

def symmratelist(jumpnetwork, pre, betaene, preT, betaeneT, invmap):
    """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
    stateene = np.array([betaene[w] for w in invmap])
    statepre = np.array([pre[w] for w in invmap])
    return [[pT * np.exp(0.5 * stateene[i] + 0.5 * stateene[j] - beT) / np.sqrt(statepre[i] * statepre[j])
             for (i, j), dx in t]
            for t, pT, beT in zip(jumpnetwork, preT, betaeneT)]

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
        for jlist in self.jumpnetwork:
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
        ratelist = ratelist(self.jumpnetwork,pre, betaene, preT, betaeneT, self.container.invmap)
        symmratelist = symmratelist(self.jumpnetwork,pre, betaene, preT, betaeneT, self.container.invmap)
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
        (self.jnet0,self.jnet0_indexed),(self.jnet2,self.jnet2toIorList)=\
        self.pdbcontainer.jumpnetwork(cutoff,solv_solv_cut,closestdistance),\
        self.mdbcontainer.jumpnetwork(cutoff,solt_solv_cut,closestdistance)
        self.crys = pdbcontainer.crys #we assume this is the same in both containers
        self.chem = pdbcontainer.chem
        # self.jnet2_indexed = self.kinetic.starset.jnet2_indexed
        self.thermo = stars.StarSet(pdbcontainer,mdbcontainer,(self.jnet0,self.jnet0_indexed),(self.jnet2,self.jnet2toIorList))
        self.kinetic = stars.StarSet(pdbcontainer,mdbcontainer,(self.jnet0,self.jnet0_indexed),(self.jnet2,self.jnet2toIorList))

        #Note - even if empty, our starsets go out to atleast the NNstar - later we'll have to keep this in mind
        self.NNstar = stars.StarSet(pdbcontainer,mdbcontainer,(self.jnet0,self.jnet0_indexed),(self.jnet2,self.jnet2toIorList),2)
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
            return GFcalc_dumbbells.GF_dumbbells(self.pdbcontainer,self.jnet0_indexed,NGFmax)
        return GFcalc_dumbbells.GF_dumbbells(self.mdbcontainer,self.jnet2toIorList,NGFmax)

    def clearcache(self):
        """
        Makes all pre-computed dicitionaries empty for pure or mixed dumbbell as specified
        """
        self.GFvalues_pdb, self.LvvValues_pdb, self.etav_pdb = {}, {}, {}
        self.GFvalues_mdb, self.LvvValues_mdb, self.etav_mdb = {}, {}, {}

    def generate_jnets(self,cutoff,solt_solv_cut,solv_solv_cut,closestdistance):

        #first omega0 and omega2 - indexed to purestates and mixed states
        self.jnet2_indexed = self.vkinetic.starset.jnet2_indexed
        # self.omeg2types = self.vkinetic.starset.jnet2_types
        self.jtags2 = self.vkinetic.starset.jtags2
        #Next - omega1 - indexed to purestates
        (self.jnet_1,self.jnet1_indexed,self.jtags1), self.om1types = self.vkinetic.starset.jumpnetwork_omega1()

        #next, omega3 and omega_4, indexed to pure and mixed states
        (self.symjumplist_omega43_all,self.symjumplist_omega43_all_indexed),(self.symjumplist_omega4,self.symjumplist_omega4_indexed,self.jtags4),\
        (self.symjumplist_omega3,self.symjumplist_omega3_indexed,self.jtags3)=self.vkinetic.starset.jumpnetwork_omega34(cutoff,solv_solv_cut,solt_solv_cut,closestdistance)

    def generate(self,Nthermo,cutoff,solt_solv_cut,solv_solv_cut,closestdistance):

        if Nthermo==getattr(self,"Nthermo",0): return
        self.Nthermo = Nthermo
        self.thermo.generate(Nthermo)
        self.kinetic.generate(Nthermo+1)
        # self.Nmixedstates = len(self.kinetic.mixedstates)
        # self.Npurestates = len(self.kinetic.purestates)
        self.vkinetic.generate(self.kinetic) #we generate the vector star out of the kinetic shell
        #Now generate the pure and mixed dumbbell Green functions expnsions - internalized within vkinetic.
        # (self.GFexpansion_pure,self.GFstarset_pure,self.GFPureStarInd), (self.GFexpansion_mixed,self.GFstarset_mixed,self.GFMixedStarInd)\
        # = self.vkinetic.GFexpansion()

        #See how the thermo2kin and all of that works later as and when needed
        #Generate the jumpnetworks
        self.generate_jnets(cutoff,solt_solv_cut,solv_solv_cut,closestdistance)
        #Generate the GFexpansions
        # self.GFcalc_pdb = self.GFCalculator(NGFmax)
        # self.GFcalc_mdb = self.GFCalculator(NGFmax,pdb=False)
        #clear the cache of GFcalcs
        self.clearcache()

    def calc_eta(self, pre0, betaene0, pre0T, betaene0T, pre2, betaene2, pre2T, betaene2T):
        """
        Function to calculate the periodic eta vectors.
        """
        rate2list = ratelist(self.jnet2toIorList, pre2, betaene2, pre2T, betaene2T, self.vkinetic.starset.mdbcontainer.invmap)
        rate0list = ratelist(self.jnet0_indexed, pre0, betaene0, pre0T, betaene0T, self.vkinetic.starset.pdbcontainer.invmap)
        # symmrate2list = symmratelist(pre, betaene, preT, betaeneT, self.container.invmap)
        #the non-local bias for the complex space has to be carried out based on the omega0 jumpnetwork, not the omega1 jumpnetwork.
        #this is because all the jumps that are allowed by omega0 out of a given dumbbell state are not there in omega1
        #That is because omega1 considers only those states that are in the kinetic shell. Not outside it.
        #We need to build it up directly from the omega0 jumpnetwork.
        ## TODO: Find a more efficient way to do this. Maybe build up a vector star separately for the omega0 space. Is it worth it though?
        # self.NlsoluteBias1 = np.zeros((len(self.vkinetic.starset.purestates),3))
        # self.NlsolventBias1 = np.zeros((len(self.vkinetic.starset.purestates),3))

        #get the bias1 and bias2 expansions
        self.biases = self.vkinetic.biasexpansion(self.jnet_1,self.jnet2,self.om1types,self.symjumplist_omega43_all)

        #generate the non-local solute and solvent biases for initial states in pure and mixed stateset of vkinetic.
        #first,generate the solute bias in complex space.
        # rate1_nonloc = np.array([rate0list[self.om1types[i]][0] for i in range(len(self.jnet_1))])
        # #Get the total bias vector with components along basis vectors of states
        # bias1SoluteTotNonLoc = np.dot(bias1solute,rate1_nonloc)
        # bias1SolventTotNonLoc = np.dot(bias1solvent,rate1_nonloc)
        #
        # #Now go state by state - bring back bias vector to cartesian form.
        # for st in self.vkinetic.starset.purestates:
        #     indlist = self.vkinetic.stateToVecStar_pure[st]
        #     if len(indlist)!=0:
        #         self.NlsoluteBias1[self.vkinetic.starset.pureindexdict[st][0]][:]=\
        #         sum([bias1SoluteTotNonLoc[tup[0]]*self.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #         self.NlsolventBias1[self.vkinetic.starset.pureindexdict[st][0]][:]=\
        #         sum([bias1SolventTotNonLoc[tup[0]]*self.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        #First check if non-local biases should be zero anyway (as is the case with highly symmetric lattice - in that case vecpos_bare should be zero)
        if len(self.vkinetic.vecpos_bare)==0:
            self.eta00_solvent = np.zeros((len(self.vkinetic.starset.purestates),3))
            self.eta00_solute = np.zeros((len(self.vkinetic.starset.purestates),3))
        #otherwise, we need to build the bare bias expansion
        else:
            #First we build up for just the bare starset
            biasBareExpansion = self.biases[-1]
            self.NlsolventBias_bare = np.zeros((len(self.vkinetic.starset.bareStates),3))
            bias0SolventTotNonLoc = np.dot(biasBareExpansion,np.array([rate0list[i][0] for i in range(len(self.jnet0))]))
            for st in self.vkinetic.starset.bareStates:
                indlist = self.vkinetic.bareStTobareStar[st]
                if len(indlist)!=0:
                    self.NlsolventBias_bare[self.vkinetic.starset.bareindexdict[st][0]][:]=\
                    sum([bias0SolventTotNonLoc[tup[0]]*self.vkinetic.vecvec_bare[tup[0]][tup[1]] for tup in indlist])

            #Next build up W0_ij
            omega0_nonloc = np.zeros((len(self.vkinetic.starset.bareStates),len(self.vkinetic.starset.bareStates)))
            #use the indexed omega2 to fill this up - need omega2 indexed to mixed subspace of starset
            for rate0,jlist in zip(rate0list,self.vkinetic.starset.jumpnetwork_omega0_indexed):
                for (i,j),dx in jlist:
                    omega0_nonloc[i,j] += rate0[0]
                    omega0_nonloc[i,i] -= rate0[0]

            g0 = pinv2(omega0_nonloc)

            self.eta00_solvent_bare = np.tensordot(g0,self.NlsolventBias_bare,axes=(1,0))
            self.eta00_solute_bare = np.zeros_like(self.eta00_solvent_bare)

            #Now match the non-local biases for complex states to the pure states
            self.eta00_solute = np.zeros((len(self.vkinetic.starset.purestates),3))
            self.eta00_solvent = np.zeros((len(self.vkinetic.starset.purestates),3))
            self.NlsolventBias0 = np.zeros((len(self.vkinetic.starset.purestates),3))

            for i in range(len(self.vkinetic.starset.purestates)):
                db = self.vkinetic.starset.purestates[i].db
                db = db - db.R
                for j in range(len(self.vkinetic.starset.bareStates)):
                    count=0
                    if db == self.vkinetic.starset.bareStates[j]:
                        count+=1
                        self.eta00_solvent[i,:] = self.eta00_solvent_bare[j,:].copy()
                        self.NlsolventBias0[i,:] = self.NlsolventBias_bare[j,:].copy()
                        break
                if count !=1:
                    raise ValueError("The dumbbell is not present in the iorlist?count={}".format(count))

        self.NlsoluteBias2 = np.zeros((len(self.vkinetic.starset.mixedstates),3))
        self.NlsolventBias2 = np.zeros((len(self.vkinetic.starset.mixedstates),3))
        bias2solute,bias2solvent = self.biases[2]
        bias2SoluteTotNonLoc = np.dot(bias2solute,np.array([rate2list[i][0] for i in range(len(self.jnet2_indexed))]))
        bias2SolventTotNonLoc = np.dot(bias2solvent,np.array([rate2list[i][0] for i in range(len(self.jnet2_indexed))]))
        #Now go state by state
        for st in self.vkinetic.starset.mixedstates:
            indlist = self.vkinetic.stateToVecStar_mixed[st]
            if len(indlist)!=0:
                self.NlsoluteBias2[self.vkinetic.starset.mixedindexdict[st][0]][:]=\
                sum([bias2SoluteTotNonLoc[tup[0]-self.vkinetic.Nvstars_pure]*\
                self.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

                self.NlsolventBias2[self.vkinetic.starset.mixedindexdict[st][0]][:]=\
                sum([bias2SolventTotNonLoc[tup[0]-self.vkinetic.Nvstars_pure]*\
                self.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        # #generate the non-local complex-complex block w0
        # omega1_nonloc = np.zeros((len(self.vkinetic.starset.purestates),len(self.vkinetic.starset.purestates)))
        # #use the indexed omega1 to fill up the elements
        # for om0rate,jlist in zip(rate1_nonloc,self.jnet1_indexed):
        #     for (i,j),dx in jlist:
        #         omega1_nonloc[i,j] += om0rate
        #         omega1_nonloc[i,i] -= om0rate #"add" the escapes
        # #invert it with pinv2
        # g0_per = pinv2(omega1_nonloc)
        # #tensordot with the cartesian bias to get the eta0 for each state
        # self.eta00_solvent = np.tensordot(g0_per,self.NlsolventBias1,axes=(1,0))
        # self.eta00_solute = np.tensordot(g0_per,self.NlsoluteBias1,axes=(1,0))
        #Now for omega2
        omega2_nonloc = np.zeros((len(self.vkinetic.starset.mixedstates),len(self.vkinetic.starset.mixedstates)))
        #use the indexed omega2 to fill this up - need omega2 indexed to mixed subspace of starset
        for rate2,jlist in zip(rate2list,self.jnet2_indexed):
            for (i,j),dx in jlist:
                omega2_nonloc[i,j] += rate2[0]
                omega2_nonloc[i,i] -= rate2[0]
        #invert it with pinv2
        g2 = pinv2(omega2_nonloc)
        # print(g2.shape)
        # print(self.NlsolventBias1.shape)
        #dot with the cartesian bias vectors to get the eta0 for each state
        self.eta02_solvent=np.tensordot(g2,self.NlsolventBias2,axes=(1,0))
        self.eta02_solute=np.tensordot(g2,self.NlsoluteBias2,axes=(1,0))

        # self.gen_new_jnets()
        #Need to bypass having to generate new jumpnetworks all over again
    def construct_new_expansion(self):
        """
        Function that allows us to construct new bias and bare expansions based on the eta vectors already calculated.

        We don't want to repeat the construction of the jumpnetwork based on the recalculated displacements after
        subtraction of the eta vectors (as in the variational principle).

        The steps are illustrated in the GM slides of Feb 25, 2019 - will include in the detailed documentation later on
        """
        #So what do we have up until now?
        #We have constructed the Nstates x 3 eta0 vectors for pure and mixed states separately
        #But the jtags assume all the eta vectors are in the same list.
        #So, we need to first concatenate the mixed eta vectors into the pure eta vectors.

        self.eta00total_solute = np.zeros((len(self.vkinetic.starset.purestates)+len(self.vkinetic.starset.mixedstates),3))
        self.eta00total_solvent = np.zeros((len(self.vkinetic.starset.purestates)+len(self.vkinetic.starset.mixedstates),3))

        self.eta02total_solute = np.zeros((len(self.vkinetic.starset.purestates)+len(self.vkinetic.starset.mixedstates),3))
        self.eta02total_solvent = np.zeros((len(self.vkinetic.starset.purestates)+len(self.vkinetic.starset.mixedstates),3))

        self.eta0total_solute[:len(self.vkinetic.starset.purestates),:]=self.eta00_solute.copy()
        self.eta0total_solute[len(self.vkinetic.starset.purestates):,:]=self.eta02_solute.copy()

        self.eta0total_solvent[:len(self.vkinetic.starset.purestates),:]=self.eta00_solvent.copy()
        self.eta0total_solvent[len(self.vkinetic.starset.purestates):,:]=self.eta02_solvent.copy()

        #create updated bias expansions
        #to get warmed up, let's do it for bias1expansion
        #Step 2 - construct the projection of eta vectors
        self.delbias1expansion_solute = np.zeros_like(self.biases[1][0])
        self.delbias1expansion_solvent = np.zeros_like(self.biases[1][0])

        self.delbias4expansion_solute = np.zeros_like(self.biases[4][0])
        self.delbias4expansion_solvent = np.zeros_like(self.biases[4][0])
        for i in range(self.vkinetic.Nvstars_pure):
            #get the representative state(its index in purestates) and vector
            v0 = self.vkinetic.vecvec[i][0]
            st0 = self.vkinetic.starset.pureindexdict[self.vkinetic.vecpos[i][0]][0]
            #Now go through the omega1 jump network tags
            for jt,initindexdict in self.jtags1:
                #see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                self.delbias1expansion_solute[i,jt] += len(self.vkinetic.vecpos[i])*np.sum(np.dot(initindexdict[st0],self.eta0total_solute))
                self.delbias1expansion_solvent[i,jt] += len(self.vkinetic.vecpos[i])*np.sum(np.dot(initindexdict[st0],self.eta0total_solvent))
            #Now let's build it for omega4
            for jt,initindexdict in self.jtags4:
                #see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                self.delbias4expansion_solute[i,jt] += len(self.vkinetic.vecpos[i])*np.sum(np.dot(initindexdict[st0],self.eta0total_solute))
                self.delbias4expansion_solvent[i,jt] += len(self.vkinetic.vecpos[i])*np.sum(np.dot(initindexdict[st0],self.eta0total_solvent))

        self.delbias3expansion_solute = np.zeros_like(self.biases[3][0])
        self.delbias3expansion_solvent = np.zeros_like(self.biases[3][0])

        self.delbias2expansion_solute = np.zeros_like(self.biases[2][0])
        self.delbias2expansion_solvent = np.zeros_like(self.biases[2][0])

        for i in range(self.vkinetic.Nvstars-self.vkinetic.Nvstars_pure):
            #get the representative state(its index in purestates) and vector
            v0 = self.vkinetic.vecvec[i][0]
            st0 = self.vkinetic.starset.mixedindexdict[self.vkinetic.vecpos[i+self.vkinetic.Nvstars_pure][0]][0]
            #Now go through the omega1 jump network tags
            for jt,initindexdict in self.jtags2:
                #see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                self.delbias2expansion_solute[i,jt] += len(self.vkinetic.vecpos[i])*np.sum(np.dot(initindexdict[st0],self.eta0total_solute))
                self.delbias2expansion_solvent[i,jt] += len(self.vkinetic.vecpos[i])*np.sum(np.dot(initindexdict[st0],self.eta0total_solvent))
            #Now let's build it for omega4
            for jt,initindexdict in self.jtags3:
                #see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                self.delbias3expansion_solute[i,jt] += len(self.vkinetic.vecpos[i])*np.sum(np.dot(initindexdict[st0],self.eta0total_solute))
                self.delbias3expansion_solvent[i,jt] += len(self.vkinetic.vecpos[i])*np.sum(np.dot(initindexdict[st0],self.eta0total_solvent))
