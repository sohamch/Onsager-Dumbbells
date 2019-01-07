import numpy as np
import onsager.crystal as crystal
from states import *
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from Onsager_calc_db import BareDumbbell
import matplotlib.pyplot as plt

kB = physical_constants['Boltzmann constant in eV/K'][0]
a0 = 0.28553
Fe = crystal.Crystal.BCC(a0, "Fe")
# print(Fe)

famp0 = [np.array([1.,1.,0.])/np.sqrt(2)*0.126]
family = [famp0]
pdbcontainer = dbStates(Fe,0,family)
jset,jind = pdbcontainer.jumpnetwork(0.25,0.01,0.01)
len(jind)
#Now extract the required jumps
dx = np.array([0.5,0.5,0.5])*a0-np.array([0.0,0.0,0.0])*a0
mod_dx=np.sqrt(np.dot(dx,dx))
jsetnew=[]
jindnew=[]
for i,jl in enumerate(jind):
    dx = jl[0][2]
    if np.allclose(np.dot(dx,dx),mod_dx**2,atol=1e-8):
        jsetnew.append(jset[i])
        jindnew.append(jind[i])

#Make the prefactors
diffuser = BareDumbbell(pdbcontainer, jindnew)

Dconv = 1e-2
vu0 = 10*Dconv
Etrans = 0.33
#Compute the thermodict
FeDbthermodict = {'pre': np.ones(len(pdbcontainer.symorlist)), 'ene': np.zeros(len(pdbcontainer.symorlist)),
                 'preT': vu0*np.ones(len(jindnew)),
                 'eneT': Etrans*np.ones(len(jindnew))}

#Now compute the diffusivity

D0 = diffuser.diffusivity(FeDbthermodict['pre'],
                             np.zeros_like(FeDbthermodict['ene']),
                             FeDbthermodict['preT'],
                             np.zeros_like(FeDbthermodict['eneT']))
D0

Trange = np.linspace(300, 1200, 91)
Dlist=[]
for T in Trange:
    beta = 1./(kB*T)
    D = diffuser.diffusivity(FeDbthermodict['pre'], beta*FeDbthermodict['ene'], FeDbthermodict['preT'], beta*FeDbthermodict['eneT'])
    Dlist.append(D[0][0])

D = np.array(Dlist)
plt.semilogy(1/Trange,D)
plt.xlabel
