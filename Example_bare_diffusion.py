import numpy as np
import onsager.crystal as crystal
from states import *
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from Onsager_calc_db import BareDumbbell

kB = physical_constants['Boltzmann constant in eV/K'][0]
latt = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])*0.55
DC_Si = crystal.Crystal(latt,[[np.array([0.,0.,0.]),np.array([0.25,0.25,0.25])]],["Si"])

#Now make the [110] dumbbell states
#Silicon atomic diameter - 0.2 nm - that is the length of the dumbbell.
famp0 = [np.array([1.,1.,0.])/np.sqrt(2)*0.2]
family = [famp0]
pdbcontainer = dbStates(DC_Si,0,family)

len(pdbcontainer.iorlist)

jset,jind = pdbcontainer.jumpnetwork(0.4,0.01,0.01)

#Lets see if cubic symmetry is there
o1 = np.array([1.,-1.,0.])*0.2/np.sqrt(2)
if any(np.allclose(-o1,o) for i,o in pdbcontainer.iorlist):
    o1 = -o1+0.

db1 = dumbbell(0,o1,np.array([0,0,0]))
db2 = dumbbell(0,o1,np.array([0,0,1]))

jmp = jump(db1,db2,1,1)
jtest=[]
for jl in jset:
    for j in jl:
        if j==jmp:
            jtest.append(jl)
len(jtest)
len(jtest[0])

###fcc - should have half the number of same type of jumps - checked the jumps by hand
fcc = crystal.Crystal.FCC(0.55)
pdbcontainer_fcc = dbStates(fcc,0,family)
jset_fcc,jind_fcc = pdbcontainer_fcc.jumpnetwork(0.4,0.01,0.01)
o1 = np.array([1.,-1.,0.])*0.2/np.sqrt(2)
if any(np.allclose(-o1,o) for i,o in pdbcontainer_fcc.iorlist):
    o1 = -o1.copy()

print(o1)

db1 = dumbbell(0,o1,np.array([0,0,0]))
db2 = dumbbell(0,o1,np.array([0,0,1]))
jmp = jump(db1,db2,1,1)
jtest=[]
for jl in jset_fcc:
    for j in jl:
        if j==jmp:
            jtest.append(jl)
len(jtest)
len(jtest[0])
