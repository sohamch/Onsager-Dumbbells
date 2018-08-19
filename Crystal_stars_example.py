import numpy as np
import onsager.crystal as crystal
from jumpnet3 import *
from stars import *
from test_structs import *
from states import *
from gensets import *
famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
family = [famp0]
pairs_pure = genpuresets(cube,0,family)

jset = purejumps(cube,0,pairs_pure,0.3,0.01,0.01)
pairs_mixed = pairs_pure.copy()
#Information about pure orientations are already taken out of jumpnetwork 'jset'.
#We need to pass in the mixed (site,orientation) list to create the mixed dumbbell stars
crys_stars = StarSet(cube,0,jset,pairs_mixed,2)
len(crys_stars.starset[0])
o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
db = dumbbell(0,o,np.array([1,0,0]))
for l in crys_stars.starset:
    for state in l:
        if state.db==db or state.db==-db:
            test_list = l.copy()
#The solute-dumbbell pairs in test_list are easy to verify in a cubic lattice if they are related by symmetry

#Next - check out the mixed dumbbells
for l in crys_stars.starset[crys_stars.mixedstartindex:]:
    for state in l:
        print(state)
    print()

#Getting the omega_1 jumps
omega1_network = crys_stars.jumpnetwork_omega1()
len(omega1_network)#there are this many symmetric jump lists
tot=0
for l in omega1_network:
    tot += len(l)
tot #there are this many total number of omega_1 jumps

# verify that we have the correct symmetric orientations of the jumps
print(len(omega1_network[0]))
print(omega1_network[0][0])
