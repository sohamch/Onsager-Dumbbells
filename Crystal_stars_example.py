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

crys_stars = StarSet(cube,0,jset,pairs_pure,2)
len(crys_stars.starset[0])
o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
db = dumbbell(0,o,np.array([1,0,0]))
for l in crys_stars.starset:
    for state in l:
        if state.db==db or state.db==-db:
            test_list = l.copy()
#These solute-dumbbell pairs are easy to verify in a cubic lattice if they are related by symmetry
test_list
