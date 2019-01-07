import numpy as np
import onsager.crystal as crystal
from states import *
from representations import *
# from test_structs import *

Mg = crystal.Crystal.HCP(0.3294,chemistry=["Mg"])

famp0 = [np.array([1.,0.,0.])*0.145]
family = [famp0]
pdbcontainer = dbStates(Mg,0,family)

for tup in pdbcontainer.iorlist:
    print(tup)

#in-place modifying iorlist
lst=[]
for i,o in pdbcontainer.iorlist:
    onew = np.absolute(o).copy()
    lst.append((i,onew))

pdbcontainer.iorlist = lst

pdbcontainer.iorlist

jset,jind,dxcount = pdbcontainer.jumpnetwork(0.45,0.01,0.01)
len(jind)
