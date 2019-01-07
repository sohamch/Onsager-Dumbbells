import numpy as np
import onsager.crystal as crystal
from states import *
from representations import *
# from test_structs import *

Mg = crystal.Crystal.HCP(0.3294,chemistry=["Mg"])
nn2 = 0.3294*8./3.
famp0 = [np.array([1.,0.,0.])*0.145]
family = [famp0]
pdbcontainer = dbStates(Mg,0,family)

pdbcontainer.iorlist

jset,jind = pdbcontainer.jumpnetwork(nn2+0.01,0.01,0.01)
len(jind)
o = np.array([0.145,0.,0.])
if any(np.allclose(-o,o1)for i,o1 in pdbcontainer.iorlist):
    o = -o+0.
o
db1 = dumbbell(0,o,np.array([0,0,0],dtype=int))
db2 = dumbbell(1,o,np.array([0,0,0],dtype=int))
testjump = jump(db1,db2,1,1)
count=0
testlist=[]
for jl in jset:
    for j in jl:
        if j==testjump:
            count+=1
            testlist=jl
count
len(testlist)
for j in testlist:
    print(j)
    print()
