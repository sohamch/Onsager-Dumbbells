import numpy as np
import onsager.crystal as crystal
from gensets import *
from test_structs import *
from states import *
from representations import *
# from states import *
famp0 = [np.array([1.,1.,0.]),np.array([1.,0.,0.])]
famp12 = [np.array([1.,1.,1.]),np.array([1.,1.,0.])]
family = [famp0,famp12]
pairs_pure = genpuresets(tet2,0,family)
len(pairs_pure)
thrange=1
sdpairs = genPairSets(tet2,0,pairs_pure,thrange)

# db=dumbbell(0,np.array([0., 1., 0.]), np.array([1, 1, 0]))
# pair = SdPair(0,np.array([0, 0, 0]),db)
# pairlist = []
# pairlist.append(pair)
# for g in tet2.G:
#     newpair = pair.gop(tet2,0,g)
#     if not any(pair1==newpair for pair1 in pairlist):
#         pairlist.append(newpair)
# pairlist
#
# dblist = []
# dblist.append(db)
# for g in tet.G:
#     newdb = db.gop(tet,0,g)
#     if not any(db1==newdb or db1==-newdb for db1 in dblist):
#         dblist.append(newdb)
#         print (crystal.GroupOp.optype(g.rot))
# dblist
#
# or_1 = np.array([0.,1.,0.])
# db1 = dumbbell(0,or_1,np.array([1.,0.,0.]))
# pair1 = SdPair(0,np.array([1.,0.,0.]),db1)
# pair_list=[]
# pair_list.append(pair1)
# for g in tet.G:
#     pairn = pair1.gop(tet,0,g)
#     if not any(pair==pairn for pair in pair_list):
#         pair_list.append(pairn)
# len(pair_list)
# pair_list
#
# dblist = []
# dblist.append(db1)
# for g in tet.G:
#     newdb = db.gop(tet,0,g)
#     if not any(db==newdb for db in dblist):
#         dblist.append(newdb)
# dblist
dblist=[]
db=dumbbell(1, np.array([-1., -1.,  0.]), np.array([-1, -1,  0]))
dblist.append(db)
for g in tet2.G:
    newdb = db.gop(tet2,0,g)
    if not any(db1==newdb or db1==-newdb for db1 in dblist):
        dblist.append(newdb)
        # print (crystal.GroupOp.optype(g.rot))
dblist
