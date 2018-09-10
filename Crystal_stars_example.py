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
crys_stars = StarSet(cube,0,jset,pairs_mixed,2)
o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
db = dumbbell(0,o,np.array([1,0,0]))
test_list=[]
for l in crys_stars.starset[:crys_stars.mixedstartindex]:
    for state in l:
        if state.is_zero():
            test_list = l.copy()
#The solute-dumbbell pairs in test_list are easy to verify in a cubic lattice if they are related by symmetry
test_list
#Next - check out the mixed dumbbells
for l in crys_stars.starset[crys_stars.mixedstartindex:]:
    for state in l:
        print(state)
    print()

#Getting the omega_1 jumps
omega1_network_cube = crys_stars.jumpnetwork_omega1()[0]
len(omega1_network_cube)#there are this many symmetric jump lists
tot=0
omega1_1 = omega1_network_cube[1]
print(omega1_1[4])
len(omega1_1)
#test that this number is correct
#Without using hash so as to make sure that hashing is working correctly
db1 = dumbbell(0,o,np.array([-1,1,-1]))
db2 = dumbbell(0,o,np.array([-1,1,0]))
pair1=SdPair(0,np.array([0,0,0]),db1)
pair2=SdPair(0,np.array([0,0,0]),db2)
jmp = jump(pair1,pair2,1,1)

def inlist(jmp,jlist):
    return any(j==jmp for j in jlist)

jlist=[]
dbstates = jset[1]
for g in cube.G:
    jnew1 = jmp.gop(cube,0,g)
    db1new = dbstates.gdumb(g,jmp.state1.db)
    db2new = dbstates.gdumb(g,jmp.state2.db)
    state1new = SdPair(jnew1.state1.i_s,jnew1.state1.R_s,db1new[0])
    state2new = SdPair(jnew1.state2.i_s,jnew1.state2.R_s,db2new[0])
    jnew = jump(state1new,state2new,jnew1.c1*db1new[1],jnew1.c2*db2new[1])
    if not inlist(jnew,jlist):
        jlist.append(jnew)
        jlist.append(-jnew)
len(jlist)

#Now try another jumpset
omega1_3 = omega1_network_cube[3]
print(omega1_3[4])
db1 = dumbbell(0,o,np.array([-2,0,-1]))
db2 = dumbbell(0,o,np.array([-2,0,0]))
pair1=SdPair(0,np.array([0,0,0]),db1)
pair2=SdPair(0,np.array([0,0,0]),db2)
jmp = jump(pair1,pair2,1,1)
jlist=[]
for g in cube.G:
    jnew1 = jmp.gop(cube,0,g)
    db1new = dbstates.gdumb(g,jmp.state1.db)
    db2new = dbstates.gdumb(g,jmp.state2.db)
    state1new = SdPair(jnew1.state1.i_s,jnew1.state1.R_s,db1new[0])
    state2new = SdPair(jnew1.state2.i_s,jnew1.state2.R_s,db2new[0])
    jnew = jump(state1new,state2new,jnew1.c1*db1new[1],jnew1.c2*db2new[1])
    if not inlist(jnew,jlist):
        jlist.append(jnew)
        jlist.append(-jnew)
len(jlist)
len(omega1_3)
#See that the above two lengths are the same. This means that our jumps are being generated correctly.

#Now to test jumpnetwork_omega34
omega3_network_cube, omega4_network_cube = crys_stars.jumpnetwork_omega34(0.3,0.01,0.01,0.01)

#First test omega_3
omega3_1 = omega3_network_cube[1]
print(omega3_1[0])

#Now construct symmetric jumps to the jump above. Remember that the initial state is mixed, so I don't need to account for negative orientations.
db1 = dumbbell(0,np.array([0.,0.,0.126]),np.array([0,0,0]))
db2 = dumbbell(0,np.array([0.,0.,-0.126]),np.array([0,1,0]))
pair1=SdPair(0,np.array([0,0,0]),db1)
pair2=SdPair(0,np.array([0,0,0]),db2)
jmp = jump(pair1,pair2,-1,-1)
jlist=[]
for g in cube.G:
    jnew = jmp.gop(cube,0,g)
    db2new = dbstates.gdumb(g,jmp.state2.db)
    state2new = SdPair(jnew.state2.i_s,jnew.state2.R_s,db2new[0])
    jnew = jump(jnew.state1,state2new,-1,jnew.c2*db2new[1])
    if not inlist(jnew,jlist):
        jlist.append(jnew)
len(jlist)
len(omega3_1)
#Notice that we are not using hashes in any of these checks. This ensures that our hash values in the original jump functions are working correctly.

#Next, test omega_4
omega4_1 = omega4_network_cube[1]
print(omega4_1[0])
db1 = dumbbell(0,np.array([0.126,0.,0.]),np.array([0,-1,0]))
db2 = dumbbell(0,np.array([-0.126,0.,0.]),np.array([0,0,0]))
pair1=SdPair(0,np.array([0,0,0]),db1)
pair2=SdPair(0,np.array([0,0,0]),db2)
jmp = jump(pair1,pair2,-1,-1)
jlist=[]
for g in cube.G:
    jnew = jmp.gop(cube,0,g)
    db1new = dbstates.gdumb(g,jmp.state1.db)
    state1new = SdPair(jnew.state1.i_s,jnew.state1.R_s,db1new[0])
    jnew = jump(state1new,jnew.state2,jnew.c1*db1new[1],-1)
    if not inlist(jnew,jlist):
        jlist.append(jnew)
len(jlist)
len(omega4_1)

#Next test jumpnetwork_omega2
omega2_network_cube = crys_stars.jumpnetwork_omega2(0.3,0.01,0.01)
omega2_1 = omega2_network_cube[1]
print(omega2_1[0])

db1 = dumbbell(0,np.array([0.126,0.,0.]),np.array([0,0,0]))
db2 = dumbbell(0,np.array([0.,-0.126,0.]),np.array([0,1,0]))
pair1=SdPair(0,np.array([0,0,0]),db1)
pair2=SdPair(0,np.array([0,1,0]),db2)
jmp = jump(pair1,pair2,1,1)
jlist=[]
for g in cube.G:
    jnew = jmp.gop(cube,0,g)
    if not inlist(jnew,jlist):
        #create the negative jump
        p11 = jnew.state1
        p21 = jnew.state2
        p1neg = SdPair(p21.i_s,np.array([0,0,0]),dumbbell(p21.db.i,p21.db.o,np.array([0,0,0])))
        p2neg = SdPair(p11.i_s,-p21.db.R,dumbbell(p11.db.i,p11.db.o,-p21.db.R))
        jnewneg = jump(p1neg,p2neg,1,1)
        #add both the jump and its negative
        jlist.append(jnew)
        jlist.append(jnewneg)
len(jlist)
len(omega2_1)
