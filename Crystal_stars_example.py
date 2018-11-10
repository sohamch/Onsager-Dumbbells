import numpy as np
import onsager.crystal as crystal
from stars import *
from test_structs import *
from states import *

#How to create a crystal star
#1. Determine orientation families. Here we consider only one for both pure and mixed.
famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
family = [famp0]
#2. Create state information containers
pdbcontainer = dbStates(cube,0,family)
mdbcontainer = mStates(cube,0,family)
#Extract omega0 and omega2 jumps
jset0 = pdbcontainer.jumpnetwork(0.3,0.01,0.01)
jset2 = mdbcontainer.jumpnetwork(0.3,0.01,0.01)
#create starset
crys_stars = StarSet(pdbcontainer,mdbcontainer,jset0,jset2,2)

o = np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126
db = dumbbell(0,o,np.array([1,0,0]))
test_list=[]
for l in crys_stars.starset[:crys_stars.mixedstartindex]:
    for state in l:
        if state.is_zero():
            test_list = l.copy()
#The solute-dumbbell pairs in test_list are easy to verify in a cubic lattice. Should be just three spectator states
test_list
#Next - check out the mixed dumbbells - should be six mixed dumbbells at the origin.
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
#Consider the jump printed above printed above.
#We will construct symmetric jumps without using hashes to make sure hashes are working correctly
jmp = omega1_1[4]

def inlist(jmp,jlist):
    return any(j==jmp for j in jlist)

jlist=[]

for g in cube.G:
    jnew1 = jmp.gop(cube,0,g)
    db1new = pdbcontainer.gdumb(g,jmp.state1.db)
    db2new = pdbcontainer.gdumb(g,jmp.state2.db)
    state1new = SdPair(jnew1.state1.i_s,jnew1.state1.R_s,db1new[0])
    state2new = SdPair(jnew1.state2.i_s,jnew1.state2.R_s,db2new[0])
    jnew = jump(state1new,state2new,jnew1.c1*db1new[1],jnew1.c2*db2new[1])
    if not inlist(jnew,jlist):
        jlist.append(jnew)
        jlist.append(-jnew)
len(jlist)
len(omega1_1)
#Now Let us try another jumpset
omega1_3 = omega1_network_cube[3]
print(omega1_3[4])
#Consider the jump printed above printed above.
#We will construct symmetric jumps without using hashes to make sure hashes are working correctly
jmp = omega1_3[4]
jlist=[]
for g in cube.G:
    jnew1 = jmp.gop(cube,0,g)
    db1new = pdbcontainer.gdumb(g,jmp.state1.db)
    db2new = pdbcontainer.gdumb(g,jmp.state2.db)
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
omega43_cube_all,omega3_network_cube,omega4_network_cube = crys_stars.jumpnetwork_omega34(0.3,0.01,0.01,0.01)
#omega_34_all has the usual organization of jumps and their negatives in the same list
#For convenience, when necessary, they have been seperated too.

#First test omega_3
omega3_1 = omega3_network_cube[1]
print(omega3_1[0])

#Now construct symmetric jumps to the jump above.
#Remember that the initial state is mixed, so we don't need to account for negative orientations.
jmp = omega3_1[0]
jlist=[]
for g in cube.G:
    jnew = jmp.gop(cube,0,g)
    db2new = pdbcontainer.gdumb(g,jmp.state2.db)
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

jmp = omega4_1[0]
jlist=[]
for g in cube.G:
    jnew = jmp.gop(cube,0,g)
    db1new = pdbcontainer.gdumb(g,jmp.state1.db)
    state1new = SdPair(jnew.state1.i_s,jnew.state1.R_s,db1new[0])
    jnew = jump(state1new,jnew.state2,jnew.c1*db1new[1],-1)
    if not inlist(jnew,jlist):
        jlist.append(jnew)
len(jlist)
len(omega4_1)

#Next test jumpnetwork_omega2
omega2_network_cube = crys_stars.jumpnetwork_omega2
omega2_1 = omega2_network_cube[1]
print(omega2_1[0])
jmp = omega2_1[0]
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
