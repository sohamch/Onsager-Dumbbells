import numpy as np
import onsager.crystal as crystal
from stars import *
from test_structs import *
from states import *
import time
def inlist(jmp,jlist):
    return any(j==jmp for j in jlist)
#How to create a crystal star
#1. Determine orientation families. Here we consider only one for both pure and mixed.
Mg = crystal.Crystal.HCP(0.3294,chemistry=["Mg"])
famp0 = [np.array([1.,0.,0.])*0.145]
family = [famp0]
#2. Create state information containers
pdbcontainer = dbStates(Mg,0,family)
mdbcontainer = mStates(Mg,0,family)
#Extract omega0 and omega2 jumps
jset0 = pdbcontainer.jumpnetwork(0.45,0.01,0.01)
jset2 = mdbcontainer.jumpnetwork(0.45,0.01,0.01)
#create starset
crys_stars = StarSet(pdbcontainer,mdbcontainer,jset0,jset2,1)

#Getting the omega_1 jumps
start = time.time()
omega1_network_hcp = crys_stars.jumpnetwork_omega1()[0]
end=time.time()
taken = end-start
print(taken)
len(omega1_network_hcp)#there are this many symmetric jump lists
tot=0
omega1_1 = omega1_network_hcp[1]
print(omega1_1[4])
#Consider the jump printed above printed above.
#We will construct symmetric jumps without using hashes to make sure hashes are working correctly
jmp = omega1_1[4]
jlist=[]

for g in Mg.G:
    jnew1 = jmp.gop(Mg,0,g)
    db1new = pdbcontainer.gdumb(g,jmp.state1.db)
    db2new = pdbcontainer.gdumb(g,jmp.state2.db)
    state1new = SdPair(jnew1.state1.i_s,jnew1.state1.R_s,db1new[0])-jnew1.state1.R_s #shift the states back to the origin
    state2new = SdPair(jnew1.state2.i_s,jnew1.state2.R_s,db2new[0])-jnew1.state2.R_s
    jnew = jump(state1new,state2new,jnew1.c1*db1new[1],jnew1.c2*db2new[1])
    if not inlist(jnew,jlist):
        jlist.append(jnew)
        jlist.append(-jnew)
len(jlist)
len(omega1_1)
count=0
for j in jlist:
    for j1 in omega1_1:
        if j==j1:
            count+=1
            break

count
#This is to confirm that both lists contain the same jump
#Now Let us try another jumpset
omega1_3 = omega1_network_hcp[3]
print(omega1_3[4])
#Consider the jump printed above printed above.
#We will construct symmetric jumps without using hashes to make sure hashes are working correctly
jmp = omega1_3[4]
jlist=[]
for g in Mg.G:
    jnew1 = jmp.gop(Mg,0,g)
    db1new = pdbcontainer.gdumb(g,jmp.state1.db)
    db2new = pdbcontainer.gdumb(g,jmp.state2.db)
    state1new = SdPair(jnew1.state1.i_s,jnew1.state1.R_s,db1new[0])-jnew1.state1.R_s
    state2new = SdPair(jnew1.state2.i_s,jnew1.state2.R_s,db2new[0])-jnew1.state1.R_s
    jnew = jump(state1new,state2new,jnew1.c1*db1new[1],jnew1.c2*db2new[1])
    if not inlist(jnew,jlist):
        jlist.append(jnew)
        jlist.append(-jnew)
len(jlist)
len(omega1_3)
#See that the above two lengths are the same. This means that our jumps are being generated correctly.
count=0
for j in jlist:
    for j1 in omega1_3:
        if j==j1:
            count+=1
            break
count
#Now to test jumpnetwork_omega34
omega43_hcp_all,omega4_network_hcp,omega3_network_hcp = crys_stars.jumpnetwork_omega34(0.45,0.01,0.01,0.01)
#omega_34_all has the usual organization of jumps and their negatives in the same list
#For convenience, when necessary, they have been seperated too.

#First test omega_3
omega3_1 = omega3_network_hcp[1]
print(omega3_1[0])

#Now construct symmetric jumps to the jump above.
#Remember that the initial state is mixed, so we don't need to account for negative orientations.
jmp = omega3_1[0]
jlist=[]
for g in Mg.G:
    jnew = jmp.gop(Mg,0,g)
    db2new = pdbcontainer.gdumb(g,jmp.state2.db)
    state2new = SdPair(jnew.state2.i_s,jnew.state2.R_s,db2new[0])-jnew.state2.R_s
    jnew = jump(jnew.state1-jnew.state1.R_s,state2new,-1,jnew.c2*db2new[1])
    if not inlist(jnew,jlist):
        jlist.append(jnew)
len(jlist)
len(omega3_1)
print(omega3_1[0])

count=0
for j in jlist:
    for j1 in omega3_1:
        if j==j1:
            count+=1
count
#Notice that we are not using hashes in any of these checks. This ensures that our hash values in the original jump functions are working correctly.

#Next, test omega_4
omega4_1 = omega4_network_hcp[1]
print(omega4_1[0])

jmp = omega4_1[0]
jlist=[]
for g in Mg.G:
    jnew = jmp.gop(Mg,0,g)
    db1new = pdbcontainer.gdumb(g,jmp.state1.db)
    state1new = SdPair(jnew.state1.i_s,jnew.state1.R_s,db1new[0])-jnew.state1.R_s
    jnew = jump(state1new,jnew.state2-jnew.state2.R_s,jnew.c1*db1new[1],-1)
    if not inlist(jnew,jlist):
        jlist.append(jnew)
len(jlist)
len(omega4_1)

#We have to check that the same jumps are there in both the lists.
count=0
for j in jlist:
    for j1 in omega4_1:
        if j==j1:
            count+=1
            break
count
