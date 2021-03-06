import numpy as np
import onsager.crystal as crystal
from representations import *
from collision import *


def disp(dbcontainer, obj1, obj2):
    """
    Computes the transport vector for the initial and final states of a jump
    param:
        dbcontainer - dumbbell states container.
        obj1,obj2 - the initial and final state objects of a jump
        Return - displacement when going from obj1 to obj2
    """
    crys, chem = dbcontainer.crys, dbcontainer.chem

    if isinstance(obj1, dumbbell):
        (i1, i2) = (dbcontainer.iorlist[obj1.iorind][0], dbcontainer.iorlist[obj2.iorind][0])
        (R1, R2) = (obj1.R, obj2.R)
    else:
        (i1, i2) = (dbcontainer.iorlist[obj1.db.iorind][0], dbcontainer.iorlist[obj2.db.iorind][0])
        (R1, R2) = (obj1.db.R, obj2.db.R)

    return crys.unit2cart(R2, crys.basis[chem][i2]) - crys.unit2cart(R1, crys.basis[chem][i1])


def disp4(pdbcontainer, mdbcontainer, obj1, obj2):
    """
    Computes the transport vector for the initial and final states of an associative jump
    param:
        dbcontainer - dumbbell states container.
        obj1,obj2 - the initial and final state objects of a jump - must be of Omega4 type
        Return - displacement when going from obj1 to obj2
    """
    true_crys = np.allclose(pdbcontainer.crys.lattice, mdbcontainer.crys.lattice, atol=pdbcontainer.crys.threshold)

    true_basis_len = len(pdbcontainer.crys.basis) == len(mdbcontainer.crys.basis)
    if not (true_basis_len and true_crys):
        raise TypeError("Different crystal structures entered for pure and dumbbell states")

    true_site_len = all([len(chem1) == len(chem2) for chem1, chem2 in
                         zip(mdbcontainer.crys.basis, pdbcontainer.crys.basis)])
    if not true_site_len:
        raise TypeError("Different site numbers entered for pure and dumbbell states for same chemistries")

    true_site_locs = all([[np.allclose(arr1, arr2) for arr1, arr2 in zip(chem1, chem2)]for chem1, chem2 in
                          zip(mdbcontainer.crys.basis, pdbcontainer.crys.basis)])

    if not true_site_locs:
        raise TypeError("basis sites are at different locations for the two containers.")

    crys, chem = pdbcontainer.crys, pdbcontainer.chem
    (i1, i2) = (pdbcontainer.iorlist[obj1.db.iorind][0], mdbcontainer.iorlist[obj2.db.iorind][0])
    (R1, R2) = (obj1.db.R, obj2.db.R)

    return crys.unit2cart(R2, crys.basis[chem][i2]) - crys.unit2cart(R1, crys.basis[chem][i1])


# Create pure dumbbell states
class dbStates(object):
    """
    Class to generate all possible dumbbell configurations for given basis sites.
    Make a "supercrystal" with the states as the dumbbell configurations, capable of handling symmetry operations.
    This is mainly to automate group operations on jumps (to return correct dumbbell states)
    """
    def __init__(self, crys, chem, family):
        if not isinstance(family, list):
            raise TypeError("Enter the families as a list of lists")

        self.crys = crys
        self.chem = chem
        self.family = family
        # make the dumbbell states, change the indexmap of the grouops and store original groupops in G_crys
        self.iorlist = self.genpuresets()
        self.G, self.G_crys, = self.makeDbGops(self.crys, self.chem, self.iorlist)
        self.symorlist, self.symIndlist = self.gensymset() # make this an indexed list
        # Store both iorlist and symorlist so that we can compare them later if needed.
        self.threshold = crys.threshold
        self.invmap = self.invmapping(self.symIndlist)
        # Invmap says which (i, or) pair is present in which symmetric (i, or) list

    @staticmethod
    def invmapping(symindlist):
        # Sanity checks between iorlist and symorlist is performed during testing
        invmap = np.zeros(sum([len(lst) for lst in symindlist]), dtype=int)
        for symind, symlist in enumerate(symindlist):
            for st_idx in symlist:
                invmap[st_idx] = symind
        return invmap

    def genpuresets(self):
        """
        generates complete (i,or) set from given family of orientations, neglects negatives, since pure
        """
        if not isinstance(self.family, list):
            raise TypeError("Enter the families as a list of lists")
        for i in self.family:
            if not isinstance(i, list):
                raise TypeError("Enter the families for each site as a list of np arrays")
            for j in i:
                if not isinstance(j, np.ndarray):
                    raise TypeError("Enter individual orientation families as numpy arrays")

        def inlist(tup, lis):
            return any(tup[0] == x[0] and np.allclose(tup[1], x[1], atol=1e-8) for x in lis)

        def negOrInList(o, lis):
            return any(np.allclose(o + tup[1], 0, atol=1e-8) for tup in lis)

        sitelist = self.crys.sitelist(self.chem)
        if not len(self.family) == len(sitelist):
            raise TypeError("Orientations must be given for every Wyckoff set. Enter [0,0,0] for Wyckoff sets that "
                            "don't have a dumbbell")
        # Get the Wyckoff sets
        iorlist = []
        for wyckind, wycksites in enumerate(sitelist):
            orlist = self.family[wyckind]  # Get the orientations allowed on the given Wyckoff set.
            if np.allclose(orlist[0], np.zeros(self.crys.dim)):
                # If zero vector is entered, then that means this set does not have dumbbells.
                continue
            site = wycksites[0]  # Get the representative site of the Wyckoff set.
            newlist = []
            for o in orlist:
                for g in self.crys.G:
                    R, (ch, i_new) = self.crys.g_pos(g, np.zeros(self.crys.dim), (self.chem, site))
                    o_new = self.crys.g_direc(g, o)
                    if not (inlist((i_new, o_new), iorlist) or inlist((i_new, -o_new), iorlist)):
                        if negOrInList(o_new, iorlist):
                            o_new = -o_new + 0.
                        iorlist.append((i_new, o_new))
        return iorlist

    def makeDbGops(self, crys, chem, iorlist):
        G = []
        G_crys = {}
        for g in crys.G:
            # Will have indexmap for each groupop
            indexmap = []
            for (i, o) in iorlist:
                # Need the elements of indexmap
                R, (ch, i_new) = crys.g_pos(g, np.zeros(self.crys.dim), (chem, i))
                o_new = crys.g_direc(g, o)
                count = 0
                for idx2, (i2, o2) in enumerate(iorlist):
                    if i2 == i_new and (np.allclose(o2, o_new, atol=crys.threshold) or
                                        np.allclose(o2, -o_new, atol=crys.threshold)):
                        indexmap.append(idx2)
                        break
                        
            gdumb = crystal.GroupOp(g.rot, g.trans, g.cartrot, tuple([tuple(indexmap)]))
            G.append(gdumb)
            G_crys[gdumb] = g

        return frozenset(G), G_crys

    def gensymset(self):
        """
        Takes in a flat list of (i,or) pairs and groups them according to symmetry
        """

        # We'll take advantage of the gdumbs we have created
        symIorList = []
        symIndlist = []
        allIndlist = set([])
        for idx, (i, o) in enumerate(self.iorlist):
            if idx in allIndlist:
                continue
            newlist=[]
            newindlist = []
            for gdumb in self.G:
                idxnew = gdumb.indexmap[0][idx]
                if idxnew in allIndlist:
                    continue
                allIndlist.add(idxnew)
                newindlist.append(idxnew)
                newlist.append(self.iorlist[idxnew])
            symIndlist.append(newindlist)
            symIorList.append(newlist)

        return symIorList, symIndlist

    def gflip(self, gdumb, idx):
        """
        Takes in a (i, or) index, idx, applies a group operation and returns -1 if the groupop reverses the orientation
        from that of the destination index (gdumb.indexmap[0][idx]), +1 if not
        """
        inew, onew = self.iorlist[gdumb.indexmap[0][idx]]
        if np.allclose(onew, -np.dot(gdumb.cartrot, self.iorlist[idx][1]), atol=self.crys.threshold):
            return -1
        return 1

    def jumpnetwork(self, cutoff, solv_solv_cut, closestdistance):
        """
        Makes a jumpnetwork of pure dumbbells within a given distance to be used for omega_0
        and to create the solute-dumbbell stars.
        Parameters:
            cutoff - maximum jump distance
            solv_solv_cut - minimum allowable distance between two solvent atoms - to check for collisions
            closestdistance - minimum allowable distance to check for collisions with other atoms. Can be a single
            number or a list (corresponding to each sublattice)
        Returns:
            jumpnetwork - the symmetrically grouped jumpnetworks (db1,db2,c1,c2)
            jumpindices - the jumpnetworks with dbs in pair1 and pair2 indexed to iorset -> (i,j,dx)
        """
        crys, chem, iorlist = self.crys, self.chem, self.iorlist

        def getjumps(j, jumpset, dx):
            "Does the symmetric list construction for an input jump and an existing jumpset"

            # If the jump has not already been considered, check if it leads to collisions.
            jlist = []
            jindlist = []
            db1 = j.state1
            db2 = j.state2
            for gdumb in self.G:
                # Get the new dumbbells
                db1new, mul1 = db1.gop(self, gdumb, pure=True)
                db2new, mul2 = db2.gop(self, gdumb, pure=True)
                R_ref = db1new.R.copy()
                db2new = db2new - R_ref
                db1new = db1new - R_ref

                jnew = jump(db1new, db2new, j.c1 * mul1, j.c2 * mul2)  # Check this part
                dx = disp(self, jnew.state1, jnew.state2)

                db1newneg = dumbbell(jnew.state2.iorind, jnew.state1.R)
                db2newneg = dumbbell(jnew.state1.iorind, -jnew.state2.R)
                jnewneg = jump(db1newneg, db2newneg, jnew.c2, jnew.c1)

                if not np.allclose(db1newneg.R, np.zeros(self.crys.dim), atol=1e-8):
                    raise RuntimeError("Initial state not at origin")

                if np.allclose(dx, np.zeros(self.crys.dim)):
                    # First, make the equivalent rotation jump
                    jnew_equiv = jump(db1new, db2new, -jnew.c1, -jnew.c2)
                    # Check if the rotation jump has also been taken into account.
                    if not jnew in jumpset and not jnew_equiv in jumpset:
                        jlist.append(jnew)
                        jlist.append(jnewneg)
                        jindlist.append(((jnew.state1.iorind, jnew.state2.iorind), dx))
                        jindlist.append(((jnewneg.state1.iorind, jnewneg.state2.iorind), -dx))
                        jumpset.add(jnew)
                        jumpset.add(jnewneg)
                else:
                    if not jnew in jumpset:
                        # add both the jump and it's negative
                        jlist.append(jnew)
                        jlist.append(jnewneg)
                        jindlist.append(((jnew.state1.iorind, jnew.state2.iorind), dx))
                        jindlist.append(((jnewneg.state1.iorind, jnewneg.state2.iorind), -dx))
                        jumpset.add(jnew)
                        jumpset.add(jnewneg)

            return jlist, jindlist

        nmax = [int(np.round(np.sqrt(cutoff ** 2 / crys.metric[i, i]))) + 1 for i in range(self.crys.dim)]

        if self.crys.dim == 2:
            Rvects = [np.array([n0, n1]) for n0 in range(-nmax[0], nmax[0] + 1)
                      for n1 in range(-nmax[1], nmax[1] + 1)]

        else:
            Rvects = [np.array([n0, n1, n2]) for n0 in range(-nmax[0], nmax[0] + 1)
                      for n1 in range(-nmax[1], nmax[1] + 1)
                      for n2 in range(-nmax[2], nmax[2] + 1)]

        jumplist = []
        jumpindices = []
        jumpset = set([])
        for R in Rvects:
            for i, tup1 in enumerate(iorlist):
                for f, tup2 in enumerate(iorlist):
                    db1 = dumbbell(i, np.zeros(self.crys.dim, dtype=int))
                    db2 = dumbbell(f, R)
                    if db1 == db2:  # catch the diagonal case
                        continue
                    dx = disp(self, db1, db2)
                    if np.dot(dx, dx) > cutoff * cutoff:
                        continue
                    for c1 in [-1, 1]:
                        # Check if the jump is a rotation - 180 degree rotations end up in the same state
                        # they are not considered
                        if np.allclose(np.dot(dx, dx), 0., atol=crys.threshold):
                            j = jump(db1, db2, c1, 1)
                            j_equiv = jump(db1, db2, -c1, -1)
                            # Also check if the equivalent rotation has been considered.
                            if j in jumpset or j_equiv in jumpset:
                                continue
                            if collision_self(self, None, j, solv_solv_cut, solv_solv_cut) or\
                                    collision_others(self, None, j, closestdistance):
                                continue
                            jlist, jindlist = getjumps(j, jumpset, dx)
                            jumplist.append(jlist)
                            jumpindices.append(jindlist)
                            continue
                        for c2 in [-1, 1]:
                            j = jump(db1, db2, c1, c2)
                            if j in jumpset:  # no point doing anything else if the jump has already been considered
                                continue
                            if collision_self(self, None, j, solv_solv_cut, solv_solv_cut) or\
                                    collision_others(self, None, j, closestdistance):
                                continue
                            jlist, jindlist = getjumps(j, jumpset, dx)
                            jumplist.append(jlist)
                            jumpindices.append(jindlist)
        return jumplist, jumpindices

    def getIndex(self, t):
        """
        :param i: input site index
        :param o: input orientation
        (i, o) contained in t
        :return: idx (integer) - the index of (i, o) in the iorlist, if it exists.
        """
        for idx,tup in enumerate(self.iorlist):
            if t[0]==tup[0] and (np.allclose(t[1], tup[1], atol=self.crys.threshold) or
                                 np.allclose(t[1], -tup[1], atol=self.crys.threshold)):
                return idx
        raise ValueError("The given site orientation pair {} is not present in the container".format(t))

    def db2ind(self, db):
        """
        :param db: dumbbell object
        :return:
        """
        if not isinstance(db, dumbbell):
            raise TypeError("Input object must be dumbbell")
        i = self.iorlist[db.iorind][0]
        o = self.iorlist[db.iorind][1]

        return self.getIndex((i,o))

class mStates(object):

    def __init__(self, crys, chem, family):
        if not isinstance(family, list):
            raise TypeError("Enter the families as a list of lists")
        # Should I just inherit dbStates here?
        self.crys = crys
        self.chem = chem
        self.family = family
        # make the dumbbell states, change the indexmap of the grouops and store original groupops in G_crys
        self.iorlist = self.genmixedsets()
        self.G, self.G_crys, = self.makeDbGops(self.crys, self.chem, self.iorlist)
        self.symorlist, self.symIndlist = self.gensymset()  # make this an indexed list
        # Store both iorlist and symorlist so that we can compare them later if needed.
        self.threshold = crys.threshold
        self.invmap = self.invmapping(self.symIndlist)

        # Invmap says which (i, or) pair is present in which symmetric (i, or) list

    @staticmethod
    def invmapping(symindlist):
        # Sanity checks between iorlist and symorlist is performed during testing
        invmap = np.zeros(sum([len(lst) for lst in symindlist]), dtype=int)
        for symind, symlist in enumerate(symindlist):
            for st_idx in symlist:
                invmap[st_idx] = symind
        return invmap

    def genmixedsets(self):
        """
        function to generate (i,or) list for mixed dumbbells.
        """
        crys, chem, family = self.crys, self.chem, self.family
        if not isinstance(family, list):
            raise TypeError("Enter the families as a list of lists")
        for i in family:
            if not isinstance(i, list):
                raise TypeError("Enter the families for each site as a list of numpy arrays")
            for j in i:
                if not isinstance(j, np.ndarray):
                    raise TypeError("Enter individual orientation families as numpy arrays")

        def inlist(tup, lis):
            return any(tup[0] == x[0] and np.allclose(tup[1], x[1], atol=1e-8) for x in lis)

        sitelist = crys.sitelist(chem)
        # Get the Wyckoff sets
        pairlist = []
        for i, wycksites in enumerate(sitelist):
            orlist = family[i]
            site = wycksites[0]
            newlist = []
            for o in orlist:
                for g in crys.G:
                    R, (ch, i_new) = crys.g_pos(g, np.zeros(self.crys.dim), (chem, site))
                    o_new = crys.g_direc(g, o)
                    if not inlist((i_new, o_new), pairlist):
                        pairlist.append((i_new, o_new))
        return pairlist

    def makeDbGops(self, crys, chem, iorlist):
        G = []
        G_crys = {}
        for g in crys.G:
            # Will have indexmap for each groupop
            indexmap = []
            for (i, o) in iorlist:
                # Need the elements of indexmap
                R, (ch, i_new) = crys.g_pos(g, np.zeros(self.crys.dim), (chem, i))
                o_new = crys.g_direc(g, o)
                count = 0
                for idx2, (i2, o2) in enumerate(iorlist):
                    if i2 == i_new and np.allclose(o2, o_new, atol=crys.threshold):
                        indexmap.append(idx2)
                        break
            gdumb = crystal.GroupOp(g.rot, g.trans, g.cartrot, tuple([tuple(indexmap)]))
            G.append(gdumb)
            G_crys[gdumb] = g
        return frozenset(G), G_crys

    def gensymset(self):
        """
        Takes in a flat list of (i,or) pairs and groups them according to symmetry
        """
        symIorList = []
        symIndlist = []
        allIndlist = set([])
        for idx, (i, o) in enumerate(self.iorlist):
            if idx in allIndlist:
                continue
            newlist = []
            newindlist = []
            for gdumb in self.G:
                idxnew = gdumb.indexmap[0][idx]
                if idxnew in allIndlist:
                    continue
                allIndlist.add(idxnew)
                newindlist.append(idxnew)
                newlist.append(self.iorlist[idxnew])
            symIndlist.append(newindlist)
            symIorList.append(newlist)

        return symIorList, symIndlist

    def jumpnetwork(self, cutoff, solt_solv_cut, closestdistance):
        """
        Makes a jumpnetwork of mixed dumbbells within a given distance to be used for omega_0
        and to create the solute-dumbbell stars.
        Parameters:
            cutoff - maximum jump distance
            solt_solv_cut - minimum allowable distance between solute and solvent atoms - to check for collisions
            closestdistance - minimum allowable distance to check for collisions with other atoms. Can be a single
            number or a list (corresponding to each sublattice)
        """
        crys, chem, mset = self.crys, self.chem, self.iorlist

        nmax = [int(np.round(np.sqrt(cutoff ** 2 / crys.metric[i, i]))) + 1 for i in range(crys.dim)]

        if crys.dim == 2:
            Rvects = [np.array([n0, n1]) for n0 in range(-nmax[0], nmax[0] + 1)
                      for n1 in range(-nmax[1], nmax[1] + 1)]

        else:
            Rvects = [np.array([n0, n1, n2]) for n0 in range(-nmax[0], nmax[0] + 1)
                      for n1 in range(-nmax[1], nmax[1] + 1)
                      for n2 in range(-nmax[2], nmax[2] + 1)]

        jumplist = []
        jumpindices = []
        jumpset = set([])

        for R in Rvects:
            for i, st1 in enumerate(mset):
                for f, st2 in enumerate(mset):
                    db1 = dumbbell(i, np.zeros(self.crys.dim, dtype=int))
                    p1 = SdPair(st1[0], np.zeros(self.crys.dim, dtype=int), db1)
                    db2 = dumbbell(f, R)
                    p2 = SdPair(st2[0], R, db2)
                    if p1 == p2:  # Get the diagonal case
                        continue
                    dx = disp(self, db1, db2)
                    if np.dot(dx, dx) > cutoff ** 2:
                        continue
                    j = jump(p1, p2, 1, 1)  # since only solute moves, both indicators are +1
                    if j in jumpset:
                        continue
                    if not (collision_self(self, None, j, solt_solv_cut, solt_solv_cut) or
                            collision_others(self, None, j, closestdistance)):
                        jlist = []
                        jindlist = []
                        for gdumb in self.G:
                            p1new = p1.gop(self, gdumb, complex=False)
                            p2new = p2.gop(self, gdumb, complex=False) - p1new.R_s
                            p1new -= p1new.R_s

                            jnew = jump(p1new, p2new, j.c1, j.c2)
                            # Place some sanity checks for safety, also helpful for tests
                            if not np.allclose(jnew.state1.R_s, np.zeros(self.crys.dim), atol=self.crys.threshold):
                                raise ValueError("The initial state is not at the origin unit cell")
                            if not np.allclose(jnew.state1.db.R, np.zeros(self.crys.dim), atol=self.crys.threshold):
                                raise ValueError("The solute is not at the same site as the dumbbell in mixed dumbbell")
                            if not np.allclose(jnew.state2.db.R, jnew.state2.R_s, atol=self.crys.threshold):
                                raise ValueError("The solute is not at the same site as the dumbbell in mixed dumbbell")

                            if not jnew in jumpset:
                                dx = disp(self, jnew.state1, jnew.state2)
                                # create the negative jump
                                p1neg = SdPair(p2new.i_s, p1new.R_s, dumbbell(p2new.db.iorind, p1new.db.R))
                                p2neg = SdPair(p1new.i_s, -p2new.R_s, dumbbell(p1new.db.iorind, -p2new.db.R))
                                jnewneg = jump(p1neg, p2neg, 1, 1)
                                # add both the jump and its negative
                                jlist.append(jnew)
                                jlist.append(jnewneg)
                                jindlist.append(((jnew.state1.db.iorind, jnew.state2.db.iorind), dx))
                                jindlist.append(((jnew.state2.db.iorind, jnew.state1.db.iorind), -dx))
                                jumpset.add(jnew)
                                jumpset.add(jnewneg)
                        jumplist.append(jlist)
                        jumpindices.append(jindlist)
        return jumplist, jumpindices

    def getIndex(self, t):
        """
        :param t = (i, o) - (site, orientation) tuple
        :return: idx (integer) - the index of (i, o) in the iorlist, if it exists.
        """
        for idx,tup in enumerate(self.iorlist):
            if t[0]==tup[0] and np.allclose(t[1], tup[1], atol = 1e-8):
                return idx
        raise ValueError("The given site orientation pair {} is not present in the container".format(t))

    def db2ind(self, db):
        """
        :param db: dumbbell object
        :return:
        """
        if not isinstance(db, dumbbell):
            raise TypeError("Input object must be dumbbell")
        i = self.iorlist[db.iorind][0]
        o = self.iorlist[db.iorind][1]

        return self.getIndex((i, o))
