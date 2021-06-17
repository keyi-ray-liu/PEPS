import numpy as np
from math import sqrt
from numpy.core import einsumfunc
from numpy.core.defchararray import multiply
import tensornetwork as tn
from functools import reduce


# The preliminary TN simulation program I have for the small scale 2D system we have. 

# Notations: upper case letter are always the collection, where lower case indicates a member of that collection

# intialize the parameters for the simulation
def initParameters():
    para = {
    'rdim' : 3,
    'cdim' :3,
    't': 1.0,
    'int_ee': 1,
    'int_ne': 1,
    'z':1,
    'zeta':0.5,
    'ex': 0.2,
    'bdim': 9,
    'batch':50,
    'step':200,
    'initstep': 1,
    'translation invariance': 0,
    'Q': 0.99,
    'lo':-0.1,
    'hi':0.1,
    'print':0,
    'occupation':5}
    return para


# Flip a site with zero occupation 
def generateposition(s, size):
    pos = int(np.rint(np.random.rand(1)* 8)[0])
    return pos if not s[pos] else generateposition(s, size)

# initialize the spin 
def initSpin(rdim, cdim, occ):
    size = rdim * cdim
    s = [0] * size

    for _ in range(occ):
        pos = generateposition(s, size)
        s[pos] = 1

    return s

def kdelta(s1, s2):
    return 1 if s1 == s2 else 0
    
# generates the initial tensors A0 and A1
def initTensor(para):
    rdim, cdim, bdim, ti =  para['rdim'], para['cdim'], para['bdim'], para['translation invariance']
    lo, hi = para['lo'], para['hi']
    if ti:
        return [np.random.uniform(lo, hi, (bdim, bdim, bdim, bdim)).astype('float32'), np.random.uniform(lo, hi, (bdim, bdim, bdim, bdim)).astype('float32')]
    else:
        return [[[np.random.uniform(lo, hi, (bdim, bdim, bdim, bdim)).astype('float32'), np.random.uniform(lo, hi, (bdim, bdim, bdim, bdim)).astype('float32')] for _ in range(cdim)] for _ in range(rdim)]

# generate the sets of 2D spin configurations to be run through the Monte Carlo simulation
def generateState(para):
    # set() makes sure the spin configurations are unique
    rdim, cdim, occ, batch = para['rdim'], para['cdim'], para['occupation'], para['batch']
    S = set()
    while len(S) < batch:
        S.add(tuple(initSpin(rdim, cdim, occ)))

    new = [0] * len(S)
    # write spin configurations to the resulting array
    for j, s in enumerate(S):
        new[j] = np.array(s).reshape((rdim, cdim))
    
    if para['print']:
        print('States : {}'.format(new))

    return new

# sets up the 2D TN using the tensornetwork module
def evalTN(s, A, skip, skiprow, skipcol, para):
    #print('evaluation start: ifskip{} skiprow {} skipcol {}'.format(skip, skiprow, skipcol))
    rdim = len(s)
    cdim = len(s[0])
    ti = para['translation invariance']

    def setoutputorder():
        return [tns[skiprow][skipcol -1][1], tns[skiprow][(skipcol + 1)% cdim][0], tns[skiprow -1][skipcol][3], tns[(skiprow +1)% rdim][skipcol][2] ]

    if ti:
        tns = [[tn.Node([A[0] if s[row][col] else A[1]][0]) for col in range(cdim)] for row in range(rdim)]
    else:
        tns = [[tn.Node([A[row][col][0] if s[row][col] else A[row][col][1]][0]) for col in range(cdim)] for row in range(rdim)]

    # now we draw the edges

    for row in range(rdim):
        for col in range(cdim):

            # horizontal edges, [0] is left edge, [1] is right edge.
            tns[row][col - 1][1] ^ tns[row][col][0]
            # vertical edges, [2] is top edge, [3] is bottom edge
            tns[row - 1][col][3] ^ tns[row][col][2]
    
    if skip:
        tn.remove_node(tns[skiprow][skipcol])

    nodes = tn.reachable(tns[skiprow - 1][skipcol - 1])
    if not skip:
        return tn.contractors.greedy(nodes).tensor
    else:
        return tn.contractors.greedy(nodes, output_edge_order=setoutputorder()).tensor
    
# define the brute force (exact) contraction for tensor network
#def contractionBruteForce(A):
#    for bond in A:
#        return A

# the hamiltonian functions acts on the state on the right and return a new state.
# the input parameters are passed externally
def hamiltonian(s, para):
    rdim, cdim, t, int_ee, int_ne, z, zeta, ex = para['rdim'], para['cdim'], para['t'], para['int_ee'],para['int_ne'], para['z'], para['zeta'], para['ex']
    new = np.zeros((rdim, cdim))

    def checkHopping(row, col):
        # set up the NN matrix
        NN = []
        if row:
            NN += [(-1, 0)]
        if not row == rdim - 1:
            NN += [(1, 0)]
        if col:
            NN += [(0, -1)]
        if not col == cdim -1 :
            NN += [(0, +1)]

        # sum the hopping terms
        return sum([int(s[row + i][col + j]) ^ int(s[row][col]) for i, j in NN])

    def ee(row, col):  
        res = 0
        for srow in range(rdim):
            for scol in range(cdim):
                r = sqrt((srow - row)**2 + (scol - col)**2)
                factor = [ 1 - ex if np.rint(r) == 1 else 1][0]
                res +=  int_ee * z * factor / ( r + zeta ) * s[srow][scol] * s[row][col]
        return res


    def ne(row, col):
        res = 0
        # sum the contribution from all sites
        for srow in range(rdim):
            for scol in range(cdim):
                r = sqrt((srow - row)**2 + (scol - col)**2)
                res += - int_ne * z / ( r + zeta ) * s[srow][scol]
        return res

    for row in range(rdim):
        for col in range(cdim):

            # the hopping part
            new [row][col] += t * checkHopping(row, col)

            # the ee interaction part, the 0.5 is for the double counting of sites. 
            new [row][col] += ee(row, col) * 0.5
            # the ne interaction part

            new [row][col] += ne(row, col)
    return new


def innerProduct(A, B):
    return np.sum(np.multiply(A, B))



# The function that calculates the tensor derivative: (return a single tensor based on input single state)
def calDelta(S, A, para):
    rdim = para['rdim']
    cdim = para['cdim']
    return [[[[evalTN(state, A, 1, i, j, para) * kdelta(o, state[i][j]) for state in S] for o in (0, 1)] for j in range(cdim)] for i in range(rdim)]

# The function that calculates monte carlo average
def monEx(W, norm, arg1, arg2=0):
    return sum([weight * arg1[i] for i, weight in enumerate(W)]) / norm  if not arg2 else sum([weight * arg1[i] * arg2[i] for i, weight in enumerate(W)]) / norm

# THe main function that does the iterative updates to the tensors (return the full set of tensors)
def stepUpdate(S, A, EST, step, DERIV, para):
    rdim = para['rdim']
    cdim = para['cdim']
    initstep = para['initstep']
    bdim = para['bdim']
    Q = para['Q']

    # decreasing step length
    newstep = initstep * np.power(Q, step)

    # random but well-bounded step length
    def randomstep():
        return np.random.rand(bdim, bdim, bdim, bdim)

    return [[[A[i][j][o] - randomstep() * newstep * np.sign(DERIV[i][j][o]) for o in (0, 1)] for j in range(cdim) ] for i in range(rdim)]

# The function that estimate the energy (return the full set of estimates)
def estimator(S, W, para):
    # initialize the simulation parameters

    
    # generate the energy expectation value
    estimate = [sum([W[j] / W[i] * innerProduct(sprime, hamiltonian(state, para)) for j, sprime in enumerate(S)]) for i, state in enumerate(S)]

    # normalization
    if para['print']:
        print('estimator : {}'.format(estimate))
    return estimate

# The function that calculates the tensor derivatives
def calDeriv(W, DELTA, EST, para):
    rdim = para['rdim']
    cdim = para['cdim']
    return [[[2 * (monEx(W, norm, DELTA[i][j][o], EST) - monEx(W, norm, DELTA[i][j][o]) * monEx(W, norm, EST)) for o in (0, 1)] for j in range(cdim)] for i in range(rdim)]

# The function that wraps the TN module functions that contract a particular TN for a 2D configuration
def calEnergy(W, EST, norm ):
    return sum(reduce(np.multiply, [W, W, EST])) / norm

def calNorm(W, para):
    norm = sum(np.multiply(W, W))
    
    if para['print']:
        print('norm : {}'.format(norm))
    return norm

# return the full set of weights W
def calWeight(S, A, para):
    W = [evalTN(state, A, 0, 0, 0, para) for state in S]
    if para['print']:
        print('Weights : {}'.format(W))
    return W

if __name__ == '__main__':
    para = initParameters()
    
    A = initTensor(para)
    
    # Start the iterative Monte Carlo updates
    energy = []
    # Here we try using a randomly generated set of occupation configuration
    S = generateState(para)

    for step in range(para['step']):

        # calculates the W(S) for all S
        W = calWeight(S, A, para)
        norm = calNorm(W, para)

        # calculates E(S) for all S
        EST = estimator(S, W, para)

        # calculates the energy on each pass
        currentenergy = calEnergy(W, EST, norm)
        

        # calculate the tensor derivatives
        DELTA = calDelta(S, A, para)

        # calculate the tensor derivatives
        DERIV = calDeriv(W, DELTA, EST, para)

        # update the tensor A's
        A = stepUpdate(S, A, EST, step, DERIV, para)
        

        #print result on each pass
        energy += [[step, currentenergy]]

        #test, printing all revelant parameters:
        if para['batch'] < 5:
            print(W, EST, DELTA, DERIV)

        print('step: {}, energy :{}'.format(step, currentenergy))

        with open('res', 'a') as f:
            f.write( 'step: {}, energy :{} \n'.format(step, currentenergy))
        
    print('energy is {}'.format(energy))
    #calEnergy(S, A, para)
    #S = initSpin(rdim, cdim)
    #print(hamiltonian(S, rdim, cdim, t, int_ee, int_ne, Z, zeta, ex))
