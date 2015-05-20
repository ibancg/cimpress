import numpy as np
import matplotlib.pyplot as plt
from random import randint
import itertools
from math import floor, sqrt
from numpy import int16
import time
import threading
import sys

x = np.array([
              [0,0,1,1,1,1,1,1,1,0,0,1],
              [1,0,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1],
              [0,0,0,0,1,1,1,1,1,1,1,0],
              [0,0,0,0,1,1,1,1,1,1,1,0],
              [0,0,0,0,1,1,1,1,1,1,0,0],
              [0,0,0,0,1,1,1,1,1,1,1,1],
              ])

x = (x == 1)

def generate(m, n, nSquares, maxSize):
    U = np.zeros((m, n), dtype=np.int)
    maxSize = min(maxSize, min(m, n))
    for i in range(0, nSquares):
        size1 = randint(1, maxSize)
        size2 = randint(1, maxSize)
        x = randint(0, m - size1)
        y = randint(0, n - size2)
        U[x:x+size1, y:y+size2] = i
    return U;

def findPossibleSquares(x, n):
    xx = np.ones((x.shape[0], x.shape[1] - (n-1)), dtype=bool)
    xy = np.ones((x.shape[0] - (n-1), x.shape[1] - (n-1)), dtype=bool)
    for j in range(0, n):
        xx &= x[:, j:j+xx.shape[1]]
    for j in range(0, n):
        xy &= xx[j:j+xy.shape[0], :]
    Ui = xy
    Ui = np.hstack([Ui, np.zeros((Ui.shape[0], n-1), dtype=bool)])
    Ui = np.vstack([Ui, np.zeros((n-1, Ui.shape[1]), dtype=bool)])
    return Ui

def fillSquare(x, i, j, n, value):
    x[i:i+n, j:j+n] = value

# Bron-Kerbosch algorithm
def bronk(graph, r, p, x, limit = 500):
    result = []
    if limit  > 0:
        if len(p) == 0 and len(x) == 0:
            result.append(r)
        else:
            for vertex in set(p):
                r_new = set(r)
                r_new.add(vertex)
                n = set(np.where(graph[vertex,:])[0])
                p_new = p.intersection(n)
                x_new = x.intersection(n)
                bk = bronk(graph, r_new, p_new, x_new, limit)
                limit -= len(bk)
                result.extend(bk)
                p.remove(vertex)
                x.add(vertex)
    return result

def combinations(x, n, maxComb = sys.maxint):
    r = set()
    
#     print('combinations n=%i ...' % (n))
    if n <= min(x.shape):
        U = findPossibleSquares(x, n)
        w = np.argwhere(U)
        i_ = np.ravel_multi_index((w[:,0], w[:,1]), x.shape)
        
        m = len(i_)
        dx = np.abs(w[:,0].reshape(m,1) - w[:,0].reshape(1,m)) >= n
        dy = np.abs(w[:,1].reshape(m,1) - w[:,1].reshape(1,m)) >= n
        # collision matrix, true where squares (i,j) do not collide    
        C = dx | dy
        
        # find the cliques: all combinations of squares than can be placed toghether
#         print('BK size=%i ...' % (m))
        ri = bronk(C, set(), set(range(0, C.shape[0])), set(), maxComb)
#         print('...ok')
    
        for s in ri:
            if len(s) > 0:
    #             r.add(frozenset(i_[np.array(s)]))
                for L in reversed(range(1, len(s)+1)):
                    for subset in itertools.combinations(s, L):
                        r.add(frozenset(i_[np.array(subset)]))
                        if (len(r) >= maxComb):
#                             print('...ok')
                            return (r, False)
         
#     print('...ok')
    full = len(r) < maxComb
    return (r, full)


def decomposeSquares(n, ni = 0):
    if ni == 0:
        ni = int(floor(sqrt(n)))
                
    bestScenario = 0
    r = 0
    while (ni > 0):
        nSquares = int(floor(n / (ni**2)))
        n -= nSquares * (ni**2)
        bestScenario += nSquares
        ni -= 1
        r + nSquares
        
    return r
    
def registerSolution(x, solution):
#     registerSolution.count += 1
#     print(str(registerSolution.count) + ' ' + str(x.shape) + ' ' + str(solution.shape[1]))
#     [h, m, dims] = hash_(x)
#     if (h is not None):
#         if np.any(solution[1,:] - m[1] < 0):
#             print(solution[0,:] - m[0])
#         print(solution[1,:] - m[1])
#         print(dims)
#         i = np.ravel_multi_index((solution[0,:] - m[0], solution[1,:] - m[1]), dims)
#         sol = np.vstack([i, solution[2,:]])
#         hash_.table[h] = sol
    return

registerSolution.count = 0
 
def place(x, n, maxComb = [], 
          partialLength = 0,
          bestLength = np.inf,
          tryPartitions = True,
          findIsolatedSingleTiles = True,
          maxDepth = 1000):
        
    shift = np.array([[0],[0],[0]], dtype=int16)
    optimalSolution = True
    
    if place.stop:
        return (None, None)
    
    if maxDepth == 0:
        optimalSolution = False
        squares = np.zeros((3, 0), dtype=int16)
    elif (n == 1):
        i = np.transpose(np.argwhere(x))
        if (partialLength + i.shape[1] >= bestLength):
            return (None, None)
        
        l = np.ones(i.shape[1], dtype=int16)
        squares = np.vstack([i, l]) 
    else:

        squares = np.zeros((3, 0), dtype=int16)
        # shrink the puzzle
        u = np.argwhere(x)    
        nRemainingTiles_ = u.shape[0]

        if (nRemainingTiles_ == 0):
            return (squares, optimalSolution)

        # we reduce n
        n = min(n, int(floor(sqrt(nRemainingTiles_))))

        shrink = True        
        if shrink:   
            m_ = np.min(u, 0)
            M_ = np.max(u, 0)
            dims_ = M_ - m_ + 1
            u0 = u - m_        
    
            x = np.zeros(dims_, dtype = bool)
            x[u0[:,0], u0[:,1]] = True
            shift = np.array([[m_[0]],[m_[1]],[0]], dtype=int16);
        else:
            x = np.array(x, dtype = bool)
        
        if findIsolatedSingleTiles:
            # find 'isolated' 1x1 squared
            # finding them in early stages will help us to find a better partition and a better best-scenario bound
            xnm1False = np.hstack( [ np.ones((x.shape[0], 1), dtype=bool), x[:,:x.shape[1]-1] == False] )
            xnp1False = np.hstack( [ x[:,1:] == False, np.ones((x.shape[0], 1), dtype=bool) ] )
            ynm1False = np.vstack( [ np.ones((1, x.shape[1]), dtype=bool), x[:x.shape[0]-1,:] == False] )
            ynp1False = np.vstack( [ x[1:,:] == False, np.ones((1, x.shape[1]), dtype=bool) ] )
            c = x & ((xnm1False & xnp1False) | (ynm1False & ynp1False))
            i = np.transpose(np.argwhere(c))
            if (i.shape[1] > 0) :
                if (partialLength + i.shape[1] >= bestLength):
                    return (None, None)
                l = np.ones(i.shape[1], dtype=int16)
                squares = np.vstack([i, l])
                partialLength += i.shape[1]
                x &= (c == False)    
                nRemainingTiles_ -= i.shape[1]
                tryPartitions = True
                 
        if (nRemainingTiles_ == 0):
            return (shift + squares, optimalSolution)

        # computes the best scenario according only to the number of tiles on,
        # not the geometry
        bestScenario = decomposeSquares(nRemainingTiles_, n)
        
        # if the best possible scenario is worse than a known solution, we do
        # not continue
        if (partialLength + bestScenario >= bestLength):
            return (None, None)        

        partitions = [] if not tryPartitions else partition(x)
        partitionExists = len(partitions) > 1;
          
        if partitionExists:
            squares_nm1 = np.zeros((3, 0), dtype=int16)                 
            # the smallest sets before
            partitions = sorted(partitions, key=lambda x: len(x))                    
            subLength = 0
            for c in partitions:
                xi = np.zeros(x.shape, dtype=bool)
                xi[np.unravel_index(c, x.shape)] = True
                [squares_nm1_i, optimalSolution_i] = place(xi, n, maxComb, partialLength + subLength, bestLength, False, findIsolatedSingleTiles, maxDepth)
                if squares_nm1_i == None:
                    squares_nm1 = None
                    break;
                squares_nm1 = np.hstack([squares_nm1, squares_nm1_i])
                subLength = squares_nm1.shape[1]
                optimalSolution &= optimalSolution_i
                
            if squares_nm1 is None:
                return (None, None)
                 
            squares = np.hstack([ squares, squares_nm1 ]);
        else:

            tryPartitions = False
#             [h, m, dims] = hash_(x)
#             if (h in hash_.table.keys()):
#                 solution = hash_.table[h]
#                 [ix, iy] = np.unravel_index(solution[0,:], dims)
#                 ix = ix + m[0]
#                 iy = iy + m[1]
#                 squares_i = np.vstack([ ix, iy, solution[1,: ]] )
#                 hash_.hits += 1
#                 return shift + np.hstack([ squares, squares_i ]);
    
            maxComb_ = 1
            for k in range(0, len(maxComb)):
                if np.isnan(maxComb[k]):
                    maxComb_ = sys.maxint
                else:
                    if (nRemainingTiles_ < maxComb[k]):
                        maxComb_ += 1
                 
            maxComb__ = 200
            (r, full) = combinations(x, n, maxComb__)
            if not full:
                optimalSolution = False
            
            if (len(r) > 0):
     
                if (len(r) < maxComb__):
                    r.add(frozenset())
                 
                # the biggest sets before
                sr = sorted(r, key=lambda x: len(x), reverse=True)
     
                if maxDepth == 1:
                    indices = sr[0]
                    length = len(indices)
                    l = n*np.ones(length, dtype=int16)
                    i = np.vstack(np.unravel_index(np.array(list(indices), dtype=int16), x.shape))
                    squares = np.vstack([i, l])
                    if (optimalSolution):
                        registerSolution(x, squares)
                    return (shift + squares, optimalSolution)
                       
#                 squaresl = []
                scores = []
                 
                for indices in sr:
         
                    length = len(indices);
                    score = -length - 1
                    scores.append(score)
#                     l = n*np.ones(length, dtype=int16)
#                     i = np.vstack(np.unravel_index(np.array(list(indices), dtype=int16), x.shape))
#                     squares_n = np.vstack([i, l])
#          
#                     for index in np.transpose(squares_n):
#                         fillSquare(x, index[0], index[1], n, False);
#                                      
#                     squares_nm1 = place(x, n - 1, maxComb, partialLength + length, bestLength, False, False, 1)
#                     if (squares_nm1 is None):
#                         squaresl.append(None)
#                         scores.append(1.0)
#                     else:
#                         sq = np.hstack( [ n*np.ones((length), dtype=int16), squares_nm1[2,:] ])
#                         squaresl.append(sq)
#     #                     sq = sq[sq > 1]
#                         if len(sq) > 0:
#                             score = -np.float(np.sum(sq**2)) / len(sq)
#                         else:
#                             score = -1000000
#                         scores.append(score)
#                              
#     #                 squares_nm1 = squares_nm1[:,squares_nm1[2,:] >= n - extraDepth]
#     #                 score = (np.sum(squares_nm1[2,:]**2) + len(indices)*(n**2))/(squares_nm1.shape[1] + len(indices))
#                     for index in np.transpose(squares_n):
#                         fillSquare(x, index[0], index[1], n, True);
             
                scores = np.array(scores)
                scores -= 1e-5*np.random.normal(0.0, 1.0, len(scores))
                scores_i = np.argsort(scores)
                scores_i = scores_i[scores[scores_i] < 0]
                squares_i = None
                optimalSolution_i = True  
                if maxComb_ < len(scores_i):
                    scores_i = scores_i[:maxComb_]
                    optimalSolution_i = False  
                    
                for score_i in scores_i:
                    indices = sr[score_i]
    
#                 squares_i = None;
#                 sr = sr[:maxComb_]
#                 for indices in sr:
    
                    length = len(indices);
    
                    l = n*np.ones(length, dtype=int16)
                    i = np.vstack(np.unravel_index(np.array(list(indices), dtype=int16), x.shape))
                    squares_n = np.vstack([i, l])
    
                    # computes the best scenario according only to the number of tiles on,
                    # not the geometry
                    nRemainingTiles = nRemainingTiles_
                    nRemainingTiles -= length * (n**2)
                    
                    if (nRemainingTiles == 0):
                        squares_i = squares_n
                        optimalSolution_i = True
                        break
    
                    bestScenario = length + decomposeSquares(nRemainingTiles, n - 1)
                    if (partialLength + bestScenario >= bestLength):
                        continue
                                    
                    for index in np.transpose(squares_n):
                        fillSquare(x, index[0], index[1], n, False);
                                
                    [squares_nm1, optimalSolution_nm1] = place(x, n - 1, maxComb, partialLength + length, bestLength, length > 0, findIsolatedSingleTiles, maxDepth - 1)                    
                        
                    for index in np.transpose(squares_n):
                        fillSquare(x, index[0], index[1], n, True);
    
                    if (squares_nm1 == None):
                        continue
                    
                    subLength = squares_nm1.shape[1]                
                    totalLength = partialLength + length + subLength
                    
                    if (totalLength < bestLength):
                        squares_i = np.hstack([ squares_n, squares_nm1])
                        bestLength = totalLength
                        optimalSolution_i &= optimalSolution_nm1
                        
                if (squares_i == None):
                    return (None, None)
                                              
#                 [h, m, dims] = hash_(x)
#                 if (h is not None):
#                     i = np.ravel_multi_index((squares_i[0,:] - m[0], squares_i[1,:] - m[1]), dims)
#                     solution = np.vstack([i, squares_i[2,:]])
#                     hash_.table[h] = solution
                    
                squares = np.hstack([ squares, squares_i ])
                optimalSolution &= optimalSolution_i
            else:
                # no squares of size n can be placed, we try with smaller ones            
                [squares_nm1, optimalSolution] = place(x, n - 1, maxComb, partialLength, bestLength, tryPartitions, findIsolatedSingleTiles, maxDepth)
                if (squares_nm1 == None):
                    return (None, None)
                squares = np.hstack([ squares, squares_nm1 ])            

    if (optimalSolution):
        registerSolution(x, squares)
    return (shift + squares, optimalSolution)

def hash_(x):
    u = np.argwhere(x)
    if (len(u) == 0):
        print(len(u))
    m = np.min(u, 0)
    M = np.max(u, 0)
    u0 = u - m
    dims = M - m + 1
    tdims = tuple(dims)
    h = None
    if min(dims) > 1 and np.prod(dims) < 1400:
        h = hash((tdims, frozenset(np.ravel_multi_index((u0[:,0], u0[:,1]), tdims))))
    return (h, m, tdims)
#     return (None, None, None)

hash_.table = {}
hash_.hits = 0

def populate(x, squares):
    r = np.zeros(x.shape, dtype=int16)
    for k in range(0, squares.shape[1]):
        i = squares[0:2, k]
        n = squares[2, k];
        assert(np.all(r[i[0]:i[0]+n, i[1]:i[1]+n] == 0))
        r[i[0]:i[0]+n, i[1]:i[1]+n] = k + 1
    return r
   
def partition(x):
    
    result = []

    x = np.array(x, dtype=bool)
    
    while True:
        [ix, iy] = np.where(x)
        
        if (len(ix) == 0):
            break
     
        # we choose the closest tile to the center of the puzzle
        d = np.abs(ix - x.shape[0]/2) + np.abs(iy - x.shape[1]/2)
        i0 = np.argmin(d)
        ix = ix[i0]
        iy = iy[i0]
        l = 1
        
        while True:
            # neighbors
            nx = np.hstack([ix, ix  - 1, ix + 1, ix, ix])
            ny = np.hstack([iy, iy, iy, iy - 1, iy + 1])
            nx[nx < 0] = 0
            nx[nx >= x.shape[0]] = x.shape[0] - 1
            ny[ny < 0] = 0
            ny[ny >= x.shape[1]] = x.shape[1] - 1
            # unique
            i = np.unique(np.ravel_multi_index(np.vstack([nx, ny]), x.shape))
            i_ = np.unravel_index(i, x.shape)
            i = i[x[i_]]
            if (len(i) == l):
                break
            [ix, iy] = np.unravel_index(i, x.shape)
            l = len(ix)
    
        result.append(i)
        x[ix,iy] = False;
    return result
    
# dealing with a graph as list of lists 
# graph = np.array([[1,1,0,0,1,1],[1,1,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]])
# c = partition(graph)
# print(c)
# if True:
#     exit(0)
# print(graph)
# print(c)
# r = bronk(graph, set(), set(range(0,graph.shape[0])), set())
# print(r)


# x = np.rot90(x)

# n = 4
# s1 = combinations(x, n)
# print(len(s1))
# s0 = combinations0(x, n)
# print(len(s0))
# 
# print(s0 - s1)
# print(len(combinations(x, n)))

# x = generate(100, 100, 55, 30)
x = np.loadtxt('puzzle.txt')
print(np.sum(x))
x = x > 0
# np.savetxt('puzzle.txt', x, fmt='%i')

fig = plt.figure(1)
ax = fig.add_subplot(121)
ax.imshow(x, interpolation='none')
ax = fig.add_subplot(122)
ax.imshow(x, interpolation='none')
plt.show(block=False)

# partition(x)
    
bestLength = 10000
squares_i = np.array((3,0), dtype=int16)
squares = np.array((3,0), dtype=int16)

maxCombThresholds = [
                     [ ],
                     [ 100 ],
                     [ 200, 100 ],
                     [ 200, 100, 50 ],
                     [ 300, 200, 100, 50 ],
                     [ 400, 300, 200, 100, 50 ],
                     [ 500, 400, 300, 200, 100, 50 ],
                     [ 600, 500, 400, 300, 200, 100, 50 ],
                     [ 700, 600, 500, 400, 300, 200, 100, 50 ],
                     [ 800, 700, 600, 500, 400, 300, 200, 100, 50 ],
                     [ 900, 800, 700, 600, 500, 400, 300, 200, 100, 50 ],
                     [ 1000, 1000, 1000, 1000, 700, 600, 500, 400, 300, 200 ],
                     [ 1000, 1000, 1000, 1000, 1000, 700, 600, 500, 400, 300, 200 ],
                     [ 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000 ],
                     [ np.NaN ],
                     ]
  

t0 = time.time()
place.stop = False
def checktime():
    elapsed0 = time.time() - t0
    if (elapsed0 > 60.0):
        place.stop = True
        print('reached time limit, quiting ...')
    else:
        threading.Timer(5.0, checktime).start()

checktime()


for k in range(0,len(maxCombThresholds)):
    if place.stop:
        break
    print('iteration %i' % (k))
    for n in range(0,4):
        if place.stop:
            break
        t = time.time()
        [squares_i, optimalSolution] = place(x, min(x.shape), maxCombThresholds[k], 0, bestLength, True, True)
        elapsed = time.time() - t
        elapsed0 = time.time() - t0
        if (squares_i is not None) and squares_i.shape[1] < bestLength:
    #         print('found solution, bestLength = %i' % (bestLength))
            bestLength = squares_i.shape[1]
            squares = squares_i
            xi = populate(x, squares)
            ax = fig.add_subplot(122)
            ax.imshow(xi, interpolation='none')
            fig.canvas.draw()
    
            print('Cache has %i entries, got %i hits' % (len(hash_.table), hash_.hits))
            print("Found solution with %i squares in %0.2fs, total elapsed %0.2fs " % (squares.shape[1], elapsed, elapsed0))
            xi = populate(x, squares)
            if np.any(xi[x] == 0):
                assert(False)

elapsed0 = time.time() - t0
print("Found solution with %i squares, total elapsed %0.2fs " % (squares.shape[1], elapsed0))

print(squares)
plt.show()