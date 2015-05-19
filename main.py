import numpy as np
import matplotlib.pyplot as plt
from random import randint
import itertools
from math import floor
from numpy import int64, int16, int32, dtype
from operator import xor

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
        size = randint(1, maxSize)
        x = randint(0, m - size)
        y = randint(0, n - size)
        U[x:x+size, y:y+size] = i
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
def bronk(graph,r,p,x):
    result = []
    if len(p) == 0 and len(x) == 0:
        result.append(r)
    else:
        for vertex in set(p):
            r_new = set(r)
            r_new.add(vertex)
            n = set(np.where(graph[vertex,:])[0])
            p_new = p.intersection(n)
            x_new = x.intersection(n)
            result.extend(bronk(graph,r_new,p_new,x_new))
            p.remove(vertex)
            x.add(vertex)
    return result

def combinations(x, n, maxComb = np.inf):    
    U = findPossibleSquares(x, n)
    w = np.argwhere(U)
    i_ = np.ravel_multi_index((w[:,0], w[:,1]), x.shape)
    
    m = len(i_)
    dx = np.abs(w[:,0].reshape(m,1) - w[:,0].reshape(1,m)) >= n
    dy = np.abs(w[:,1].reshape(m,1) - w[:,1].reshape(1,m)) >= n
    # collision matrix, true where squares (i,j) do not collide    
    C = dx | dy
    
    ri = bronk(C, set(), set(range(0, C.shape[0])), set())

    r = set()
    for s in ri:
        if len(s) > 0:
#             r.add(frozenset(i_[np.array(s)]))
            for L in reversed(range(1, len(s)+1)):
                for subset in itertools.combinations(s, L):
                    r.add(frozenset(i_[np.array(subset)]))
         
#     if len(r) > combinations.count:
#         print(len(r))
#         combinations.count = len(r)
    return r

# combinations.count = 0


bestLength = np.inf

def place(x, n, partialLength = 0, bestLength = np.inf):
        
    if (n == 1):
        i = np.ravel_multi_index(np.where(x), x.shape)
        if (partialLength + len(i) >= bestLength):
            return None
        
        l = np.ones(len(i), dtype=int32)
        return np.vstack([i, l])
    
    else:

        squares = np.zeros((2, 0), dtype=int32)
        
        # copy
        x = np.array(x, dtype = bool)
        
        # find 'isolated' 1x1 squared
        # finding them in early stages will help to find a better partition and a better best-scenario bound
        xnm1False = np.hstack( [ np.ones((x.shape[0], 1), dtype=bool), x[:,:x.shape[1]-1] == False] )
        xnp1False = np.hstack( [ x[:,1:] == False, np.ones((x.shape[0], 1), dtype=bool) ] )
        ynm1False = np.vstack( [ np.ones((1, x.shape[1]), dtype=bool), x[:x.shape[0]-1,:] == False] )
        ynp1False = np.vstack( [ x[1:,:] == False, np.ones((1, x.shape[1]), dtype=bool) ] )
        c = x & ((xnm1False & xnp1False) | (ynm1False & ynp1False))
        i = np.ravel_multi_index(np.where(c), x.shape)
        if (partialLength + len(i) >= bestLength):
            return None
        l = np.ones(len(i), dtype=int32)
        squares = np.vstack([i, l])
        partialLength += len(i)
        x &= (c == False)

        # computes the best scenario according only to the number of tiles on,
        # not the geometry     
        nTilesOn_ = np.sum(x)
        nTilesOn = nTilesOn_
        
        if (nTilesOn == 0):
            return squares
        
        ni = n
        bestScenario = 0
        while (ni > 0):
            nSquares = floor(nTilesOn / (ni**2))
            nTilesOn -= nSquares * (ni**2)
            bestScenario += nSquares
            ni -= 1
        
        # if the best possible scenario is worse than a known solution, we do
        # not continue
        if (partialLength + bestScenario >= bestLength):
            print('!')
            return None
        
        r = combinations(x, n)
        
        if (len(r)):
            
            # the biggest sets before
            sr = sorted(r, key=lambda x: len(x), reverse=True)
            sr.append(set())
            sr = sr[:3]
    
            squares_i = None;
    
            for indices in sr:

                length = len(indices);

                l = n*np.ones(length, dtype=int32)
                squares_n = np.vstack([np.array(list(indices), dtype=int32), l])

                # computes the best scenario according only to the number of tiles on,
                # not the geometry
                nTilesOn = nTilesOn_
                nTilesOn -= length * (n**2)
                
                if (nTilesOn == 0):
                    squares_i = squares_n
                    break

                ni = n - 1
                bestScenario = length
                while (ni > 0):
                    nSquares = floor(nTilesOn / (ni**2))
                    nTilesOn -= nSquares * (ni**2)
                    bestScenario += nSquares
                    ni -= 1
                if (partialLength + bestScenario >= bestLength):
                    print('!')
                    continue
                                
                if partialLength + length > bestLength:
                    print('XXX')
                    
                partitioning = True
                            
                for index in indices:
                    i = np.unravel_index(index, x.shape)
                    fillSquare(x, i[0], i[1], n, False);
                            
                if partitioning:
                    clusters = cluster(x);
                    squares_nm1 = np.zeros((2, 0), dtype=int32)
                    subLength = 0
                    for c in clusters:
                        xi =  np.zeros(x.shape, dtype=bool)
                        xi[np.unravel_index(c, x.shape)] = True
                        squares_nm1_i = place(xi, n - 1, partialLength + length + subLength, bestLength)
                        if squares_nm1_i == None:
                            squares_nm1 = None
                            break;
                        squares_nm1 = np.hstack([squares_nm1, squares_nm1_i])
                        subLength = squares_nm1.shape[1]                
                else:
                    squares_nm1 = place(x, n - 1, partialLength + length, bestLength)                    
                    
                for index in indices:
                    i = np.unravel_index(index, x.shape)
                    fillSquare(x, i[0], i[1], n, True);

                if (squares_nm1 == None):
                    continue
                
                subLength = squares_nm1.shape[1]                
                totalLength = partialLength + length + subLength
                
                if (totalLength < bestLength):
                    squares_i = np.hstack([ squares_n, squares_nm1])
                    bestLength = totalLength
                    print("Found length %i for n=%i" % (totalLength, n))

#                     xi = populate(x, squares_i)
#                     fig = plt.figure(1)
#                     ax = fig.add_subplot(122)
#                     ax.imshow(xi, interpolation='none')
#                     fig.canvas.draw()


            if (squares_i == None):
                return None
                                          
            squares = np.hstack([ squares, squares_i ])
            return squares
                
        else:
            # no squares of size n can be placed, we try with smaller ones            
            squares_nm1 = place(x, n - 1, partialLength, bestLength)
            if (squares_nm1 == None):
                return None
            squares = np.hstack([ squares, squares_nm1 ])
            return squares


def populate(x, squares):
    r = np.zeros(x.shape, dtype=int32)
    for k in range(0, squares.shape[1]):
        i = np.unravel_index(squares[0, k], x.shape)
        n = squares[1, k];
        r[i[0]:i[0]+n, i[1]:i[1]+n] = k + 1
    return r
   
def cluster(x):
    
    result = []

    x = np.array(x, dtype=bool)
    
    while True:
        [ix, iy] = np.where(x)
        
        if (len(ix) == 0):
            break
     
        ix = ix[0]
        iy = iy[0]
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
            i = np.array(list(set(np.ravel_multi_index(np.vstack([nx, ny]), x.shape))))
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
# c = cluster(graph)
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

#x = generate(50, 50, 25, 20)
# x = np.loadtxt('puzzle.txt')
#x = x > 0
#np.savetxt('puzzle.txt', x, fmt='%i')

fig = plt.figure(1)
ax = fig.add_subplot(121)
ax.imshow(x, interpolation='none')
ax = fig.add_subplot(122)
ax.imshow(x, interpolation='none')
plt.show(block=False)

# cluster(x)
    
squares = place(x, min(x.shape));
print(squares)

# print(place(U, 2));
# U = generate(10, 10, 7, 5)
# U = (U != 0)

xi = populate(x, squares)

assert np.all(xi[x] != 0)

print("Found solution with %i squares" % (squares.shape[1]))

ax = fig.add_subplot(122)
ax.imshow(xi, interpolation='none')

plt.show()