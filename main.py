import numpy as np
import matplotlib.pyplot as plt
from random import randint
import itertools
from math import floor
from numpy import int64, int16, int32

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

def squares_bool(x, n):
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
#     Ui = [Ui; false(n - 1, size(Ui, 2))];

def fillSquare(x, i, j, n, value):
    x[i:i+n, j:j+n] = value

    
def combinations0(x, n, maxComb = np.inf):
    U = squares_bool(x, n)
    w = np.argwhere(U)
    i_ = np.ravel_multi_index((w[:,0], w[:,1]), x.shape)
    r = set()
    
    if len(i_):
        for index in i_:                
            i = np.unravel_index(index, x.shape)
            fillSquare(x, i[0], i[1], n, False);
            subcombs = combinations0(x, n, maxComb)
            for comb in subcombs:
                c = set(comb)
                c.add(index)
                r.add(frozenset(c));
            r.add(frozenset([index]));
            fillSquare(x, i[0], i[1], n, True);

    return r

def combinations(x, n, maxComb = np.inf):
    U = squares_bool(x, n)
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
         
    return r


bestLength = np.inf

def place(x, n, partialLength = 0, bestLength = np.inf):
#     global bestLength
    if (n == 1):
        i = np.ravel_multi_index(np.where(x), x.shape)
        if (partialLength + len(i) >= bestLength):
            return None
        
        l = np.ones(len(i), dtype=int32)
        return np.vstack([i, l])
    
    else:

        # computes the best scenario according only to the number of tiles on,
        # not the geometry     
        nTilesOn = np.sum(x)
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
        il = None
        
        if (len(r)):
            # the biggest sets before
            sr = sorted(r, key=lambda x: len(x), reverse=True)
            sr.append(set())
#             print(len(sr))
            sr = sr[:10]
    
            for indices in sr:
                
                for index in indices:
                    i = np.unravel_index(index, x.shape)
                    fillSquare(x, i[0], i[1], n, False);

                length = len(indices);
                
                if partialLength + length > bestLength:
                    print('XXX')
                    
                il_nm1 = place(x, n - 1, partialLength + length, bestLength)
                
                for index in indices:
                    i = np.unravel_index(index, x.shape)
                    fillSquare(x, i[0], i[1], n, True);

                if (il_nm1 == None):
                    continue
                
                subLength = il_nm1.shape[1]


                l = n*np.ones(length, dtype=int32)
                il_n = np.vstack([np.array(list(indices), dtype=int32), l])
                
                totalLength = partialLength + length + subLength
                
                if (totalLength < bestLength):
                    il = np.hstack([ il_n, il_nm1])
                    bestLength = totalLength
                    print("Found length %i for n=%i" % (totalLength, n))
                    
                    x_ = np.zeros(x.shape, dtype=int32)
                    for k in range(0, il.shape[1]):
                        i_ = np.unravel_index(il[0, k], x.shape)
                        n_ = il[1, k];
                        x_[i_[0]:i_[0]+n_, i_[1]:i_[1]+n_] = k + 1
                     
                    fig = plt.figure(1)
                    ax = fig.add_subplot(122)
                    ax.imshow(x_, interpolation='none')
                    fig.canvas.draw()


                                
            return il
                
        else:
            return place(x, n - 1, partialLength, bestLength)
# 
#     end
# end

# stuff = [1, 2, 3]
# for L in range(0, len(stuff)+1):
#   for subset in itertools.combinations(stuff, L):
#     print(subset)
#     
# U = squares_bool(x, 7)

# r = combinations(x, 4)


# print(len(r))
# sr = sorted(r, key=lambda x: len(x), reverse=True)
# print(r)

#the Bron-Kerbosch recursive algorithm
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

# dealing with a graph as list of lists 
graph = np.array([[0,1,0,0,1,1],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]])
print(graph)
r = bronk(graph, set(), set(range(0,graph.shape[0])), set())
print(r)

# x = np.rot90(x)

# n = 4
# s1 = combinations(x, n)
# print(len(s1))
# s0 = combinations0(x, n)
# print(len(s0))
# 
# print(s0 - s1)
# print(len(combinations(x, n)))

# x = generate(20, 20, 20, 9)
# x = x > 0

fig = plt.figure(1)
ax = fig.add_subplot(121)
ax.imshow(x, interpolation='none')
ax = fig.add_subplot(122)
ax.imshow(x, interpolation='none')
plt.show(block=False)

il = place(x, min(x.shape));
print(il)

# print(place(U, 2));
# U = generate(10, 10, 7, 5)
# U = (U != 0)

x_ = np.zeros(x.shape, dtype=int32)
for k in range(0, il.shape[1]):
    i = np.unravel_index(il[0, k], x.shape)
    n = il[1, k];
    x_[i[0]:i[0]+n, i[1]:i[1]+n] = k + 1

assert np.all(x_[x] != 0)
 
ax = fig.add_subplot(122)
ax.imshow(x_, interpolation='none')

plt.show()