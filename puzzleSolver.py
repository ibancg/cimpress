#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from random import randint
import itertools
from math import floor, sqrt
from numpy import int16
import time
import sys
import threading

class PuzzleSolver:

    def __init__ (self):
        # configuration
        self.plots = False
        self.allowPromptFindIsolatedSingleTiles = True
        self.sigmaNoiseInScores = 1.0
        self.allowPartitions = True
        self.allowShrink = True
        self.cachedSolutions = True
        
        self.timeLimit = 5.0
        self.timeLimitGrowFactor = 0.0

        self.stop = False
        self.cache = {}
        self.cacheHits = 0
        
    def findPossibleSquares(self, x, n):
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
    
    def fillSquare(self, x, i, j, n, value):
        x[i:i+n, j:j+n] = value
    
    # Bron-Kerbosch algorithm
    def bronk(self, graph, r, p, x, limit = 500):
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
                    bk = self.bronk(graph, r_new, p_new, x_new, limit)
                    limit -= len(bk)
                    result.extend(bk)
                    p.remove(vertex)
                    x.add(vertex)
        return result
    
    def combinations(self, x, n, maxComb = sys.maxsize):
        r = set()
        
    #     print('combinations n=%i ...' % (n))
        if n <= min(x.shape):
            U = self.findPossibleSquares(x, n)
            w = np.argwhere(U)
            i_ = np.ravel_multi_index((w[:,0], w[:,1]), x.shape)
            
            m = len(i_)
            dx = np.abs(w[:,0].reshape(m,1) - w[:,0].reshape(1,m)) >= n
            dy = np.abs(w[:,1].reshape(m,1) - w[:,1].reshape(1,m)) >= n
            # collision matrix, true where squares (i,j) do not collide    
            C = dx | dy
            
            # find the cliques: all combinations of squares than can be placed toghether
    #         print('BK size=%i ...' % (m))
            ri = self.bronk(C, set(), set(range(0, C.shape[0])), set(), maxComb)
    #         print('...ok')
        
            for s in ri:
                if len(s) > 0:
    #                 r.add(frozenset(i_[np.array(list(s))]))
                    for L in reversed(range(1, len(s)+1)):
                        for subset in itertools.combinations(s, L):
                            r.add(frozenset(i_[np.array(subset)]))
                            if (len(r) >= maxComb):
    #                             print('...ok')
                                return (r, False)
             
    #     print('...ok')
        full = len(r) < maxComb
        return (r, full)
    
    
    def decomposeSquares(self, n, ni = 0):
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
        
    def registerSolution(self, x, solution):
        [h, m, dims] = self.hash_(x)
    #     np.array(x, dtype=np.int8)
        if (h is not None):
            i = np.ravel_multi_index((solution[0,:] - m[0], solution[1,:] - m[1]), dims)
            sol = np.vstack([i, solution[2,:]])
            self.cache[h] = sol
        return
    
    registerSolution.count = 0
     
    def place(self, x_, n, maxComb = [], 
              partialLength = 0,
              bestLength = sys.maxsize,
              tryPartitions = True,
              findIsolatedSingleTiles = True,
              maxDepth = 1000):
            
        shift = np.array([[0],[0],[0]], dtype=int16)
        optimalSolution = True
        
        if self.stop:
            return (None, None)
        
        if maxDepth == 0:
            optimalSolution = False
            squares = np.zeros((3, 0), dtype=int16)
        elif (n == 1):
            i = np.transpose(np.argwhere(x_))
            if (partialLength + i.shape[1] >= bestLength):
                return (None, None)
            
            l = np.ones(i.shape[1], dtype=int16)
            squares = np.vstack([i, l]) 
        else:
    
            squares = np.zeros((3, 0), dtype=int16)
            # shrink the puzzle
            u = np.argwhere(x_)    
            nRemainingTiles_ = u.shape[0]
    
            if (nRemainingTiles_ == 0):
                return (squares, optimalSolution)
    
            # we reduce n
            n = min(n, int(floor(sqrt(nRemainingTiles_))))
    
            if self.allowShrink:   
                m_ = np.min(u, 0)
                M_ = np.max(u, 0)
                dims_ = M_ - m_ + 1
                u0 = u - m_        
        
                x = np.zeros(dims_, dtype = bool)
                x[u0[:,0], u0[:,1]] = True
                shift = np.array([[m_[0]],[m_[1]],[0]], dtype=int16);
            else:
                x = np.array(x_, dtype = bool)
                    
            if self.allowPromptFindIsolatedSingleTiles and findIsolatedSingleTiles:
                # find 'isolated' 1x1 squared
                # finding them in early stages will help us to find a better partition and a better best-scenario bound
                xnm1False = np.hstack( [ np.ones((x.shape[0], 1), dtype=bool), x[:,:-1] == False] )
                xnp1False = np.hstack( [ x[:,1:] == False, np.ones((x.shape[0], 1), dtype=bool) ] )
                ynm1False = np.vstack( [ np.ones((1, x.shape[1]), dtype=bool), x[:-1,:] == False] )
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
    
            [h, m, dims] = self.hash_(x)
            if h in self.cache.keys():
                solution = self.cache[h]
                [ix, iy] = np.unravel_index(solution[0,:], dims)
                ix = ix + m[0]
                iy = iy + m[1]
                squares_i = np.vstack([ ix, iy, solution[1,: ]] )
                self.cacheHits += 1
                return (shift + np.hstack([ squares, squares_i ]), True)
    
            # computes the best scenario according only to the number of tiles on,
            # not the geometry
            bestScenario = self.decomposeSquares(nRemainingTiles_, n)
            
            # if the best possible scenario is worse than a known solution, we do
            # not continue
            if (partialLength + bestScenario >= bestLength):
                return (None, None)        
    
            tryPartitions &= self.allowPartitions
            partitions = [] if not tryPartitions else self.partition(x)
            partitionExists = len(partitions) > 1;
              
            if partitionExists:
                squares_nm1 = np.zeros((3, 0), dtype=int16)                 
                # the smallest sets before
                partitions = sorted(partitions, key=lambda x: len(x))                    
                subLength = 0
                for xi in partitions:
                    [squares_nm1_i, optimalSolution_i] = self.place(xi, n, maxComb, partialLength + subLength, bestLength, False, findIsolatedSingleTiles, maxDepth)
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
        
                maxComb_ = 1
                for k in range(0, len(maxComb)):
                    if np.isnan(maxComb[k]):
                        maxComb_ = sys.maxsize
                    else:
                        if (nRemainingTiles_ < maxComb[k]):
                            maxComb_ += 1
                     
                maxComb__ = 200
                (r, full) = self.combinations(x, n, maxComb__)
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
                            self.registerSolution(x, shift + squares)
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
                    if self.sigmaNoiseInScores > 0:
                        scores += np.random.normal(0.0, self.sigmaNoiseInScores, len(scores))
                    scores_i = np.argsort(scores)
    #                 scores_i = scores_i[scores[scores_i] < 0]
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
        
                        bestScenario = length + self.decomposeSquares(nRemainingTiles, n - 1)
                        if (partialLength + bestScenario >= bestLength):
                            continue
                                        
                        for index in np.transpose(squares_n):
                            self.fillSquare(x, index[0], index[1], n, False);
                                    
                        [squares_nm1, optimalSolution_nm1] = self.place(x, n - 1, maxComb, partialLength + length, bestLength, length > 0, findIsolatedSingleTiles, maxDepth - 1)                    
                            
                        for index in np.transpose(squares_n):
                            self.fillSquare(x, index[0], index[1], n, True);
        
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
                    [squares_nm1, optimalSolution] = self.place(x, n - 1, maxComb, partialLength, bestLength, tryPartitions, findIsolatedSingleTiles, maxDepth)
                    if (squares_nm1 == None):
                        return (None, None)
                    squares = np.hstack([ squares, squares_nm1 ])            
    
        if (optimalSolution):
            self.registerSolution(x_, shift + squares)
        return (shift + squares, optimalSolution)
    
    def hash_(self, x):
        if self.cachedSolutions:
            u = np.argwhere(x)
            m = np.min(u, 0)
            M = np.max(u, 0)
            u0 = u - m
            dims = M - m + 1
            tdims = tuple(dims)
            h = None
            if min(dims) > 1 and np.prod(dims) < 1400:
                h = hash((tdims, frozenset(np.ravel_multi_index((u0[:,0], u0[:,1]), tdims))))
            return (h, m, tdims)
        else:
            return (None, None, None)
    
    def populate(self, x, squares):
        r = np.zeros(x.shape, dtype=int16)
        for k in range(0, squares.shape[1]):
            i = squares[0:2, k]
            n = squares[2, k];
            assert(np.all(r[i[0]:i[0]+n, i[1]:i[1]+n] == 0))
            r[i[0]:i[0]+n, i[1]:i[1]+n] = k + 1
        return r
       
    def partition(self, x):
        
        result = []
        
        x = np.array(x, dtype = bool)
        while True:
            i = np.argwhere(x)        
            if (i.shape[0] == 0):
                break
         
            # we choose the closest tile to the center of the puzzle
            d = np.sum(np.abs(i - np.array(x.shape)/2), 1)
            i0 = np.argmin(d)
            i = i[i0, :]
            xi = np.zeros(x.shape, dtype=bool)
            xi[tuple(i)] = True
            xiSize_nm1 = 0
            xiSize = 1
            
            while (xiSize > xiSize_nm1):
                # neighbors
                xn = np.hstack( [ np.zeros((xi.shape[0], 1), dtype=bool), xi[:,:-1]] )
                xn |= np.hstack( [ xi[:,1:], np.zeros((xi.shape[0], 1), dtype=bool) ] )
                xn |= np.vstack( [ np.zeros((1, xi.shape[1]), dtype=bool), xi[:-1,:]] )
                xn |= np.vstack( [ xi[1:, :], np.zeros((1, xi.shape[1]), dtype=bool) ] )
                xn &= x
                xi |= xn
                xiSize_nm1 = xiSize
                xiSize = np.argwhere(xi).shape[0]
        
            result.append(xi)
            x ^= xi
            
        return result
        
    def solve(self, x, bestLength = sys.maxsize):
    
        self.cache = {}
        self.cacheHits = 0
        self.stop = False
    
        if self.plots:
            fig = plt.figure(1)
            ax = fig.add_subplot(121)
            ax.imshow(x, interpolation='none')
            ax = fig.add_subplot(122)
            ax.imshow(x, interpolation='none')
            plt.show(block=False)
        
        # partition(x)
            
        squares_i = None
        squares = None
        
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
          
        
    #     timeLimit = max(10.0, 0.4*(np.sqrt(np.prod(np.array(x.shape))) - 15))
    #     print('time limit for this puzzle %.2fs' % (timeLimit))
        timeLimit = self.timeLimit
        t0 = time.time()
        def checktime():
            elapsed0 = time.time() - t0
            if (elapsed0 > timeLimit):
                self.stop = True
    #             print('reached time limit, quiting ...')
            else:
                threading.Timer(0.2, checktime).start()
        
        checktime()
        
        for k in range(0,len(maxCombThresholds)):
            if self.stop:
                break
    #         print('iteration %i' % (k))
            for n in range(0, 4 if self.sigmaNoiseInScores > 0 else 1):
                if self.stop:
                    break
                t = time.time()
                [squares_i, optimalSolution] = self.place(x, min(x.shape), maxCombThresholds[k], 0, bestLength, True, True)
                elapsed0 = time.time() - t0
                if (squares_i is not None) and squares_i.shape[1] < bestLength:
                    timeLimit = max(timeLimit, elapsed0 * (1 + self.timeLimitGrowFactor))
                    bestLength = squares_i.shape[1]
                    squares = squares_i
                    if self.plots:
                        xi = self.populate(x, squares)
                        ax = fig.add_subplot(122)
                        ax.imshow(xi, interpolation='none')
                        fig.canvas.draw()
            
    #                 print('Cache has %i entries, got %i hits' % (len(hash_.table), hash_.hits))
    #                 print("Found solution with %i squares, total elapsed %0.2fs " % (squares.shape[1], elapsed0))
                    xi = self.populate(x, squares)
                    if np.any(xi[x] == 0):
                        assert(False)
        
        elapsed0 = time.time() - t0
    #     if squares is None:
    #         print("Solution not found")
    #     else:
    #         print("Found solution with %i squares, total elapsed %0.2fs " % (squares.shape[1], elapsed0))
        
        return squares
    
#plt.show()

