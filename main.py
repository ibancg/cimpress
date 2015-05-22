import numpy as np
from puzzleSolver import PuzzleSolver
from random import randint

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
    

# dealing with a graph as list of lists 
# graph = np.array([[1,1,0,0,1,1],[1,1,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]])
# c = partition( np.array(graph, dtype=bool) )
# print(c)
# print(len(c))

# if True:
#     exit(0)
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

  
# x = np.array([
#               [0,0,1,1,1,1,1,1,1,0,0,1],
#               [1,0,1,1,1,1,1,1,1,1,1,1],
#               [1,1,1,1,1,1,1,1,1,1,1,1],
#               [1,1,1,1,1,1,1,1,1,1,1,1],
#               [1,1,1,1,1,1,1,1,1,1,1,1],
#               [1,1,1,1,1,1,1,1,1,1,1,1],
#               [1,1,1,1,1,1,1,1,1,1,1,1],
#               [1,1,1,1,1,1,1,1,1,1,1,1],
#               [0,0,0,0,1,1,1,1,1,1,1,0],
#               [0,0,0,0,1,1,1,1,1,1,1,0],
#               [0,0,0,0,1,1,1,1,1,1,0,0],
#               [0,0,0,0,1,1,1,1,1,1,1,1],
#               ])
#      
# x = (x == 1)
#     
# solve(x)

# cacheFileName = 'cache.txt'
# if os.path.isfile(cacheFileName):
#     with open(cacheFileName, 'rb') as f:
#         print('Reading cache from file ...')
#         hash_.table = load(f)
#         print('... done')
#            
#      
# 
# puzzles = glob.glob("puzzles/*.npy")
# for puzzle in puzzles:
#     print(puzzle)
#     solution = puzzle[:-4] + str('.sol')
#          
#     bestSize = sys.maxsize
#     if os.path.isfile(solution):
#         bestSol = np.loadtxt(solution)
#         bestSize = bestSol.shape[1]
#         print('best solution has %i squares' % (bestSize))
#               
#     print(solution)
#     # x = generate(30, 30, 25, 15)
#     x = np.load(puzzle)
#     # # print(np.sum(x))
#     # x = x > 0
#     # np.savetxt('puzzle.txt', x, fmt='%i')
#     sol = solve(x, bestSize)
#     if (sol is not None):
#         np.savetxt(solution, sol, fmt='%i')

#            
#    
# x = generate(100, 100, 60, 30)
# x = np.loadtxt('puzzle.txt')
# # print(np.sum(x))
# x = x > 0
# np.savetxt('puzzle.txt', x, fmt='%i')
# solve(x)
#     print('Saving cache to file ...')
#     with open(cacheFileName, 'wb') as f:
#         dump(hash_.table, f)    
#     print('... done')

solver = PuzzleSolver()
solver.plots = False
solver.timeLimit = 9.4

# x = generate(100, 100, 60, 30) > 0
x = np.load('/tmp/puzzle.npy')
y = solver.solve(x)
if y is None:
    y = np.zeros((0, 0))
s = ' '.join('%i'%F for F in np.hstack([y.shape[0], y.shape[1], np.reshape(y, y.shape[0]*y.shape[1])]) )
print(s)
