
from threading import Thread
import commands
import numpy as np

LOCALHOST = 'localhost'

class thread_it(Thread):
    def __init__ (self, host):
        Thread.__init__(self)
        self.host = host
        self.solution = None
        
    def run(self):
#         call(["ssh", "%s" % (self.host), "ls", "-l"])
#         str = commands.getstatusoutput('ssh %s cd ~/workspace-p; python main' % (self.host))
        if self.host == LOCALHOST:
            output  = commands.getstatusoutput('python main.py')
        else:
            output = commands.getstatusoutput('ssh %s python /tmp/main.py' % (self.host))
        r = np.fromstring(output[1], dtype=int, sep=' ')
        self.solution = np.reshape(r[2:], (r[0], r[1]))
        
    def sol(self):
        return self.solution
    
class DistributedSolver:
    def __init__ (self):
        self.threads = {}

    def addHost(self, host, nThreads):
        self.threads[host] = nThreads
        
    def start(self):
        for host in self.threads.keys():
            if host is not 'localhost':
                commands.getstatusoutput('scp main.py* puzzleSolver.py* %s:/tmp/' % (host))
                
    def solve(self, x):

        np.save('/tmp/puzzle', x)
        for host in self.threads.keys():
            if host is not 'localhost':
                commands.getstatusoutput('scp /tmp/puzzle.npy %s:/tmp' % (host))
        
        threads = []
        for host, nThreads in self.threads.items():
            for i in range(0, nThreads):
                threads.append(thread_it(host))

        for t in threads:
            t.start()
         
        for t in threads:
            t.join()

        sol = np.zeros((3, 0))
        for t in threads:
            soli = t.sol()
            print(t.host + ': ' + str(soli.shape[1]))
            if (soli.shape[1] == 0):
                continue
            if (sol.shape[1] == 0) or (soli.shape[1] < sol.shape[1]):
                sol = soli
               
        return sol