import sys
sys.path.append(
    '..'
)
from data_preprocess.point2road import dist
from numpy import array, zeros, argmin, inf, equal, ndim
from data_preprocess.point2road import MIN_LINE_LENGTH

MIN_LINE_LENGTH = MIN_LINE_LENGTH - 1


def distances(x1,x2):
    return dist(x1,x2)

def DTW(s1,s2):
    line_saver = None
    if dist(s1[0],s2[0]) > 0.001:
        return 1,None
    r, c = len(s1), len(s2)
    D0 = zeros((r+1,c+1))
    D0[0,1:] = inf
    D0[1:,0] = inf
    D1 = D0[1:,1:]
    for i in range(r):
        for j in range(c):
            D1[i,j] = distances(s1[i],s2[j])
    M = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i,j] += min(D0[i,j],D0[i,j+1],D0[i+1,j])
    i,j = array(D0.shape) - 2
    p,q = [i],[j]
    while(i>0 or j>0):
        tb = argmin((D0[i,j],D0[i,j+1],D0[i+1,j]))
        if i == r - 1:
            line_saver = s2[j]
        if tb==0 :
            i-=1
            j-=1
        elif tb==1 :
            i-=1
        else:
            j-=1
        p.insert(0,i)
        q.insert(0,j)
    return D1[-1,-1],line_saver

if __name__ == '__main__':
    s1 = [[1, 2], [3, 4], [5, 5], [5, 4]]
    s2 = [[3, 4], [5, 5], [5, 4]]
    print(DTW(s1,s2))