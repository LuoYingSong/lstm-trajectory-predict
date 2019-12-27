import sys
sys.path.append(
    '..'
)
from data_preprocess.point2road import dist
from numpy import array, zeros, argmin, inf, equal, ndim
# from scipy.spatial.distance import cdist
#如果比较的是二维数组，则用欧几里得距离


def distances(x1,x2):
    return dist(x1,x2)

def DTW(s1,s2):
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
    #代码核心，动态计算最短距离


    i,j = array(D0.shape) - 2
    #最短路径
    # print i,j
    p,q = [i],[j]
    while(i>0 or j>0):
        tb = argmin((D0[i,j],D0[i,j+1],D0[i+1,j]))
        if tb==0 :
            i-=1
            j-=1
        elif tb==1 :
            i-=1
        else:
            j-=1
        p.insert(0,i)
        q.insert(0,j)

    return D1[-1,-1]

if __name__ == '__main__':
    s1 = [[1, 2], [3, 4], [5, 5], [5, 4]]
    s2 = [[3, 4], [5, 5], [5, 4]]
    print(DTW(s1,s2))