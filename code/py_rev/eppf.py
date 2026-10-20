# Exact Pitman EPPF for n=3,4, and partition type probabilities
from math import gamma, prod
from fractions import Fraction as Fr
def rising(x,m):
    r=1
    for i in range(m): r*=x+i
    return r
def eppf(blocks,s,t):
    k=len(blocks); n=sum(blocks)
    num=prod([t+i*s for i in range(1,k)]) if k>1 else 1
    return num/rising(t+1,n-1)*prod([rising(1-s,b-1) for b in blocks])
types={3:[((3,),1),((2,1),3),((1,1,1),1)],
       4:[((4,),1),((3,1),4),((2,2),3),((2,1,1),6),((1,1,1,1),1)]}
for (s,t) in [(Fr(0),Fr(1)),(Fr(0),Fr(2)),(Fr(1,2),Fr(1)),(Fr(1,2),Fr(0)),(Fr(1,4),Fr(2))]:
    print('sigma,theta',s,t)
    for n in (3,4):
        tot=0
        for b,c in types[n]:
            v=eppf(b,s,t); tot+=c*v
            print(n,b,c,'p=',v,float(v),' type prob',c*v,float(c*v))
        print(' sum',tot)
