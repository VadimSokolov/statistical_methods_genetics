import mpmath as mp
mp.mp.dps=40
for n in [25,40]:
    k=3
    f=lambda x: x**(k-1)/mp.fprod([x+i for i in range(1,n)])/((x+0.5)*(x+1.5))
    pts=[0]+[mp.mpf(2)**j for j in range(-6,14)]+[mp.inf]
    print(n, mp.quad(f,pts,maxdegree=10))
