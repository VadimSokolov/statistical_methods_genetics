# Spectrum estimators on two classic frequency of frequency data sets
import numpy as np
from scipy.special import gammaln
def ratio(j,s): return np.exp(gammaln(j-s)-gammaln(j+1)-gammaln(1-s))
# Efron and Thisted (1976): Shakespeare, first ten classes; n=884647 tokens, K=31534 types
A=np.array([14376,4343,2292,1463,1043,837,638,519,430,364]); n=884647; K=31534
s1=A[0]/K; s2=1-2*A[1]/A[0]
print('Shakespeare A1/K',s1,' 1-2A2/A1',s2,' logK/logn',np.log(K)/np.log(n))
for j in range(1,11):
    print(j,A[j-1],'obs A_j/A_1 %.4f'%(A[j-1]/A[0]),' pred(s1) %.4f'%ratio(j,s1)/1 if False else '', ' pred(s1) %.4f  pred(s2) %.4f'%(ratio(j,s1)/ratio(1,s1),ratio(j,s2)/ratio(1,s2)))
# predicted A_j from A1/K: A_j = K q_j, q_j=s Gamma(j-s)/(Gamma(1-s) j!)
for s in (s1,s2):
    q=s*ratio(np.arange(1,11),s)
    print('s=%.4f'%s,' predicted A_j = K q_j:',np.round(K*q).astype(int), ' sum q_1..10 %.4f'%q.sum())
# Corbet (Fisher 1943), classes 1..24
C=np.array([118,74,44,24,29,22,20,19,20,15,12,14,6,12,6,9,9,6,10,10,11,5,3,3])
print('Corbet K',C.sum(),' n',(np.arange(1,25)*C).sum(),' A1/K',C[0]/C.sum(),' 1-2A2/A1',1-2*C[1]/C[0])
# ESF(164) at n=3306 exact means of A1, A2
th=163.99; nn=3306
def EA(j): return th/j*np.exp(gammaln(nn+1)-gammaln(nn-j+1)+gammaln(th+nn-j)-gammaln(th+nn))
print('ESF EA1,EA2',EA(1),EA(2),' ratio',EA(2)/EA(1))
