"""Series expansion of the Galton Watson survival probability with Poisson(1+s) offspring."""
import sympy as sp
s = sp.symbols('s')
c = sp.symbols('c1:5')
pi = sum(c[i]*s**(i+1) for i in range(4))
eq = sp.series(pi - 1 + sp.exp(-(1+s)*pi), s, 0, 6).removeO()
sol = sp.solve([sp.expand(eq).coeff(s, k) for k in range(2, 6)], c, dict=True)
print(sol)
