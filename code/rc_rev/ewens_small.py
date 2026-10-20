"""Worked residue example: Ewens likelihood n=4, K=2 under the half horseshoe PH(0,1,2,1).
Exact partial fractions with sympy, checked by numerical integration with mpmath."""
import sympy as sp
import mpmath as mp
mp.mp.dps = 30
t = sp.symbols('theta', positive=True)

def residue_integral(num, poles):
    """poles: dict r -> multiplicity, denominator prod (t+r)^mu. Returns exact integral and coefficients."""
    den = sp.prod([(t + r) ** mu for r, mu in poles.items()])
    R = num / den
    total = 0
    coef = {}
    for r, mu in poles.items():
        h = sp.simplify(R * (t + r) ** mu)
        for j in range(1, mu + 1):
            g = sp.diff(h, t, mu - j).subs(t, -r) / sp.factorial(mu - j)
            g = sp.nsimplify(sp.simplify(g))
            coef[(r, j)] = g
            if j == 1:
                total += -g * sp.log(r)
            else:
                total += g / ((j - 1) * sp.Integer(r) ** (j - 1))
    s1 = sum(v for (r, j), v in coef.items() if j == 1)
    return sp.nsimplify(sp.expand(total)), coef, s1, R

poles = {1: 2, 2: 2, 3: 1}
for name, num in [("C", t), ("first moment", t ** 2), ("second moment", t ** 3)]:
    val, coef, s1, R = residue_integral(num, poles)
    q = mp.quad(sp.lambdify(t, R, 'mpmath'), [0, 1, 10, mp.inf])
    print(name, val, float(val), "quad", q, "coef", coef, "sum gamma_i1", s1)

C = residue_integral(t, poles)[0]
M1 = residue_integral(t ** 2, poles)[0]
M2 = residue_integral(t ** 3, poles)[0]
print("posterior mean", float(M1 / C), "E theta^2", float(M2 / C), "sd", float(sp.sqrt(M2 / C - (M1 / C) ** 2)))

# predictive probability of a new allele at draw 5: E[theta/(theta+4)]
val, coef, s1, R = residue_integral(t ** 2, {1: 2, 2: 2, 3: 1, 4: 1})
q = mp.quad(sp.lambdify(t, R, 'mpmath'), [0, 1, 10, mp.inf])
print("pred numerator", val, float(val), "quad", q, "coef", coef)
print("P(new allele at draw 5)", sp.simplify(val / C), float(val / C))

# marginal likelihood P(K=2) under prior: |s(4,2)| = 11, prior normaliser log 2
print("P(K_4=2) marginal", float(11 * C / sp.log(2)), "under theta=1:", 11 / 24)
# all K: P(K=k) = |s(4,k)| int theta^k/[theta]^4 p(theta)
st = {1: 6, 2: 11, 3: 6, 4: 1}
tot = 0
for kk, s in st.items():
    f = lambda x, kk=kk: x ** kk / (x * (x + 1) * (x + 2) * (x + 3)) / ((x + 1) * (x + 2) * mp.log(2))
    pk = s * mp.quad(f, [0, 1, 10, mp.inf])
    tot += pk
    print(" prior predictive P(K=%d)=%.6f" % (kk, pk))
print(" total", tot)

# MLE for n=4, K=2: sum_{i=0}^{3} theta/(theta+i) = 2
mle = mp.findroot(lambda x: sum(x / (x + i) for i in range(4)) - 2, 1)
print("MLE", mle, "plug-in predictive", mle / (mle + 4))
# posterior median by bisection on the cdf
Rf = lambda x: x / ((x + 1) ** 2 * (x + 2) ** 2 * (x + 3))
Cn = mp.quad(Rf, [0, mp.inf])
cdf = lambda A: mp.quad(Rf, [0, A]) / Cn
med = mp.findroot(lambda A: cdf(A) - 0.5, 1)
print("posterior median", med)
for qq in [0.025, 0.975]:
    print(" quantile", qq, mp.findroot(lambda A: cdf(A) - qq, 1 if qq < .5 else 8))
