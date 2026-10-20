"""Normal toy model: theta ~ N(0, s0^2), y_i ~ N(theta, 1), n obs. ABC with summary ybar and
uniform tolerance eps targets the posterior given ybar + V, V ~ U(-eps, eps). Compare exact ABC
posterior variance (by quadrature) with the Gaussian approximation 1/(1/s0^2 + 1/(1/n + eps^2/3))."""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
n, s0 = 100, 1.0
for s in (0.0, 1.0):
    for eps in (0.0, 0.05, 0.1, 0.2):
        if eps == 0:
            v = s0**2 / (1 + n * s0**2); print(s, eps, "exact posterior var", v); continue
        lik = lambda t: norm.cdf((s + eps - t) * np.sqrt(n)) - norm.cdf((s - eps - t) * np.sqrt(n))
        f = lambda t: norm.pdf(t, 0, s0) * lik(t)
        Z = quad(f, -6, 6, points=[s])[0]
        m1 = quad(lambda t: t * f(t), -6, 6, points=[s])[0] / Z
        m2 = quad(lambda t: t * t * f(t), -6, 6, points=[s])[0] / Z
        approx = 1 / (1 / s0**2 + 1 / (1 / n + eps**2 / 3))
        print(s, eps, "ABC var", m2 - m1**2, "approx", approx, "ratio to exact", (m2 - m1**2) / (s0**2 / (1 + n * s0**2)))
print("y1 summary factor", (1 + n * s0**2) / (1 + s0**2))
