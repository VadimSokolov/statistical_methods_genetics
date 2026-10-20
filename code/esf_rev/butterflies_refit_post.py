"""Post processing of butterflies_refit.json for the esf chapter (item esf_a-06).

Prints, at the fitted two parameter values: E K_n of the finite pool (the exercise on
Remark finite, by the Beta formula m{1 - ((m-1)a)_n/(ma)_n}); the expected contribution of
the classes j > 24 to the Good and Toulmin alternating sum under both fits; and the
chi squared reference values used for the likelihood ratio statistic.
Run after butterflies_refit.py: python3 -I code/esf_rev/butterflies_refit_post.py
"""
import json
import os
import sys
import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from butterflies_refit import EA_py, EK_py  # noqa: E402

R = json.load(open(os.path.join(_HERE, "butterflies_refit.json")))
tp, ew = R["twoparam"], R["ewens"]
al, th, n = tp["alpha"], tp["theta"], tp["n"]
a = -al; m = th / a
EK = m * (1 - np.exp(gammaln((m - 1) * a + n) - gammaln((m - 1) * a) - gammaln(m * a + n) + gammaln(m * a)))
print("two parameter fit alpha %.4f theta %.2f n %d m %.1f" % (al, th, n, m))
print("E K_n finite pool formula %.2f   EK_py %.2f" % (EK, EK_py(al, th, n)))
EA24 = EA_py(al, th, n, np.arange(1, 25)).sum()
print("two parameter expected species <= 24: %.2f, > 24: %.2f" % (EA24, EK - EA24))
for name, (aa, tt, nn) in (("Ewens", (0.0, ew["theta"], ew["n"])), ("two parameter", (al, th, n))):
    j = np.arange(25, nn + 1)
    print(name, "expected alternating sum over j > 24: %.2f" % ((-1.0) ** (j + 1) * EA_py(aa, tt, nn, j)).sum())
LR = 2 * (tp["loglik"] - ew["loglik"])
print("LR %.2f  chi2_1 99 percent point %.2f  p value %.4f" % (LR, chi2.ppf(0.99, 1), chi2.sf(LR, 1)))
