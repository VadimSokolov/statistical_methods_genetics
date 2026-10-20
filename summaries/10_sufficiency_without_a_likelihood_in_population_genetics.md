# Chapter 10. Sufficiency Without a Likelihood in Population Genetics

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 10. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509850. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

Approximate Bayesian computation was invented in population genetics because coalescent likelihoods are intractable, and it has two defects: the analyst chooses summary statistics, so the object computed is the posterior given the summary and not the posterior given the data, and the analyst chooses a tolerance. This chapter replaces the apparatus with a learned transport map. By noise outsourcing the posterior is a deterministic function of the data and an independent uniform, and that function is the population minimiser of an expected check loss, so a regression with a quantile network can fit it. The central theorem is that a map factoring through a bottleneck summary and attaining the Bayes risk makes the summary sufficient, so sufficiency becomes a test on risks; a converse and a bound by the Wasserstein distance between the posterior and its summary based version complete the picture. The chapter treats Ewens's sufficiency of the number of alleles and proves that Watterson's number of segregating sites is not sufficient for the mutation rate. Experiments report learned posteriors against exact ones, including a repair of location scale structure that works in some seeds and fails in others.

## From the literature

Rejection approximate Bayesian computation in genetics is due to Tavare and coauthors (1997) and Beaumont and coauthors (2002). Noise outsourcing is a standard lemma. The check loss is the loss of quantile regression. Ewens (1972) showed that the number of alleles is sufficient for theta. Watterson (1975) is the source of the segregating sites statistic.

## New results

<!-- results:begin -->
- [The check loss identifies the posterior](../RESULTS.md#the_check_loss_identifies_the_posterior) (Theorem 10.8, page 394; proved): Minimizing the expected check loss over maps of a uniform variable and the data recovers the posterior quantile function exactly. New in this book.
- [A bottleneck that attains the Bayes risk is sufficient](../RESULTS.md#a_bottleneck_that_attains_the_bayes_risk_is_sufficient) (Theorem 10.12, page 399; proved): If a learned map that factors through a summary of the data attains the minimal check loss, the summary is sufficient, so sufficiency can be tested by comparing risks. New in this book.
- [Excess risk is a posterior distance](../RESULTS.md#excess_risk_is_a_posterior_distance) (Theorem 10.16, page 402; proved): The extra check loss of a summary based map is sandwiched between multiples of the expected squared Wasserstein distance between the posterior and its summary based version. New in this book.
- [Converse of the bottleneck sufficiency theorem](../RESULTS.md#converse) (Proposition 10.21, page 404; proved): A Bayes sufficient statistic always admits a map through it that attains the minimal risk. New in this book.
- [The number of segregating sites is not sufficient for theta](../RESULTS.md#suff_prop_notsuff) (Proposition 10.25, page 412; proved): Given a single segregating site, the chance that it is carried by n minus 1 sequences changes with theta, so the count of sites is not sufficient. New in this book.
<!-- results:end -->

## Key terms

approximate Bayesian computation, likelihood free inference, sufficiency, Bayes sufficiency, noise outsourcing, check loss, quantile regression, quantile neural network, Wasserstein distance, generative Bayesian computation, Ewens sampling formula, segregating sites, Watterson, site frequency spectrum
