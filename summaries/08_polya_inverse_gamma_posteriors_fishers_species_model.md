# Chapter 8. Pólya Inverse Gamma Posteriors, Fisher's Species Model, and Good's Predictive Rule for the Ewens Sampling Problem

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 8. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509613. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

This chapter gives an inferential account of Fisher's species model, the Ewens sampling formula and Good's predictive rule, connected by Polya inverse gamma augmentation. The Ewens richness likelihood is a regular one parameter natural exponential family in log theta with sufficient statistic K, the number of species, so the maximum likelihood equation is pure moment matching. A Bernstein von Mises argument shows that the Bayes estimator agrees with the maximum likelihood estimate to relative order one over K and the credible interval with the Wald interval, so in this sense the Bayes estimator of theta adds little. Fisher's two parameters map onto the two PIG representations: the richness needs only the gamma ratio block, and the shape that governs departure from the log series needs the full PIG(0) engine. Good's estimator, once given the Ewens model, is the Ewens predictive rule, and the predictive law of the number of new species is a sum of independent Bernoulli variables. The posterior of theta has a Pareto tail under an improper flat prior, which limits its moments. A three block Gibbs sampler is given.

## From the literature

Fisher, Corbet and Williams (1943) introduced the index of diversity and the log series. Good (1953) and Good and Toulmin (1956) asked the predictive questions. McCullagh (2016) shows that Fisher's two parameter Poisson process contains the Ewens law. Polya inverse gamma augmentation is He, Polson and Xu (2019). The Ewens sampling formula is Ewens (1972).

## New results

<!-- results:begin -->
- [Inverse gamma series for PIG(0)](../RESULTS.md#inverse_gamma_series_for_pig0) (Proposition 8.7, page 313; proved): The Polya Inverse Gamma variable that turns a reciprocal gamma factor into a Gaussian kernel is an explicit series of independent inverse gamma variables, with finite mean and infinite variance. New in this book.
- [Good's rule is the Ewens predictive](../RESULTS.md#goods_rule_is_the_ewens_predictive) (Proposition 8.9, page 316; proved): Under the Ewens model, Good's predictive probability for a species seen r times is r over n plus theta, and the unseen mass is theta over n plus theta. New in this book.
- [Predictive law of the number of new species](../RESULTS.md#predictive_law_of_new_species) (Proposition 8.11, page 317; proved): The number of new species in m further specimens is a sum of independent Bernoulli variables given theta, with explicit mean and variance in terms of the digamma function. New in this book.
- [Asymptotic redundancy of the Bayes estimator of theta](../RESULTS.md#asymptotic_redundancy) (Proposition 8.13, page 319; proved): For the Ewens richness likelihood, the posterior of the log concentration is asymptotically normal, and the Bayes estimator and credible interval agree with the maximum likelihood estimator and Wald interval to leading order. New in this book.
- [Tail of the concentration posterior](../RESULTS.md#tail_of_the_concentration_posterior) (Proposition 8.15, page 321; proved): Under a gamma type prior with no exponential decay, the posterior of theta has a Pareto tail, so its moments exist only when the number of species is small enough. New in this book.
<!-- results:end -->

## Key terms

Fisher species model, index of diversity, log series, Ewens sampling formula, Good Turing, Good Toulmin, unseen species, Polya inverse gamma, data augmentation, natural exponential family, Bernstein von Mises, Gibbs sampler, richness estimation, Pareto tail posterior
