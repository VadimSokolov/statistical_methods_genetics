# Chapter 11. Learning to Coalesce

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 11. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509850. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

Kingman's coalescent is the historical home of likelihood free inference: simulation is trivial and likelihood evaluation is not. This chapter studies generative Bayesian computation for it. Under the infinite sites model the number of segregating sites is a sum of independent geometric variables, one per level of the coalescent, which gives Watterson's mean and variance, an exactly computable likelihood for the site count, and a Bernoulli geometric duality with the Ewens block count. The haplotype partition has the Ewens law, and all these statistics carry Fisher information of order the log of the sample size over theta. The number of segregating sites is not sufficient for theta, shown by an explicit calculation for three and four sequences, so a summary must be learned and not derived. A ceiling is proved: any DeepSets encoder built from per site permutation invariant features factors through the site frequency spectrum, a strict coarsening of the maximal invariant, and the haplotype count escapes that ceiling. Experiments with n equal to 30 compare learned posteriors with an exact posterior given S, including a growth model, and simulation based calibration shows residual miscalibration of the learned maps.

## From the literature

Kingman's coalescent is Kingman (1982). The geometric structure of segregating sites is implicit in Watterson (1975). The Feller coupling is classical. Likelihood free inference for the coalescent starts with Tavare and coauthors (1997) and Beaumont and coauthors (2002). DeepSets encoders are due to Zaheer and coauthors.

## New results

<!-- results:begin -->
- [Bernoulli geometric duality between the Ewens block count and segregating sites](../RESULTS.md#bernoulli_geometric_duality) (Corollary 11.7, page 439; proved): The number of blocks of the Ewens sampling formula has the same law as one plus the number of coalescent levels that carry at least one mutation. New in this book.
- [The number of segregating sites is not sufficient for theta, likelihood argument](../RESULTS.md#is_not_sufficient_for) (Proposition 11.12, page 443; proved): The conditional law of the infinite sites data given the site count depends on theta through the latent tree, shown explicitly for three and four sequences. New in this book.
- [DeepSets encoders factor through the site frequency spectrum](../RESULTS.md#deepsets_encoders_factor_through_the_spectrum) (Theorem 11.16, page 445; proved): Any encoder built from per site permutation invariant features depends on the data only through the site frequency spectrum, which caps what such an encoder can learn. New in this book.
- [The spectrum is a strict coarsening of the gene tree](../RESULTS.md#spectrum_is_a_strict_coarsening) (Corollary 11.17, page 445; proved): The site frequency spectrum is strictly coarser than the gene tree, which is the maximal invariant, so no per site DeepSets encoder can recover what the spectrum loses. New in this book.
- [The haplotype count escapes the site frequency spectrum](../RESULTS.md#haplotype_count_escapes_the_spectrum) (Proposition 11.18, page 446; proved): The haplotype count is invariant under relabelling of lineages and sites but is not a function of the site frequency spectrum. New in this book.
<!-- results:end -->

## Key terms

Kingman coalescent, infinite sites model, segregating sites, Watterson estimator, geometric decomposition, Feller coupling, haplotype count, site frequency spectrum, DeepSets, permutation invariant encoder, generative Bayesian computation, simulation based calibration, demographic growth, likelihood free inference
