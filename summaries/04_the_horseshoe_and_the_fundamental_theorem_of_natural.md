# Chapter 4. The Horseshoe and the Fundamental Theorem of Natural Selection

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 4. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509850. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

This chapter reads the Fundamental Theorem through estimation. Once the average effects are the coefficients of a weighted least squares regression of genotypic fitness on allelic dosage, the theorem is an algebraic identity and its empirical content becomes the estimation of a regression with more loci than individuals. The plug in estimate of the additive variance is then inflated by the number of loci times the noise variance over the sample size, and the implied rate of adaptation is overstated. The chapter estimates the average effects under a horseshoe prior and reads the theorem through the shrinkage weights. Three results are proved: any shrinkage rule gives a conservative estimate of the additive variance; oracle shrinkage tuned for the average effects necessarily understates it; and the coherent estimator is the posterior mean of the quadratic functional, not the quadratic of the posterior mean. A simulation with 400 loci and 300 individuals reports the naive estimator, ridge and the horseshoe, and the chapter then treats the secondary theorem, bilinear and general quadratic forms with linkage disequilibrium, truncation on measured fitness, and ascertainment of survivors.

## From the literature

The partial change reading of the theorem is due to Price (1972) and Ewens (1989), with the account of Ewens and Lessard (2015). Fisher (1930) is the original. The horseshoe prior is Carvalho, Polson and Scott (2010). The secondary theorem is Robertson's.

## New results

<!-- results:begin -->
- [Bias of the plug in estimate of additive variance](../RESULTS.md#bias_of_the_plug_in) (Proposition 4.4, page 133; proved): With orthogonal columns, the plug in estimator of the additive variance is inflated by the number of loci times the noise variance over the sample size. New in this book.
- [Shrinkage is conservative](../RESULTS.md#shrinkage_is_conservative) (Proposition 4.9, page 139; proved): Any shrinkage weights give an estimate of the additive variance no larger than the plug in estimate. New in this book.
- [Oracle shrinkage understates the additive variance](../RESULTS.md#oracle_shrinkage_understates_the_additive_variance) (Proposition 4.10, page 140; proved): Shrinkage tuned to estimate each average effect as well as possible necessarily understates the additive variance, by an explicit amount. New in this book.
- [The coherent estimator is the posterior mean of the quadratic](../RESULTS.md#the_coherent_estimator) (Proposition 4.11, page 141; proved): The posterior mean of the additive variance is the sum of the squared posterior means plus the posterior variances, so it is never below the plug in of the posterior means. New in this book.
<!-- results:end -->

## Key terms

Fundamental Theorem of Natural Selection, horseshoe prior, shrinkage, rate of adaptation, additive genetic variance estimation, whole genome regression, p greater than n, plug in bias, oracle shrinkage, Robertson secondary theorem, truncation selection, ascertainment, linkage disequilibrium, quadratic functional
