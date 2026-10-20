# Chapter 5. Many Parameter Models in Population Genetics

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 5. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509850. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

Population genetics has become a many parameter subject, and this chapter asks what that does to the quantities of the Fundamental Theorem. Fisher's average effect, as restated by Ewens (2024), is a weighted least squares projection. The plug in additive variance from a linear smoother carries a noise term, which for the unregularised fit is an inflation of exactly p sigma squared over N, and the inflation stops growing at the interpolation threshold. The natural prior for such problems is a species sampling prior. The chapter proves that the Beta law of the horseshoe shrinkage weight is the law of the first stick breaking weight of a two parameter GEM family, which ties shrinkage to the Ewens Pitman partition. It derives the allele frequency spectrum induced by a shrinkage prior, Fisher's log series as the special case, and the effective model size, whose growth rate matches the number of species under the matching Poisson Dirichlet law with constants that differ by a factor between about 0.89 and 1. The two constructions fluctuate differently: the shrinkage count concentrates while the species count has a Mittag Leffler limit.

## From the literature

The Hoppe urn is Hoppe (1984). The Poisson Kingman partition formula and the Pitman Yor size biased results are due to Pitman and to Pitman and Yor (1997). Ewens's sufficiency of the number of alleles is Ewens (1972). The horseshoe is Carvalho, Polson and Scott (2010).

## New results

<!-- results:begin -->
- [Inflation of Fisher's fitted projection](../RESULTS.md#inflation_of_fishers_fitted_projection) (Corollary 5.6, page 167; proved): The unregularized fit of the projection that defines the average effects inflates the estimated additive variance by exactly the number of parameters times the noise variance over the sample size. New in this book.
- [Saturation of the optimism past the interpolation threshold](../RESULTS.md#saturation_of_the_optimism) (Corollary 5.10, page 174; proved): The inflation of the fitted additive variance grows linearly in the number of parameters up to the sample size and is constant afterwards. New in this book.
- [Shrinkage weights are stick breaking weights](../RESULTS.md#shrinkage_weights_are_stick_breaking_weights) (Theorem 5.15, page 180; proved): The Beta law of the horseshoe type shrinkage weight is exactly the law of the first stick breaking weight of the two parameter GEM family, which links shrinkage priors to Ewens Pitman partitions. New in this book.
- [Spectrum and shrinkage profile](../RESULTS.md#spectrum_and_shrinkage_profile) (Theorem 5.18, page 181; proved): The mean intensity of the atoms of a random discrete distribution is the density of a size biased pick divided by x, which gives the expected allele frequency spectrum of a shrinkage prior from the density of its shrinkage weight. New in this book.
- [Discovery rates of shrinkage priors](../RESULTS.md#discovery_rates_of_shrinkage_priors) (Corollary 5.22, page 182; proved): A global local prior with a Beta shrinkage weight discovers a number of signals that grows like a power of the sample size, n to the one half for the horseshoe, or only logarithmically when a equals one. New in this book.
- [Effective model size of a shrinkage prior](../RESULTS.md#effective_model_size) (Theorem 5.23, page 183; proved): The expected number of coefficients that survive the penalty grows like a power of the sample size, or like a logarithm when a equals one, at the canonical global scale. New in this book.
- [Rates agree between effective model size and number of species](../RESULTS.md#the_rates_agree_and_the_constants_nearly_do) (Corollary 5.24, page 183; proved): The growth rates of the effective model size and of the expected number of blocks under the matching Poisson Dirichlet law are the same, and the constants differ by a factor between about 0.89 and 1. New in this book.
- [The shrinkage count and the species count fluctuate differently](../RESULTS.md#the_two_constructions_fluctuate_differently) (Proposition 5.27, page 186; proved): The effective model size concentrates around its mean while the number of blocks under the Pitman Yor partition has a nondegenerate random limit. New in this book.
<!-- results:end -->

## Key terms

many parameter models, overparameterization, double descent, optimism of a linear smoother, horseshoe prior, shrinkage weight, stick breaking, GEM distribution, Pitman Yor process, Poisson Dirichlet, Hoppe urn, Poisson Kingman partition, Fisher log series, allele frequency spectrum, effective model size, Mittag Leffler
