# Chapter 9. Rational Conjugacy: Pochhammer Priors and Gamma Ratios

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 9. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509850. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

Pochhammer priors, introduced by Wang and Polson (2024), give conjugate inference for concentration parameters that enter a likelihood through ratios of Gamma functions with integer shifts. This chapter develops the theory around one principle, rational conjugacy: the Pochhammer family is not closed under updating, but proper rational densities on the positive half line are, and on that class every normalizing constant, moment and predictive probability is a finite sum of residues. It proves a residue formula with poles of any multiplicity, a closed form Mellin transform, an exact tail index, and a corrected reading of a Kullback Leibler risk bound: every member attains the rate N to the minus one times log N, and the half horseshoe only gives the smaller constant. A completely monotone representation identifies the half horseshoe on structural grounds and gives an exact latent variable sampler for Negative Binomial dispersion. An exponentially tilted family has a normalizer that is a finite sum of exponential integrals. The chapter locates the boundary of the method, gives a Zipf tail theorem, and quantifies the loss of digits in the residue expansion. Running examples are the Ewens concentration parameter and Corbet's butterflies.

## From the literature

Pochhammer priors are Wang and Polson (2024). The Cesaro risk bound for horseshoe type priors is Polson and Scott (2011). The reciprocal gamma mixtures are He, Polson and Xu (2019). Damsleth (1975) gave the conjugate class for the two parameter Gamma. The chapter states which of these results it corrects.

## New results

<!-- results:begin -->
- [Residue formula for the normalizing constant of a rational density](../RESULTS.md#normalizing_constant) (Theorem 9.3, page 346; proved): The integral of a rational function with poles of any multiplicity in the cut plane is a finite sum of residues times logarithms and powers. New in this book.
- [Mellin transform of a Pochhammer density](../RESULTS.md#mellin_transform) (Theorem 9.6, page 347; proved): The Mellin transform of a Pochhammer density has a closed form on its maximal strip, which gives fractional and negative moments and the expected log. New in this book.
- [Shrinkage rate of the half horseshoe, corrected](../RESULTS.md#shrinkage_rate_corrected) (Corollary 9.8, page 348; proved): Every Pochhammer prior attains the Kullback Leibler rate N^{-1} log N, and the half horseshoe only gives the smaller constant, which corrects the reading of the earlier bound. Corrects a statement of Wang and Polson (2024), Pochhammer priors for sparse count models, arXiv:2402.09583.
- [Exact tail index of the Pochhammer law](../RESULTS.md#exact_tail_index_of_the_pochhammer_law) (Corollary 9.7, page 348; proved): The Pochhammer density is regularly varying with an exact tail index given by its parameters, so it has power law tails and finitely many moments. New in this book.
- [Conditional exponentiality of the half horseshoe](../RESULTS.md#conditional_exponentiality) (Theorem 9.9, page 348; proved): A Pochhammer density is completely monotone exactly when m and d are zero, in which case it is a scale mixture of exponentials, which yields an exact latent variable sampler. New in this book.
- [Tilted normalizer with exponential integrals](../RESULTS.md#tilted_normalizer) (Theorem 9.16, page 353; proved): Exponentially tilting a rational density gives a normalizer that is a finite sum of exponential integrals, valid for every degree of the numerator. New in this book.
- [Exact dispersion block for Negative Binomial regression](../RESULTS.md#exact_dispersion_block) (Proposition 9.21, page 358; proved): The dispersion parameter of Negative Binomial regression with a Pochhammer prior has an exact conditional law, so a partially collapsed Gibbs sampler needs no Metropolis step. New in this book.
- [Zipf tails of the Negative Binomial Beta Pochhammer hierarchy](../RESULTS.md#zipf_tails) (Theorem 9.20, page 358; proved): The marginal law of the counts in the hierarchy has an exact power law tail with a logarithmic correction, so a Zipf exponent can be set as a prior hyperparameter. New in this book.
- [Boundary Zipf law of the Yule Simon prior predictive](../RESULTS.md#boundary_zipf_law) (Proposition 9.22, page 363; proved): Under a Pochhammer prior on the Zipf exponent the Yule Simon prior predictive is proper but has no positive moments. New in this book.
<!-- results:end -->

## Key terms

Pochhammer prior, rational conjugacy, residue calculus, Mellin transform, half horseshoe, completely monotone, concentration parameter, Negative Binomial regression, dispersion, Zipf law, Yule Simon distribution, exponential integral, Dirichlet process, Pitman Yor, Kullback Leibler risk, Corbet butterflies
