# Chapter 7. Ascertainment and the Two Parameter Poisson Dirichlet Family

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 7. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509613. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

Sampling a Poisson Dirichlet subdivision of the unit interval with a nonuniform density is a weighted distribution problem for Kingman's paintbox. In the one parameter case it has an exact answer: a Beta(1, alpha) weight carries the Ewens sampling formula with parameter theta to the one with parameter theta over alpha. This chapter asks what survives in the two parameter family. The reduction survives. Exactness does not, and the obstruction is a rigidity theorem: the residual allocation structure is preserved only by power weights, and among power weights the Ewens family alone is preserved, so no sampling density that charges every interval carries the size biased sticks of PD(sigma, theta) with sigma positive to another member. No Kingman type genealogy in which the mutation probability depends only on the branch count can give the family either. The chapter classifies the growth of the number of types under a general weight by its behaviour at the origin, shows that the abundance spectrum degenerates, and proves that the singleton fraction estimates the exponent with an explicit bias.

## From the literature

Kingman's paintbox is Kingman (1978). The Ewens Pitman family and its sampling formula are due to Pitman and to Pitman and Yor (1997). The one parameter reduction by a Beta weight is as stated in the chapter. Karlin's occupancy correspondence is used as cited there.

## New results

<!-- results:begin -->
- [Closure of the Ewens Pitman family under weighted sampling](../RESULTS.md#py_thm_pitman) (Theorem 7.9, page 269; proved): Sampling a two parameter Poisson Dirichlet population in the size biased layout with a power density keeps the Ewens Pitman family closed only in the Ewens case or when the density is uniform. New in this book.
- [Rigidity of the residual allocation structure](../RESULTS.md#py_thm_rigid) (Theorem 7.11, page 271; proved): Among sampling densities that charge every interval, only power weights keep the induced subdivision of the size biased layout a residual allocation model, so for a discount strictly between zero and one no such density other than the uniform one carries a two parameter Poisson Dirichlet law to the stick breaking sequence of another member of the family. New in this book.
- [No Kingman type genealogy produces the two parameter family](../RESULTS.md#py_prop_nocoal) (Proposition 7.12, page 273; proved): A genealogy in which the mutation probability depends only on the number of branches gives a count of types with variance at most its mean, which rules out positive discount. New in this book.
- [Exponent map of the number of types under power sampling](../RESULTS.md#exponent_map_under_power_sampling) (Proposition 7.13, page 275; proved): Sampling a Poisson Dirichlet population in the size biased layout with a power density gives an exact power law envelope and moves the growth exponent of the number of types from sigma to sigma star. New in this book.
- [The sigma diversity is a power of the stick envelope](../RESULTS.md#sigma_diversity_is_a_power_of_the_stick_envelope) (Lemma 7.16, page 279; proved): The almost sure limit of the number of types over n to the sigma equals the sigma th power of the stick envelope constant over sigma, with no constant left over. New in this book.
- [The weighted diversity is a power of the unweighted diversity](../RESULTS.md#diversity_is_a_power_of_the_unweighted_diversity) (Proposition 7.17, page 280; proved): Under power sampling the limit of the number of types over n to the sigma star is a constant times a power of the unweighted sigma diversity. New in this book.
- [The constant in the diversity power law](../RESULTS.md#constant_in_the_diversity_power_law) (Proposition 7.18, page 280; proved): The constant relating the weighted and unweighted diversities has a closed form in gamma functions, does not depend on theta, and equals one for uniform sampling. New in this book.
- [Bias of the singleton fraction](../RESULTS.md#py_prop_rate) (Proposition 7.27, page 288; proved): Under an exact power law envelope with exponent sigma star, the expected singleton fraction exceeds sigma star by a relative amount of one over twice the expected number of types. New in this book.
- [Growth regimes of the number of types under weighted sampling](../RESULTS.md#py_thm_regimes) (Theorem 7.33, page 296; proved): For a discount strictly between zero and one, the growth of the number of types under a general sampling weight is classified by the behaviour of the weight at the origin, from bounded through a power of log n and a power of n to n divided by a slowly varying function. New in this book.
<!-- results:end -->

## Key terms

ascertainment, Poisson Dirichlet distribution, Ewens Pitman family, Pitman Yor process, weighted sampling, paintbox, residual allocation model, stick breaking, rigidity theorem, power weights, regular variation, number of types, singleton fraction, abundance spectrum
