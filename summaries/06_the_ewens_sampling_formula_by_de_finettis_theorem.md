# Chapter 6. The Ewens Sampling Formula by de Finetti's Theorem

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 6. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509613. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

This chapter derives the Ewens sampling formula from de Finetti's theorem in the elementary form of Heath and Sudderth, where the mixing measure is a weak limit of the laws of the empirical distribution function. Helly selection and the Helly Bray lemma give the representation, and the only inequality needed is that drawing k items with and without replacement from n differs by k(k-1)/(2n) in total variation. If the one step forecast depends on the past only through the empirical distribution and is affine in it, exchangeability forces the weights theta over theta plus n and n over theta plus n, and the directing measure is a Dirichlet process. The chapter gives an exact finite sample variance and a moment estimator of theta, the exchangeable partition probability function, the sufficiency of the number of alleles and its law, and moments of the abundance counts. It then weights the paintbox by abundance, shows that the de Finetti architecture is invariant, treats the two parameter Pitman family including its negative branch as a finite allele model, and applies the tools to Corbet's butterflies, with the fit corrected for the censoring of the collection.

## From the literature

De Finetti's theorem in the elementary form is Heath and Sudderth (1976). The Ewens sampling formula is Ewens (1972). Kingman's paintbox is Kingman (1978). Pitman's sampling formula is Pitman (1995). Corbet's butterfly data are from Fisher, Corbet and Williams (1943). The finite allele reading of the negative branch is classical.

## New results

<!-- results:begin -->
- [Exchangeability forces the Ewens predictive weights](../RESULTS.md#the_weights_are_forced) (Proposition 6.8, page 217; proved): If the one step forecast is affine in the empirical distribution and the sequence is exchangeable, the weights are theta over theta plus n and n over theta plus n. New in this book.
- [Exact finite n variance and a moment estimator of theta](../RESULTS.md#exact_finite_variance_and_a_moment_estimator) (Proposition 6.11, page 219; proved): The variance of an empirical average under the Ewens predictive rule is exact for every n and can be inverted to give theta without any asymptotic approximation. New in this book.
- [The de Finetti architecture is invariant under abundance weighting](../RESULTS.md#the_de_finetti_architecture_is_invariant) (Proposition 6.23, page 233; proved): Weighting the sampling probabilities by a function of abundance keeps the sample exchangeable and changes only the mixing measure. New in this book.
- [The negative branch is not a weighted paintbox](../RESULTS.md#the_negative_branch_is_not_a_weighted_paintbox) (Proposition 6.34, page 239; proved): A two parameter partition law with negative discount cannot be reproduced by weighting a one parameter Poisson Dirichlet paintbox. New in this book.
<!-- results:end -->

## Key terms

Ewens sampling formula, de Finetti theorem, Heath Sudderth, exchangeability, Helly selection, Dirichlet process, Chinese restaurant process, exchangeable partition probability function, number of alleles, sufficiency, Pitman sampling formula, Pitman Yor, finite allele model, weighted paintbox, species sampling, Corbet butterflies
