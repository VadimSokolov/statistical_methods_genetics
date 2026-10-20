# Chapter 3. The Fundamental Theorem of Natural Selection: From Deterministic to Stochastic, Fisher to Bayes

<!-- cite:begin -->
From *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme* by Nicholas G. Polson (2026), chapter 3. Cite as Polson (2026), *On Statistical Methods for Genetics: Fisher, Ewens and the Bayesian Programme*, edition 1, https://doi.org/10.5281/zenodo.23509850. Book page: https://vsokolov.org/books/statistical_methods_genetics/. Source: https://github.com/VadimSokolov/statistical_methods_genetics.
<!-- cite:end -->

Fisher's Fundamental Theorem of Natural Selection says that the additive genetic variance is the rate of change of mean fitness. As a statement about the total change it is false, as Kimura showed in 1958 and as Ewens (2024) sets out. This chapter asks what the falsehood costs and what replaces it. The deterministic half rests on a lemma: the average effects are the coefficients of an orthogonal projection of fitness onto a fixed subspace taken in a moving inner product. The result is an exact decomposition of the rate of change of mean fitness into Fisher's additive term and a remainder, valid for arbitrary frequency paths and for the whole genome. The remainder vanishes for every fitness assignment exactly when the log frequencies move inside the span of the gene contents. The chapter gives the remainder in closed form at one biallelic locus and at two loci, an exact remainder and relative error for one locus with k alleles, and a factorization when the distribution moves in an exponential family. The stochastic half asks what data can say about Fisher's claim, using Bayes factors and global local shrinkage.

## From the literature

Fisher (1930) stated the theorem. Price (1972) and Ewens (1989) gave the reading in terms of the partial change in mean fitness, and Kimura (1958) gave the counterexample to the total change reading. Ewens (2024) is the account the chapter builds on. Lindley (1988) supplies the coordinates used for the biallelic remainder.

## New results

<!-- results:begin -->
- [Moving projection lemma](../RESULTS.md#moving_projection_lemma) (Lemma 3.2, page 63; proved): For arbitrary smooth genotype frequency paths the rate of change of mean Malthusian fitness is Fisher's additive term plus a remainder that pairs the least squares deviations with the logarithmic velocity of the genotype frequencies. New in this book.
- [Whole genome Fundamental Theorem decomposition](../RESULTS.md#whole_genome) (Theorem 3.5, page 66; proved): For arbitrary genotype frequency paths, the rate of change of mean Malthusian fitness splits exactly into an additive part and a remainder; the remainder vanishes whenever the log frequencies move inside the span of the gene contents, and it vanishes for every assignment of fitnesses only then. New in this book.
- [Exponential family factorization of the remainder](../RESULTS.md#exponential_family_factorization) (Theorem 3.8, page 69; proved): When the genotype distribution moves inside an exponential family whose main effects are the gene contents, the main effects contribute nothing to the remainder, which depends on the interaction coordinates alone. New in this book.
- [Remainder in Lindley coordinates](../RESULTS.md#remainder_in_lindley_coordinates) (Corollary 3.10, page 70; proved): At a biallelic locus, in the coordinates of allele frequency and Hardy Weinberg disequilibrium, the remainder is the dominance contrast times the Fisher information times the velocity of the disequilibrium coefficient. New in this book.
- [The residual in closed form at a biallelic locus](../RESULTS.md#the_residual_in_closed_form) (Proposition 3.11, page 71; proved): At a biallelic locus the residual of the additive projection is one known vector times one number, which gives the dominance variance and the remainder in closed form. New in this book.
- [Two locus residual](../RESULTS.md#two_locus_residual) (Proposition 3.14, page 73; proved): For two biallelic loci the residual is an interaction contrast times the information about linkage disequilibrium, so recombination alone changes mean fitness at an exact rate. New in this book.
- [Exact remainder for one locus with k alleles](../RESULTS.md#exact_remainder_one_locus_alleles) (Theorem 3.16, page 76; proved): In the one locus random mating model with constant fitnesses, the change in mean fitness is the additive term divided by mean fitness plus an explicit quadratic remainder in the dominance deviations. New in this book.
- [Exact relative error of the Fundamental Theorem, two alleles](../RESULTS.md#exact_relative_error_two_alleles) (Corollary 3.17, page 76; proved): At a biallelic locus the relative size of the remainder against the additive term is exactly D pq divided by twice mean fitness, which is first order in the selection coefficient. New in this book.
<!-- results:end -->

## Key terms

Fundamental Theorem of Natural Selection, Fisher, additive genetic variance, average effects, weighted least squares projection, mean fitness, partial change, remainder, dominance variance, epistatic variance, linkage disequilibrium, quasi linkage equilibrium, Lindley coordinates, exponential family, Bayes factor, shrinkage
