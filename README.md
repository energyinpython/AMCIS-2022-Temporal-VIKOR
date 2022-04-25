# ISD-2022-Temporal-VIKOR

The methodological framework for Temporal VIKOR - New MCDA Method Supporting Sustainability Assessment

This Python project provides a methodological framework for Temporal VIKOR. This framework is dedicated to temporal sustainability assessment.

## Methods

- The Multi-Criteria Decision Analysis (MCDA) VIKOR method `vikor` for multi-criteria assessment of alternatives
- Objective criteria weighting methods:

	- the Equal weighting method `equal_weighting`
	- the Entropy weighting method `entropy_weighting`
	- the Standard Deviation weighting method `std_weighting`
	- the CRITIC weighting method `critic_weighting`
	
- Normalization methods for decision matrix normalization

	- `linear_normalization`
	- `minmax_normalization`
	- `max_normalization`
	- `sum_normalization`
	- `vector_normalization`
	
- `rank_preferences` function for generating rankings based on MCDA utility function values

- Correlation coefficients for determining correlation between rankings

	- `spearman`
	- `weighted_spearman`
	- `coeff_WS`
	- `pearson_coeff`
	
`daria` class with methods for temporal MCDA assessment, including:

- Measures for data variability calculation:

	- `gini`
	- `entropy`
	- `std`
	- `stat_var`
	
- Method for determining variability direction `direction`

- Method for updating overall aggregated temporal efficiency considering results variability `update_efficiency`

- `main_daria.py` file including usage example of provided framework.

## License

This project is licensed under the terms of the MIT license.
