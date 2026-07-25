from ._activations import get_activations


DEFAULT_FIXED_HP = {
    # SCMPrior
    "mix_probs": (0.7, 0.3),
    # TreeSCM
    "tree_model": "random_forest",
    "tree_depth_lambda": 0.5,
    "tree_n_estimators_lambda": 0.5,
    "max_categories": 50,
    "corrupt_U": True,
    "enforce_categorical_completeness": True,
    "proxy_cat_cover_prob": 0.5,
    "propensity_epsilon": 1e-3,
    "prior_mc_mem_budget": 100_000_000,
    "enforce_completeness_heuristic": True,
    "completeness_max_attempts": 50,
    "dim_U_distribution": "triangular",
    "confounding_dim_control": True,
    "proxy_edge_subsampling": True,
    "covariate_edge_subsampling": True,
    "noise_std": 1e-3,
}

DEFAULT_SAMPLED_HP = {
    # MLPSCM
    "mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
    "block_wise_dropout": {
        "distribution": "meta_choice",
        "choice_values": [True, False],
    },
    "mlp_dropout_prob": {
        "distribution": "meta_beta",
        "scale": 0.9,
        "min": 0.1,
        "max": 5.0,
    },
    # MLPSCM and TreeSCM
    "num_layers": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 6,
        "min_mean": 1,
        "round": True,
        "lower_bound": 2,
    },
    "hidden_dim": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 130,
        "min_mean": 5,
        "round": True,
        "lower_bound": 4,
    },
    "init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "sampling": {
        "distribution": "meta_choice",
        "choice_values": ["normal", "mixed", "uniform"],
    },
    "pre_sample_cause_stats": {
        "distribution": "meta_choice",
        "choice_values": [True, False],
    },
    # CATEGORIZATION
    "cat_col_prob": {
        "distribution": "beta",
        "a": 1.0,
        "b": 3.0,
    },
    # PROPENSITY SHIFT AND SCALE
    "propensity_shift": {
        "distribution": "uniform",
        "min": -2.5,
        "max": 2.5,
    },
    "propensity_scale": {
        "distribution": "meta_trunc_norm_log_scaled",
        "min_mean": 0.5,
        "max_mean": 3.0,
        "lower_bound": 0.1,
        "round": False,
    },
    # CONFOUNDING EDGE SHARES
    "confounding_share_of_U_in_Y": {
        "distribution": "beta",
        "a": 1.2,
        "b": 2.8,
    },
    "confounding_share_of_U_in_A": {
        "distribution": "beta",
        "a": 1.2,
        "b": 2.8,
    },
    # PROXY EDGE SHARES
    "direct_edge_share_of_W_in_Y": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
    "direct_edge_share_of_Z_in_A": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
    # COV SHARE OF X IN U/W/Z/A/Y
    "covariate_share_of_X_in_U": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
    "covariate_share_of_X_in_W": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
    "covariate_share_of_X_in_Z": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
    "covariate_share_of_X_in_A": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
    "covariate_share_of_X_in_Y": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
    # NOISE
    "noise_share_of_input_in_U": {
        "distribution": "beta",
        "a": 1.2,
        "b": 2.8,
    },
    "noise_share_of_input_in_W": {
        "distribution": "beta",
        "a": 1.2,
        "b": 2.8,
    },
    "noise_share_of_input_in_Z": {
        "distribution": "beta",
        "a": 1.2,
        "b": 2.8,
    },
    "noise_share_of_input_in_A": {
        "distribution": "beta",
        "a": 1.2,
        "b": 2.8,
    },
    "noise_share_of_input_in_Y": {
        "distribution": "beta",
        "a": 1.2,
        "b": 2.8,
    },
}
