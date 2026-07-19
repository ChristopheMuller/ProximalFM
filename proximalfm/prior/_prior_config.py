from ._activations import get_activations


DEFAULT_FIXED_HP = {
    # SCMPrior
    "mix_probs": (0.7, 0.3),
    # TreeSCM
    "tree_model": "random_forest",
    "tree_depth_lambda": 0.5,
    "tree_n_estimators_lambda": 0.5,
    "cat_prob": 0.2,
    "max_categories": 50,
    "corrupt_U": True,
    "enforce_categorical_completeness": True,
    "proxy_cat_cover_prob": 0.5,
    "propensity_epsilon": 1e-3,
    "prior_mc_mem_budget": 100_000_000,
    "enforce_completeness_heuristic": True,
    "completeness_max_attempts": 50,
    "dim_U_distribution": "triangular",
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
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 0.0001,
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
    "pre_sample_noise_std": {
        "distribution": "meta_choice",
        "choice_values": [True, False],
    },
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
    "confounding_duplication_of_U_in_Y": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
    "confounding_duplication_of_U_in_A": {
        "distribution": "beta",
        "a": 1.0,
        "b": 1.0,
    },
}
