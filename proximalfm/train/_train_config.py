"""Define argument parser for Proximal TabICL training."""

import argparse

from proximalfm.prior._prior_config import DEFAULT_FIXED_HP


def str2bool(value):
    return value.lower() == "true"


def nullable_int(value):
    if value is None or str(value).lower() == "none":
        return None
    return int(value)


def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def build_parser():
    """Build an argument parser with all Proximal TabICL training arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all training, model, and
        checkpoint arguments.
    """
    parser = argparse.ArgumentParser()

    ###########################################################################
    ###### Wandb Config #######################################################
    ###########################################################################
    parser.add_argument(
        "--wandb_log", default=False, type=str2bool, help="Log results using wandb"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="ProximalFM", help="Wandb project name"
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID")
    parser.add_argument(
        "--wandb_dir", type=str, default=None, help="Wandb logging directory"
    )
    parser.add_argument(
        "--wandb_mode",
        default="online",
        type=str,
        help="Wandb logging mode: online, offline, or disabled",
    )

    ###########################################################################
    ###### Training Config ####################################################
    ###########################################################################
    parser.add_argument(
        "--device",
        default="mps",
        type=str,
        help="Device for training: cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="AMP dtype. bfloat16 recommended for H100 (no GradScaler needed).",
    )
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument(
        "--torch_seed", type=int, default=42, help="Random seed for torch"
    )
    parser.add_argument("--max_steps", type=int, default=60000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help="Size of micro-batches for gradient accumulation",
    )
    parser.add_argument(
        "--train_dataloader_num_workers",
        type=int,
        default=4,
        help="Number of worker processes for training data loading",
    )
    parser.add_argument(
        "--train_dataloader_prefetch_factor",
        type=nullable_int,
        default=None,
        help="Number of batches to prefetch (must be None if num_workers=0)",
    )
    parser.add_argument(
        "--gen_block_size",
        type=int,
        default=2000,
        help="Number of prior datasets to generate up front into a resident "
        "in-memory block (= ceil(gen_block_size / batch_size) generated batches). "
        "Generation cost is amortized over gen_reuse_passes passes of this block. "
        "To recover near-infinite-stream behaviour (regenerate every step), set "
        "--gen_block_size <= batch_size together with --gen_reuse_passes 1.",
    )
    parser.add_argument(
        "--gen_reuse_passes",
        type=int,
        default=5,
        help="Number of training passes over each generated block before "
        "regenerating a fresh one. This is the data reuse factor K.",
    )

    # Optimization Config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_warmup",
        help="Learning rate scheduler: see optim.py for options.",
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.2,
        help="The proportion of total steps over which we warmup."
        "If this value is set to -1, we warmup for a fixed number of steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="The number of steps over which we warm up. Only used when warmup_proportion is set to -1",
    )
    parser.add_argument(
        "--gradient_clipping", type=float, default=1.0, help="If > 0, clip gradients."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay / L2 regularization penalty",
    )
    parser.add_argument(
        "--cosine_num_cycles",
        type=int,
        default=1,
        help="Number of hard restarts for cosine schedule. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_amplitude_decay",
        type=float,
        default=1.0,
        help="Amplitude scaling factor per cycle. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_lr_end",
        type=float,
        default=0,
        help="Final learning rate for cosine_with_restarts",
    )
    parser.add_argument(
        "--poly_decay_lr_end",
        type=float,
        default=1e-7,
        help="Final learning rate for polynomial decay scheduler",
    )
    parser.add_argument(
        "--poly_decay_power",
        type=float,
        default=1.0,
        help="Power factor for polynomial decay scheduler",
    )

    ###########################################################################
    ###### Prior Dataset Config ###############################################
    ###########################################################################
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="If set, load pre-generated prior datasets directly from this directory on disk instead of generating them on the fly.",
    )
    parser.add_argument(
        "--load_prior_start",
        type=int,
        default=0,
        help="Batch index to start loading from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--delete_after_load",
        default=False,
        type=str2bool,
        help="Delete prior data after loading. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--batch_size_per_gp", type=int, default=4, help="Batch size per group"
    )

    # Proximal Causal Dimensions
    parser.add_argument(
        "--min_dim_U",
        type=int,
        default=3,
        help="Minimum dimensionality of latent confounder U",
    )
    parser.add_argument(
        "--max_dim_U",
        type=int,
        default=10,
        help="Maximum dimensionality of latent confounder U",
    )
    parser.add_argument(
        "--min_dim_W",
        type=int,
        default=3,
        help="Minimum dimensionality of outcome proxy W",
    )
    parser.add_argument(
        "--max_dim_W",
        type=int,
        default=10,
        help="Maximum dimensionality of outcome proxy W",
    )
    parser.add_argument(
        "--min_dim_Z",
        type=int,
        default=3,
        help="Minimum dimensionality of treatment proxy Z",
    )
    parser.add_argument(
        "--max_dim_Z",
        type=int,
        default=10,
        help="Maximum dimensionality of treatment proxy Z",
    )
    parser.add_argument(
        "--min_dim_X",
        type=int,
        default=5,
        help="Minimum dimensionality of baseline covariates X",
    )
    parser.add_argument(
        "--max_dim_X",
        type=int,
        default=20,
        help="Maximum dimensionality of baseline covariates X",
    )

    # Sequence Length and Split
    parser.add_argument(
        "--min_seq_len", type=int, default=None, help="Minimum samples per dataset"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024, help="Maximum samples per dataset"
    )
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size",
        type=train_size_type,
        default=0.1,
        help="Starting position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--max_train_size",
        type=train_size_type,
        default=0.9,
        help="Ending position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type",
        default="mlp_scm",
        type=str,
        help="Prior type: dummy, mlp_scm, tree_scm, mix_scm",
    )
    parser.add_argument(
        "--prior_device",
        default="cpu",
        type=str,
        help="Device for prior data generation",
    )
    parser.add_argument(
        "--mc_cate",
        type=int,
        default=200,
        help="Monte Carlo latent/noise worlds per X used to estimate the CATE "
        "target MU0/MU1. Higher => lower target variance (cleaner labels), "
        "linearly more generation cost. Only used by the on-the-fly SCM prior.",
    )
    parser.add_argument(
        "--mc_targets_on_context",
        type=str2bool,
        default=False,
        help="If True, estimate MU0/MU1 for every row. If False (the default), "
        "only the query rows [train_size:] are Monte-Carlo replicated and "
        "MU0/MU1 are NaN on the context prefix. Training only ever reads the "
        "query slice, so False is the recommended setting for the on-the-fly "
        "SCM prior: it roughly halves generation cost at the default "
        "train_size range.",
    )
    parser.add_argument(
        "--prior_mc_mem_budget",
        type=int,
        default=DEFAULT_FIXED_HP["prior_mc_mem_budget"],
        help="Element-count cap on the (seq_len*mc_cate)-row prior generation "
        "tensors; auto-lowers effective mc_cate per dataset to stay under it. "
        "Raise on large-memory GPUs so mc_cate is not throttled; set 0 to disable.",
    )

    ###########################################################################
    ##### Model Architecture Config ###########################################
    ###########################################################################
    parser.add_argument(
        "--amp",
        default=True,
        type=str2bool,
        help="If True, use automatic mixed precision (AMP) which can provide significant speedups on compatible GPU",
    )
    parser.add_argument(
        "--model_compile",
        default=False,
        type=str2bool,
        help="If True, compile the model using torch.compile for speedup",
    )

    # Shared Architecture Config
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Base embedding dimension"
    )
    parser.add_argument(
        "--ff_factor",
        type=int,
        default=2,
        help="Expansion factor for feedforward dimensions",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability"
    )
    parser.add_argument(
        "--activation", type=str, default="gelu", help="Activation function type"
    )
    parser.add_argument(
        "--norm_first",
        default=True,
        type=str2bool,
        help="If True, use pre-norm transformer architecture",
    )
    parser.add_argument(
        "--bias_free_ln",
        default=False,
        type=str2bool,
        help="If True, removes bias from all LayerNorm layers",
    )
    parser.add_argument(
        "--recompute",
        default=False,
        type=str2bool,
        help="If True, uses gradient checkpointing to save memory",
    )

    # Column Embedding Config
    parser.add_argument(
        "--col_num_blocks",
        type=int,
        default=3,
        help="Number of blocks in column embedder",
    )
    parser.add_argument(
        "--col_nhead",
        type=int,
        default=4,
        help="Number of attention heads in column embedder",
    )
    parser.add_argument(
        "--col_num_inds",
        type=int,
        default=128,
        help="Number of inducing points in column embedder",
    )
    parser.add_argument(
        "--col_affine",
        default=False,
        type=str2bool,
        help="If True, computes column embeddings as an affine transformation",
    )
    parser.add_argument(
        "--col_ssmax",
        type=str,
        default="qassmax-mlp-elementwise",
        help="Type of scalable softmax to use in the column embedding transformer",
    )

    # Row Interaction Config
    parser.add_argument(
        "--row_num_blocks",
        type=int,
        default=3,
        help="Number of blocks in row interactor",
    )
    parser.add_argument(
        "--row_nhead",
        type=int,
        default=8,
        help="Number of attention heads in row interactor",
    )
    parser.add_argument(
        "--row_num_cls",
        type=int,
        default=4,
        help="Number of CLS tokens in row interactor",
    )
    parser.add_argument(
        "--row_rope_base",
        type=float,
        default=100000,
        help="RoPE base value for row interactor",
    )
    parser.add_argument(
        "--row_rope_interleaved",
        default=False,
        type=str2bool,
        help="If True, uses interleaved rotation for RoPE",
    )

    # ICL Config
    parser.add_argument(
        "--icl_num_blocks",
        type=int,
        default=12,
        help="Number of transformer blocks in ICL predictor",
    )
    parser.add_argument(
        "--icl_nhead",
        type=int,
        default=4,
        help="Number of attention heads in ICL predictor",
    )
    parser.add_argument(
        "--icl_ssmax",
        type=str,
        default="qassmax-mlp-elementwise",
        help="Type of scalable softmax to use in the in-context learning transformer",
    )

    ###########################################################################
    ###### Checkpointing ######################################################
    ###########################################################################
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        help="Directory for checkpoint saving and loading",
    )
    parser.add_argument(
        "--save_temp_every",
        default=50,
        type=int,
        help="Steps between temporary checkpoints",
    )
    parser.add_argument(
        "--save_perm_every",
        default=5000,
        type=int,
        help="Steps between permanent checkpoints",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=3,
        help="Maximum number of temporary checkpoints to keep. Permanent checkpoints are not counted.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Path to specific checkpoint file to load",
    )
    parser.add_argument(
        "--transfer_rules",
        default="transfer_rules.yaml",
        type=str,
        help="Path to the yaml of per-parameter transfer/freeze rules "
        "(finetune / freeze / scratch, longest matching prefix wins). Resolved "
        "relative to the CWD, then to the repository root.",
    )
    parser.add_argument(
        "--save_best",
        default=True,
        type=str2bool,
        help="Save a 'best_model.ckpt' (overwriting the previous one) whenever the "
        "monitored validation metric improves. Runs in addition to the regular "
        "step-based checkpoints.",
    )
    parser.add_argument(
        "--best_metric",
        default="val/point/cate/npehe",
        type=str,
        help="Validation metric key to monitor for best-model checkpointing "
        "(e.g. 'val/point/cate/rmse', 'val/point/ate/rmse', 'val/point/cate/r2'). "
        "See CausalEvaluator for available keys.",
    )
    parser.add_argument(
        "--best_metric_mode",
        default="min",
        choices=["min", "max"],
        help="Whether a lower ('min', e.g. an RMSE) or higher ('max', e.g. an R2) "
        "value of --best_metric is considered better.",
    )
    parser.add_argument(
        "--only_load_model",
        default=True,
        type=str2bool,
        help="Whether to only load model weights",
    )

    ###########################################################################
    ###### Validation #########################################################
    ###########################################################################
    parser.add_argument(
        "--val_dir",
        default=None,
        type=str,
        help="Directory for validation data.",
    )
    parser.add_argument(
        "--point_val_interval",
        default=50,
        type=int,
        help="Steps between validation checks",
    )

    parser.add_argument(
        "--val_micro_batch_size",
        default=None,
        type=nullable_int,
        help="Micro-batch size used during validation. Validation runs under "
        "inference_mode (no backward pass), so it can use a larger micro-batch "
        "than training before hitting memory limits. Defaults to "
        "--micro_batch_size when unset.",
    )

    ###########################################################################
    ###### Distributional (prior) evaluation ##################################
    ###########################################################################
    parser.add_argument(
        "--dist_eval_dir",
        default=None,
        type=str,
        help="Root directory of the frozen posterior-eval set (holds both the "
        "val/-format episodes and frozen_meta.pt; see "
        "posterior_eval/build_prior_frozen_set.py). None disables the prior/* "
        "distributional evaluation surface.",
    )
    parser.add_argument(
        "--dist_eval_interval",
        default=-1,
        type=int,
        help="Steps between distributional evaluation passes against "
        "--dist_eval_dir. 0 disables it; -1 (default) resolves to "
        "--point_val_interval.",
    )

    ###########################################################################
    ###### LOSS ###############################################################
    ###########################################################################
    parser.add_argument(
        "--lambda_cate",
        type=float,
        default=0.0,
        help="Weight for the CATE loss term. If 0, CATE loss is not used. "
        "Set it explicitly for --loss_type cate_single / cate_single_noise_aware: "
        "the MSE term protects the predicted mean and gives the from-scratch head "
        "an early learning signal. It needs no adjustment under the noise-aware "
        "objective, since convolving the prediction leaves its mean unchanged.",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="bins_po",
        choices=[
            "bins_po",
            "hl",
            "cate_decomposition",
            "cate_single",
            "cate_single_noise_aware",
        ],
        help="Training objective. 'cate_single_noise_aware' uses the same head as "
        "'cate_single' (checkpoints are interchangeable) but treats the target as a "
        "Monte Carlo estimate of tau and deconvolves its sampling noise, so the head "
        "converges to a posterior over tau rather than to that posterior blurred by "
        "MC error. It reads the per-row noise variance the generator emits, so "
        "pre-generated data written before that existed trains as if noise-free "
        "(watch noise/frac_fallback).",
    )

    parser.add_argument(
        "--nbins",
        type=int,
        default=1000,
        help="Number of bins in the histogram head(s).",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=-10.0,
        help="Lower edge of the potential-outcome bin grid (standardized scale).",
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=10.0,
        help="Upper edge of the potential-outcome bin grid (standardized scale).",
    )

    parser.add_argument(
        "--cate_vmin",
        type=float,
        default=-3.0,
        help="Lower edge of the tau bin grid used by --loss_type cate_single and "
        "cate_single_noise_aware (standardized scale). tau is much narrower than "
        "Y(a), hence its own grid.",
    )

    parser.add_argument(
        "--cate_vmax",
        type=float,
        default=3.0,
        help="Upper edge of the tau bin grid used by --loss_type cate_single and "
        "cate_single_noise_aware.",
    )

    parser.add_argument(
        "--hl_start_sigma",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--hl_warmup_sigma_steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--hl_linear_decay_steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--hl_final_sigma",
        type=float,
        default=0.01,
    )

    return parser
