import argparse


def _add_optional_bool(parser, name, default=None, help_text=None):
    parser.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no_{name}", dest=name, action="store_false")


def _finalize_target_enrichment_args(args):
    if args.target_enrichment:
        if args.freeze_host is None:
            args.freeze_host = True
        if args.use_host_loss is None:
            args.use_host_loss = False
        if args.use_target_retrieval_loss is None:
            args.use_target_retrieval_loss = True
        if args.use_target_robust_loss is None:
            args.use_target_robust_loss = True
    else:
        if args.freeze_host is None:
            args.freeze_host = False
        if args.use_host_loss is None:
            args.use_host_loss = True
        if args.use_target_retrieval_loss is None:
            args.use_target_retrieval_loss = False
        if args.use_target_robust_loss is None:
            args.use_target_robust_loss = False

    if args.enrichment_space not in ("global", "default"):
        raise ValueError("IRRA target enrichment supports only global/default enrichment_space")
    if args.context_module != "mixer":
        raise ValueError("target enrichment requires context_module=mixer")
    if args.residual_gate == "static" and args.enrich_gamma is None:
        raise ValueError("residual_gate=static requires --enrich_gamma")
    if args.residual_gate == "residual" and args.enrich_gamma is not None:
        raise ValueError("residual_gate=residual forbids --enrich_gamma")
    if args.top_m <= 0:
        raise ValueError("top_m must be positive")
    if args.num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if args.tau <= 0:
        raise ValueError("tau must be positive")
    if args.pnp_text_only:
        if not args.target_enrichment:
            raise ValueError("pnp_text_only requires target_enrichment")
        if not args.freeze_host or not args.use_freeze_indices:
            raise ValueError("pnp_text_only requires freeze_host and use_freeze_indices")
        if args.use_host_loss:
            raise ValueError("pnp_text_only requires host loss to be disabled")
    if args.use_freeze_indices and not args.freeze_host:
        raise ValueError("use_freeze_indices requires freeze_host")
    return args


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id+mlm', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    ######################## target enrichment ########################
    parser.add_argument("--target_enrichment", default=False, action="store_true")
    parser.add_argument("--enrichment_start", type=int, default=1)
    parser.add_argument("--enrichment_space", type=str, default="global")
    _add_optional_bool(parser, "freeze_host", default=None)
    _add_optional_bool(parser, "use_host_loss", default=None)
    parser.add_argument("--lambda_host", type=float, default=1.0)
    parser.add_argument("--host_ckpt_file", default="", help="optional host checkpoint for frozen plug-in training")

    parser.add_argument("--use_shared_k", default=False, action="store_true")
    parser.add_argument("--pool_k_mode", type=str, default="static", choices=["static", "adaptive"])
    parser.add_argument("--pool_k", type=int, default=0)
    parser.add_argument("--pool_k_candidates", type=str, default="256,512,1024,2048,4096,8192")
    parser.add_argument("--pool_clusters", type=int, default=32)
    parser.add_argument("--positive_ratio_max", type=float, default=0.5)
    parser.add_argument("--pool_dist_metric", type=str, default="l1", choices=["l1", "js"])
    parser.add_argument("--pool_dist_threshold", type=float, default=0.25)
    parser.add_argument("--pool_coverage_epochs", type=int, default=1)
    parser.add_argument("--recompute_level", type=str, default="epoch", choices=["epoch", "step"])
    parser.add_argument("--recompute_interval", type=int, default=-1)
    parser.add_argument("--use_freeze_indices", default=False, action="store_true")
    parser.add_argument("--pnp_text_only", default=False, action="store_true")
    parser.add_argument("--pool_seed", type=int, default=1)

    parser.add_argument("--top_m", type=int, default=8)
    parser.add_argument("--extractor_mode", type=str, default="global,horizontal")
    parser.add_argument("--num_parts", type=int, default=6)
    parser.add_argument("--robust_hard_k", type=int, default=16)

    parser.add_argument("--context_module", type=str, default="mixer")
    parser.add_argument("--mixer_dim", type=int, default=256)
    parser.add_argument("--mixer_depth", type=int, default=2)
    parser.add_argument("--mixer_hidden_part", type=int, default=128)
    parser.add_argument("--mixer_hidden_rank", type=int, default=128)
    parser.add_argument("--mixer_hidden_channel", type=int, default=512)
    parser.add_argument("--mixer_hidden_readout", type=int, default=512)
    parser.add_argument("--context_pooling", type=str, default="mlp", choices=["mlp", "late_attention", "hybrid_attention"])

    parser.add_argument("--residual_gate", type=str, default="residual", choices=["residual", "static"])
    parser.add_argument("--enrich_gamma", type=float, default=None)
    parser.add_argument("--residual_gate_hidden_dim", type=int, default=256)

    _add_optional_bool(parser, "use_target_retrieval_loss", default=None)
    _add_optional_bool(parser, "use_target_robust_loss", default=None)
    parser.add_argument("--lambda_ret", type=float, default=1.0)
    parser.add_argument("--lambda_rob", type=float, default=1.0)
    parser.add_argument("--lambda_gain", type=float, default=1.0)
    parser.add_argument("--gain_margin", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.02)

    parser.add_argument("--eval_target_enrichment", default=False, action="store_true")
    parser.add_argument("--eval_fusion_ablation", default=False, action="store_true")
    parser.add_argument("--lambda_proto", type=float, default=0.5)
    parser.add_argument("--fusion_lambdas", type=str, default="0.0,0.25,0.5,0.75,1.0")

    args = parser.parse_args()

    return _finalize_target_enrichment_args(args)
