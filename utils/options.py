import argparse


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

    ######################## target-aware enrichment ########################
    parser.add_argument("--target_enrichment", default=False, action="store_true", help="enable target-aware text enrichment")
    parser.add_argument("--enrichment_start", type=int, default=1, help="first epoch that passes target cache into training")
    parser.add_argument("--enrichment_space", type=str, default="global", help="[global, retrieval, grab]")
    parser.add_argument("--top_m", type=int, default=32, help="number of target-pool images gathered per query")
    parser.add_argument("--topm_rank_space", type=str, default="host_global", help="[host_global, retrieval, hybrid_global_retrieval, hybrid_global_grab]")
    parser.add_argument("--topm_rank_lambda", type=float, default=0.5, help="global-score weight for hybrid top-M ranking")
    parser.add_argument("--lambda_ret", type=float, default=1.0, help="target-pool retrieval loss weight")
    parser.add_argument("--freeze_host", default=False, action="store_true", help="train only target_enricher parameters")
    parser.add_argument("--use_host_loss", dest="use_host_loss", default=True, action="store_true", help="include host losses when target enrichment is active")
    parser.add_argument("--no-use_host_loss", dest="use_host_loss", action="store_false", help="disable host losses when target enrichment is active")
    parser.add_argument("--lambda_host", type=float, default=1.0, help="host loss weight when target enrichment is active")
    parser.add_argument("--extractor_mode", type=str, default="global,horizontal", help="comma-separated target evidence providers")
    parser.add_argument("--num_parts", type=int, default=6, help="layout evidence parts")
    parser.add_argument("--target_relative_space", type=str, default="host_global", help="[host_global, retrieval]")
    parser.add_argument("--target_relative_num_clusters", type=int, default=16, help="target-relative k-means clusters")
    parser.add_argument("--target_relative_cluster_method", type=str, default="kmeans", help="target-relative clustering method")
    parser.add_argument("--evidence_token_budget", type=int, default=0, help="0 disables evidence slot budget check")
    parser.add_argument("--evidence_projection", type=str, default="auto", help="[auto, linear, none]")
    parser.add_argument("--recompute_level", type=str, default="epoch", help="[epoch, step]")
    parser.add_argument("--recompute_interval", type=int, default=1, help="-1 builds once; otherwise refresh every N units")
    parser.add_argument("--pool_interval", type=int, default=None, help="alias for recompute_interval")
    parser.add_argument("--use_freeze_indices", default=False, action="store_true", help="precompute top-M rows once and reuse them by query index")
    parser.add_argument("--freeze_indices", default=False, action="store_true", help="alias for use_freeze_indices")
    parser.add_argument("--pnp_text_only", default=False, action="store_true", help="frozen-host mode that encodes only batch text online")
    parser.add_argument("--target_cache_batch_size", type=int, default=512, help="target image cache encoding batch size")
    parser.add_argument("--target_query_batch_size", type=int, default=512, help="target frozen-rank/query encoding batch size")
    parser.add_argument("--context_module", type=str, default="mixer", help="target context module")
    parser.add_argument("--mixer_dim", type=int, default=256, help="rank-slot mixer width")
    parser.add_argument("--mixer_depth", type=int, default=2, help="rank-slot mixer depth")
    parser.add_argument("--mixer_hidden_part", type=int, default=32, help="slot-mixing hidden width")
    parser.add_argument("--mixer_hidden_rank", type=int, default=64, help="rank-mixing hidden width")
    parser.add_argument("--mixer_hidden_channel", type=int, default=512, help="channel-mixing hidden width")
    parser.add_argument("--mixer_hidden_readout", type=int, default=128, help="mixer readout hidden width")
    parser.add_argument("--context_pooling", type=str, default="mlp", help="target context pooling")
    parser.add_argument("--mixer_context_pooling", type=str, default="mlp", help="alias for context_pooling")
    parser.add_argument("--residual_gate", type=str, default="residual", help="[residual, static]")
    parser.add_argument("--gate_mode", type=str, default="residual", help="alias for residual_gate")
    parser.add_argument("--enrich_gamma", type=float, default=None, help="static gate value when residual_gate=static")
    parser.add_argument("--residual_gate_hidden_dim", type=int, default=128, help="learned gate hidden width")
    args = parser.parse_args()

    return args
