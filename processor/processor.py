import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.wandb_utils import log_wandb


def _scalar_value(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return value.detach().float().item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_trainable_loss(value):
    return torch.is_tensor(value) and value.numel() == 1 and value.requires_grad


def _is_loss_key(key):
    return key == "loss" or key.endswith("_loss")


def _should_track_log_scalar(key):
    return (
        key in {"loss", "img_acc", "txt_acc", "mlm_acc", "temperature"}
        or "loss" in key
        or key.endswith("grad_norm")
    )


def _should_track_wandb_scalar(key):
    return (
        key in {"loss", "img_acc", "txt_acc", "mlm_acc"}
        or "loss" in key
        or key.endswith("grad_norm")
        or key.startswith("pool_")
        or key.startswith("target_")
        or key.startswith("mixer/")
    )


def _grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.float().norm(2).item()
        total += param_norm ** 2
    return total ** 0.5


def _loss_grad_norm(loss, parameters):
    if not parameters or not _is_trainable_loss(loss):
        return 0.0
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_norm = grad.detach().float().norm(2).item()
        total += grad_norm ** 2
    return total ** 0.5


def _iter_loss_grad_sources(ret):
    sources = {}
    explicit_sources = ret.get("_loss_grad_sources", {})
    if isinstance(explicit_sources, dict):
        for key, value in explicit_sources.items():
            if _is_loss_key(key):
                sources[key] = value

    for key, value in ret.items():
        if key == "_loss_grad_sources":
            continue
        if _is_loss_key(key) and key not in sources:
            sources[key] = value
    return sources.items()


def _update_meter(meters, key, value, batch_size):
    if key not in meters:
        meters[key] = AverageMeter()
    meters[key].update(value, batch_size)


def _format_learning_rates(scheduler):
    try:
        values = scheduler.get_lr()
    except AttributeError:
        values = []
    unique_values = []
    for value in values:
        value = float(value)
        if not any(abs(value - existing) < 1e-16 for existing in unique_values):
            unique_values.append(value)
    if not unique_values:
        return "n/a"
    return "/".join("{:.2e}".format(value) for value in unique_values)


def _format_loss_meters(meters):
    preferred_keys = (
        "loss",
        "host_loss",
        "sdm_loss",
        "itc_loss",
        "id_loss",
        "mlm_loss",
        "cmpm_loss",
        "target_enrichment_loss",
        "target_retrieval_loss",
    )
    ordered_keys = []
    seen = set()
    for key in preferred_keys:
        if key in meters:
            ordered_keys.append(key)
            seen.add(key)
    for key in sorted(meters.keys()):
        if key not in seen and _is_loss_key(key):
            ordered_keys.append(key)

    parts = []
    for key in ordered_keys:
        meter = meters[key]
        if meter.count <= 0:
            continue
        label = "total_loss" if key == "loss" else key
        parts.append("{}: {:.4f}".format(label, meter.avg))
    return ", ".join(parts) if parts else "total_loss: 0.0000"


def _move_batch(batch, device, skip_images=False):
    moved = {}
    for key, value in batch.items():
        if skip_images and key == "images":
            continue
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _set_loader_epoch(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    batch_sampler_inner = getattr(batch_sampler, "sampler", None)
    if hasattr(batch_sampler_inner, "set_epoch") and batch_sampler_inner is not sampler:
        batch_sampler_inner.set_epoch(epoch)


def _target_enrichment_active(args, epoch):
    enrichment_start = getattr(args, "enrichment_start", 1)
    if enrichment_start < 1:
        raise ValueError("--enrichment_start must be a positive integer")
    return getattr(args, "target_enrichment", False) and epoch >= enrichment_start


def _should_run_eval(args, epoch):
    eval_after_epoch = getattr(args, "eval_after_epoch", 0)
    if eval_after_epoch < 0:
        raise ValueError("--eval_after_epoch must be a non-negative integer")
    return epoch >= eval_after_epoch


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _set_wandb_summary(wandb_run, summary_values):
    if wandb_run is None or not hasattr(wandb_run, "summary"):
        return
    for key, value in summary_values.items():
        wandb_run.summary[key] = int(value)


def _count_module_tensors(module, name_filter=None):
    def _include(name):
        return name_filter is None or name_filter(name)

    total_params = 0
    trainable_params = 0
    for name, parameter in module.named_parameters():
        if not _include(name):
            continue
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    buffer_params = sum(
        buffer.numel()
        for name, buffer in module.named_buffers()
        if _include(name)
    )
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "buffers": buffer_params,
    }


def _flatten_param_summary(prefix, stats):
    return {
        f"{prefix}_total_params": stats["total_params"],
        f"{prefix}_trainable_params": stats["trainable_params"],
        f"{prefix}_frozen_params": stats["frozen_params"],
        f"{prefix}_buffers": stats["buffers"],
    }


def _log_param_scope(logger, label, stats):
    logger.info(
        "%s params: total=%s (%.3fM), trainable=%s (%.3fM), frozen=%s (%.3fM), "
        "buffers=%s (%.3fM)",
        label,
        f"{stats['total_params']:,}",
        stats["total_params"] / 1_000_000.0,
        f"{stats['trainable_params']:,}",
        stats["trainable_params"] / 1_000_000.0,
        f"{stats['frozen_params']:,}",
        stats["frozen_params"] / 1_000_000.0,
        f"{stats['buffers']:,}",
        stats["buffers"] / 1_000_000.0,
    )


def _log_enrichment_branch_size(model, logger, wandb_run=None):
    model = _unwrap_model(model)
    target_enricher = getattr(model, "target_enricher", None)
    model_stats = _count_module_tensors(model)
    host_stats = _count_module_tensors(
        model,
        name_filter=lambda name: not name.startswith("target_enricher."),
    )
    summary = {}
    summary.update(_flatten_param_summary("model", model_stats))
    summary.update(_flatten_param_summary("host", host_stats))

    _log_param_scope(logger, "Model", model_stats)
    _log_param_scope(logger, "Host", host_stats)

    if target_enricher is None:
        logger.info("Target enrichment branch params: disabled")
        summary.update(_flatten_param_summary("target_enrichment_branch", {
            "total_params": 0,
            "trainable_params": 0,
            "frozen_params": 0,
            "buffers": 0,
        }))
        _set_wandb_summary(wandb_run, summary)
        return

    target_stats = _count_module_tensors(target_enricher)
    _log_param_scope(logger, "Target enrichment branch", target_stats)
    summary.update(_flatten_param_summary("target_enrichment_branch", target_stats))
    _set_wandb_summary(wandb_run, summary)


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, target_pool=None, wandb_run=None):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0
    arguments["epoch"] = start_epoch

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')
    if get_rank() == 0:
        _log_enrichment_branch_size(model, logger, wandb_run=wandb_run)
    if target_pool is not None and getattr(args, "enrichment_start", 1) > 1:
        logger.info(
            "Target enrichment delayed until epoch {}; earlier epochs use host training only".format(
                args.enrichment_start
            )
        )
    if getattr(args, "eval_after_epoch", 0) > 0:
        logger.info(
            "Evaluation delayed until epoch {}; earlier epochs skip validation".format(
                args.eval_after_epoch
            )
        )

    meters = {name: AverageMeter() for name in [
        "loss",
        "sdm_loss",
        "itc_loss",
        "id_loss",
        "mlm_loss",
        "host_loss",
        "target_enrichment_loss",
        "target_retrieval_loss",
        "img_acc",
        "txt_acc",
        "mlm_acc",
        "grad_norm",
        "host_loss_grad_norm",
        "sdm_loss_grad_norm",
        "itc_loss_grad_norm",
        "id_loss_grad_norm",
        "mlm_loss_grad_norm",
        "target_enrichment_loss_grad_norm",
        "target_retrieval_loss_grad_norm",
    ]}
    wandb_meters = {}
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    if _should_run_eval(args, 0):
        initial_top1 = evaluator.eval(
            model.eval(),
            use_target_enrichment=_target_enrichment_active(args, start_epoch),
        )
        if get_rank() == 0:
            initial_metrics = dict(getattr(evaluator, "last_metrics", {}))
            initial_metrics["eval/top_R1"] = initial_top1
            if "eval/ablation_best_R1" in initial_metrics:
                initial_metrics["eval/best_ablation_R1"] = initial_top1
            log_wandb(wandb_run, initial_metrics, step=0, epoch=start_epoch - 1)
            logger.info("Initial R1: {:.2f}".format(initial_top1))

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        for meter in wandb_meters.values():
            meter.reset()
        model.train()
        model.epoch = epoch
        _set_loader_epoch(train_loader, epoch)
        use_target = target_pool is not None and _target_enrichment_active(args, epoch)
        if target_pool is not None and epoch == getattr(args, "enrichment_start", 1):
            logger.info("Target enrichment starts at epoch {}".format(epoch))

        for n_iter, batch in enumerate(train_loader):
            global_step = arguments["iteration"] + 1
            skip_images = bool(getattr(args, "pnp_text_only", False))
            batch = _move_batch(batch, device, skip_images=skip_images)

            target_cache = None
            if use_target:
                target_cache = target_pool.get_train_cache(model, batch, epoch, global_step)

            ret = model(batch, epoch=epoch, current_step=global_step, target_cache=target_cache)
            if target_cache is not None and "diagnostics" in target_cache:
                for diag_key, diag_value in target_cache["diagnostics"].items():
                    if isinstance(diag_value, (int, float)):
                        ret[diag_key] = diag_value
            total_loss = ret.get("loss")
            if total_loss is None:
                total_loss = sum(
                    value for key, value in ret.items()
                    if "loss" in key and _is_trainable_loss(value)
                )

            batch_size = batch['caption_ids'].shape[0]
            _update_meter(meters, "loss", _scalar_value(total_loss), batch_size)
            _update_meter(wandb_meters, "loss", _scalar_value(total_loss), batch_size)
            for key, value in ret.items():
                if key == "loss":
                    continue
                scalar = _scalar_value(value)
                if scalar is None:
                    continue
                if _should_track_log_scalar(key):
                    _update_meter(meters, key, scalar, batch_size)
                if _should_track_wandb_scalar(key):
                    _update_meter(wandb_meters, key, scalar, batch_size)

            optimizer.zero_grad()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            loss_grad_sources = dict(_iter_loss_grad_sources(ret))
            loss_grad_sources["loss"] = total_loss
            for loss_key, loss_value in loss_grad_sources.items():
                grad_norm_key = f"{loss_key}_grad_norm"
                grad_norm_value = _loss_grad_norm(loss_value, trainable_params)
                _update_meter(meters, grad_norm_key, grad_norm_value, batch_size)
                _update_meter(wandb_meters, grad_norm_key, grad_norm_value, batch_size)
            total_loss.backward()
            grad_norm_value = _grad_norm(model.parameters())
            _update_meter(meters, "grad_norm", grad_norm_value, batch_size)
            _update_meter(wandb_meters, "grad_norm", grad_norm_value, batch_size)
            optimizer.step()
            synchronize()
            arguments["iteration"] = global_step

            if (n_iter + 1) % log_period == 0:
                grad_avg = meters["grad_norm"].avg if meters["grad_norm"].count > 0 else 0.0
                logger.info(
                    "Epoch[{}] Iteration[{}/{}], {}, grad_norm: {:.4f}, lr: {}".format(
                        epoch,
                        n_iter + 1,
                        len(train_loader),
                        _format_loss_meters(meters),
                        grad_avg,
                        _format_learning_rates(scheduler),
                    )
                )
                if get_rank() == 0:
                    train_metrics = {
                        "train/{}".format(key): meter.avg
                        for key, meter in wandb_meters.items()
                        if meter.count > 0
                    }
                    train_metrics["train/lr"] = scheduler.get_lr()[0]
                    train_metrics["train/temperature"] = _scalar_value(ret.get("temperature"))
                    log_wandb(wandb_run, train_metrics, step=global_step, epoch=epoch)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        if 'temperature' in ret:
            tb_writer.add_scalar('temperature', _scalar_value(ret['temperature']), epoch)
        for key, meter in meters.items():
            if meter.count > 0:
                tb_writer.add_scalar(key, meter.avg, epoch)
        if get_rank() == 0:
            epoch_metrics = {
                "train_epoch/{}".format(key): meter.avg
                for key, meter in wandb_meters.items()
                if meter.count > 0
            }
            epoch_metrics["train_epoch/lr"] = scheduler.get_lr()[0]
            epoch_metrics["train_epoch/temperature"] = _scalar_value(ret.get("temperature"))
            log_wandb(wandb_run, epoch_metrics, step=arguments["iteration"], epoch=epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0 and _should_run_eval(args, epoch):
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(
                        model.module.eval(),
                        use_target_enrichment=_target_enrichment_active(args, epoch),
                    )
                else:
                    top1 = evaluator.eval(
                        model.eval(),
                        use_target_enrichment=_target_enrichment_active(args, epoch),
                    )

                new_best = top1 > best_top1
                if new_best:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                eval_metrics = dict(getattr(evaluator, "last_metrics", {}))
                eval_metrics["eval/top_R1"] = top1
                eval_metrics["eval/best_R1"] = best_top1
                if "eval/ablation_best_R1" in eval_metrics:
                    eval_metrics["eval/best_ablation_R1"] = best_top1
                log_wandb(wandb_run, eval_metrics, step=arguments["iteration"], epoch=epoch)
                logger.info(
                    "Epoch {} R1: {:.2f}; best R1: {:.2f} at epoch {}".format(
                        epoch,
                        top1,
                        best_top1,
                        arguments["epoch"],
                    )
                )

                torch.cuda.empty_cache()
                if new_best:
                    if wandb_run is not None:
                        wandb_run.summary["best_R1"] = float(best_top1)
                        wandb_run.summary["best_R1_row"] = str(getattr(evaluator, "last_best_task", ""))
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader, args=None):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    if args is None:
        args = getattr(model, "args", None)
    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())
