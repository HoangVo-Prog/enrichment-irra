import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter


def meter_scalar(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return value


def _should_meter(key, value):
    if key.startswith("_") or not (torch.is_tensor(value) or isinstance(value, (float, int))):
        return False
    return (
        key in {"loss", "img_acc", "txt_acc", "mlm_acc", "temperature"}
        or "loss" in key
        or key.startswith("pool_")
        or key.startswith("target_")
        or key.startswith("mixer/")
    )


def _move_batch(batch, device, skip_images=False):
    moved = {}
    for key, value in batch.items():
        if skip_images and key == "images":
            continue
        else:
            moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


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


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, target_pool=None):

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
        "loss", "sdm_loss", "itc_loss", "id_loss", "mlm_loss", "target_enrichment_loss",
        "target_retrieval_loss", "img_acc", "txt_acc", "mlm_acc"
    ]}
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    if _should_run_eval(args, 0):
        _ = evaluator.eval(
            model.eval(),
            use_target_enrichment=_target_enrichment_active(args, start_epoch),
        )

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            global_step = arguments["iteration"] + 1
            use_target = target_pool is not None and _target_enrichment_active(args, epoch)
            skip_images = bool(getattr(args, "pnp_text_only", False))
            batch = _move_batch(batch, device, skip_images=skip_images)

            target_cache = None
            if use_target:
                target_cache = target_pool.get_train_cache(model, batch, epoch, global_step)

            ret = model(batch, epoch=epoch, current_step=global_step, target_cache=target_cache)
            if target_cache is not None and "diagnostics" in target_cache:
                ret.update(target_cache["diagnostics"])
            total_loss = ret["loss"] if "loss" in ret else sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['caption_ids'].shape[0]
            for key, value in ret.items():
                if _should_meter(key, value):
                    if key not in meters:
                        meters[key] = AverageMeter()
                    meters[key].update(meter_scalar(value), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()
            arguments["iteration"] = global_step

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for key, meter in meters.items():
                    if meter.avg > 0:
                        info_str += f", {key}: {meter.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        if 'temperature' in ret:
            tb_writer.add_scalar('temperature', meter_scalar(ret['temperature']), epoch)
        for key, meter in meters.items():
            if meter.avg > 0:
                tb_writer.add_scalar(key, meter.avg, epoch)

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

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
