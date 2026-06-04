import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from model.target_pool import TargetPoolManager

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter(object):
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _is_scalar(value):
    return torch.is_tensor(value) and value.dim() == 0


def _scalar_value(value):
    if torch.is_tensor(value):
        return value.detach().float().item()
    return value


def _move_batch(batch, device, keep_images_cpu=False):
    moved = {}
    for key, value in batch.items():
        if keep_images_cpu and key == "images":
            moved[key] = value
        else:
            moved[key] = value.to(device)
    return moved


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {"loss": AverageMeter()}

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    target_pool = TargetPoolManager(train_loader.dataset, args, logger) if getattr(args, "target_enrichment", False) else None

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            global_step = arguments["iteration"]
            use_target = target_pool is not None and epoch >= args.enrichment_start
            batch = _move_batch(
                batch,
                device,
                keep_images_cpu=use_target and getattr(args, "pnp_text_only", False),
            )
            target_cache = None
            if use_target:
                target_cache = target_pool.get_train_cache(
                    _unwrap_model(model), batch, epoch, global_step
                )
                pool_reused = target_cache.get("diagnostics", {}).get("pool_interval_reused")
                if torch.is_tensor(pool_reused) and pool_reused.item() == 0:
                    synchronize()

            ret = model(batch, epoch=epoch, current_step=global_step, target_cache=target_cache)
            if target_cache is not None and "diagnostics" in target_cache:
                ret.update(target_cache["diagnostics"])
            total_loss = ret["loss"] if "loss" in ret else sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['caption_ids'].shape[0]
            for key, value in ret.items():
                if not _is_scalar(value):
                    continue
                if key not in meters:
                    meters[key] = AverageMeter()
                meters[key].update(_scalar_value(value), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()
            arguments["iteration"] += 1

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', _scalar_value(ret['temperature']), epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader, args=None):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    top1 = evaluator.eval(model.eval())
