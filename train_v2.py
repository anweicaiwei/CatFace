import argparse
import os
from datetime import datetime

import torch
from torch import distributed
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import logging
from backbones import get_model
from dataset import get_dataloader, get_record_info
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging

# 调用另一个 Python 文件
#subprocess.run(["python", "path/to/other_script.py"])

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."
try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # 获取当前时间作为训练开始时间
    training_start_time = datetime.now()
    
    # get config
    cfg = get_config(args.config)
    
    # 获取record文件的样本数和种类数
    print("读取Record文件信息...")
    num_samples, num_classes = get_record_info(cfg.rec)
    print(f"Record文件信息: 样本数={num_samples}, 种类数={num_classes}")
    
    # 更新配置文件中的参数
    cfg.num_image = num_samples
    cfg.num_classes = num_classes
    print("配置文件参数已更新")
    
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    # Create timestamp folder for output
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(cfg.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化日志系统
    init_logging(rank, output_dir)
    logging.info("="*80)
    logging.info(f"训练开始于: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"配置文件: {args.config}")
    logging.info(f"输出目录: {output_dir}")
    logging.info("="*80)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    #backbone = backbone.half()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()
    
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    
    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        # When resuming, we look for checkpoints in the latest timestamp folder
        # This assumes there's only one checkpoint folder, or you need to modify to select a specific one
        timestamp_folders = [f for f in os.listdir(cfg.output) if os.path.isdir(os.path.join(cfg.output, f))]
        if timestamp_folders:
            latest_folder = max(timestamp_folders)
            checkpoint_path = os.path.join(cfg.output, latest_folder, f"checkpoint_gpu_{rank}.pt")
            if os.path.exists(checkpoint_path):
                logging.info(f"从检查点恢复训练: {checkpoint_path}")
                dict_checkpoint = torch.load(checkpoint_path)
                start_epoch = dict_checkpoint["epoch"]
                global_step = dict_checkpoint["global_step"]
                backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
                module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
                opt.load_state_dict(dict_checkpoint["state_optimizer"])
                lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
                logging.info(f"成功恢复到 epoch: {start_epoch}, global_step: {global_step}")
                del dict_checkpoint
            else:
                logging.warning(f"未找到检查点文件: {checkpoint_path}")
        else:
            logging.warning(f"在 {cfg.output} 中未找到时间戳文件夹")

    logging.info("\n===== 训练配置 =====")
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(f": {key}{' ' * num_space}{str(value)}")
    logging.info("==================\n")

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        output_dir=output_dir,  # 传入输出目录用于保存最佳模型
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        total_epoch=cfg.num_epoch,
        start_step = global_step,
        writer=summary_writer
    )

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        # 重置当前epoch的损失统计和准确率统计
        loss_am = AverageMeter()
        accuracy_am = AverageMeter()
        # 初始化当前轮次的最佳批次准确率和对应的批次号
        current_epoch_best_accuracy = 0.0
        current_epoch_best_step = 0
        
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        
        # 为当前轮次初始化进度条
        if rank == 0:
            # 显式计算每轮的总批次数
            total_batches = cfg.num_image // cfg.batch_size
            train_loader_tqdm = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{cfg.num_epoch}", 
                unit="batch", 
                total=total_batches,
                leave=False,
                dynamic_ncols=True,
                position=0,
                ncols=100,
                mininterval=0.1,
                smoothing=0.1,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )
        else:
            train_loader_tqdm = train_loader
            
        for idx, (img, local_labels) in enumerate(train_loader_tqdm):
            # 获取当前 batch 的文件名
            saveImage = False
            if rank == 0 and idx == 0:
                saveImage = True
            global_step += 1
            #img = img.half().cuda()
            #local_embeddings = backbone(img, saveImage)
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
                    lr_scheduler.step()  # 在优化器步骤后调用学习率调度器
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
                    lr_scheduler.step()  # 在优化器步骤后调用学习率调度器

            with torch.no_grad():
                # 每50批次记录一次额外的详细日志
                if global_step % 50 == 0 and rank == 0:
                    # 计算当前批次的准确率 - 使用与forward方法相同的逻辑
                    # 归一化嵌入向量和权重
                    norm_embeddings = torch.nn.functional.normalize(local_embeddings)
                    
                    # 检查是否使用部分FC
                    if module_partial_fc.sample_rate < 1:
                        # 如果使用部分FC，需要获取当前批次使用的权重和调整后的标签
                        # 复制forward方法中的逻辑
                        labels = local_labels.clone().view(-1, 1)
                        index_positive = (module_partial_fc.class_start <= labels) & \
                                        (labels < module_partial_fc.class_start + module_partial_fc.num_local)
                        labels[~index_positive] = -1
                        labels[index_positive] -= module_partial_fc.class_start
                        
                        # 获取当前批次使用的权重
                        weight_activated = module_partial_fc.sample(labels, index_positive)
                        norm_weight_activated = torch.nn.functional.normalize(weight_activated)
                        
                        # 计算logits
                        cosine = torch.mm(norm_embeddings, norm_weight_activated.t())
                        
                        # 获取预测标签
                        _, predictions = torch.max(cosine, dim=1)
                        
                        # 只计算有效的标签（即属于当前GPU的样本）的准确率
                        valid_mask = (labels.squeeze() != -1)
                        if valid_mask.sum() > 0:
                            accuracy = (predictions[valid_mask] == labels.squeeze()[valid_mask]).float().mean().item()
                        else:
                            accuracy = 0.0
                    else:
                        # 如果不使用部分FC，直接计算
                        norm_weight = torch.nn.functional.normalize(module_partial_fc.weight.detach())
                        cosine = torch.mm(norm_embeddings, norm_weight.t())
                        _, predictions = torch.max(cosine, dim=1)
                        accuracy = (predictions == local_labels).float().mean().item()

                    # batch_loss = loss.item()
                    # avg_loss = loss_am.avg
                    # current_lr = lr_scheduler.get_last_lr()[0]
                    
                    # 如果有wandb_logger，也记录准确率
                    if wandb_logger:
                        wandb_logger.log({
                            'Accuracy/Train Accuracy': accuracy,
                            'Process/Step': global_step,
                            'Process/Epoch': epoch
                        })
                    
                    # 检查当前批次准确率是否为当前轮次最佳（仅用于日志记录）
                    if accuracy > current_epoch_best_accuracy:
                        current_epoch_best_accuracy = accuracy
                        current_epoch_best_step = global_step
                
                # 更新进度条信息
                if rank == 0:
                    batch_loss = loss.item()
                    avg_loss = loss_am.avg
                    current_lr = lr_scheduler.get_last_lr()[0]
                    train_loader_tqdm.set_postfix({'loss': f'{batch_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.6f}'})
                
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
                    
                # 计算当前batch的准确率 - 使用与forward方法相同的逻辑
                # 归一化嵌入向量和权重
                norm_embeddings = torch.nn.functional.normalize(local_embeddings)
                
                # 检查是否使用部分FC
                if module_partial_fc.sample_rate < 1:
                    # 如果使用部分FC，需要获取当前批次使用的权重和调整后的标签
                    # 复制forward方法中的逻辑
                    labels = local_labels.clone().view(-1, 1)
                    index_positive = (module_partial_fc.class_start <= labels) & \
                                    (labels < module_partial_fc.class_start + module_partial_fc.num_local)
                    labels[~index_positive] = -1
                    labels[index_positive] -= module_partial_fc.class_start
                    
                    # 获取当前批次使用的权重
                    weight_activated = module_partial_fc.sample(labels, index_positive)
                    norm_weight_activated = torch.nn.functional.normalize(weight_activated)
                    
                    # 计算logits
                    cosine = torch.mm(norm_embeddings, norm_weight_activated.t())
                    
                    # 获取预测标签
                    _, predictions = torch.max(cosine, dim=1)
                    
                    # 只计算有效的标签（即属于当前GPU的样本）的准确率
                    valid_mask = (labels.squeeze() != -1)
                    if valid_mask.sum() > 0:
                        accuracy = (predictions[valid_mask] == labels.squeeze()[valid_mask]).float().mean().item()
                    else:
                        accuracy = 0.0
                else:
                    # 如果不使用部分FC，直接计算
                    norm_weight = torch.nn.functional.normalize(module_partial_fc.weight.detach())
                    cosine = torch.mm(norm_embeddings, norm_weight.t())
                    _, predictions = torch.max(cosine, dim=1)
                    accuracy = (predictions == local_labels).float().mean().item()
                
                accuracy_am.update(accuracy, img.size(0))
                
                # 计算准确率用于callback_logging，每50批次计算一次
                train_accuracy = None
                if global_step % 50 == 0 and rank == 0:
                    train_accuracy = accuracy
                
                loss_am.update(loss.item(), 1)
                # 每50批次调用一次callback_logging
                if global_step % 50 == 0 and rank == 0:
                    callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp, train_accuracy)
        
        # 轮次结束后执行验证
        with torch.no_grad():
            callback_verification(global_step, backbone)
            
            # 记录当前轮次中准确率最高的批次信息（仅用于日志记录）
            if current_epoch_best_step > 0 and rank == 0:
                # 记录当前轮次最佳批次信息
                logging.info(f"[第{epoch+1}轮结束] 当前轮次中准确率最高的批次为 #{current_epoch_best_step}, 批次准确率: {current_epoch_best_accuracy:.4f}")
                
                # 如果有wandb_logger，记录轮次最佳批次信息
                if wandb_logger:
                    wandb_logger.log({
                        'Best/Epoch Batch Accuracy': current_epoch_best_accuracy,
                        'Process/Best Step in Epoch': current_epoch_best_step,
                        'Process/Best Epoch': epoch+1,
                        'Process/Step': global_step
                    })
                
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        # 保存最后一轮模型
        last_model_path = os.path.join(output_dir, "last_model.pt")
        torch.save(backbone.module.state_dict(), last_model_path)
        logging.info(f"已保存最后一轮模型到: {last_model_path}")
   
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(last_model_path)
            wandb_logger.log_artifact(model)
            logging.info(f"已将最终模型上传到 WandB: {artifact_name}")
    
    # 记录训练结束信息
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    logging.info("="*80)
    logging.info(f"训练结束于: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"总训练时间: {str(training_duration).split('.')[0]} (小时:分钟:秒)")
    logging.info(f"最终输出目录: {output_dir}")
    logging.info("训练完成！")
    logging.info("="*80)



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())