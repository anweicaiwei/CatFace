import logging
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed


class CallBackVerification(object):
    
    def __init__(self, val_targets, rec_prefix, output_dir, summary_writer=None, image_size=(112, 112), wandb_logger=None):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = []
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.output_dir = output_dir  # 保存最佳模型的目录
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)
            # 根据实际加载的验证集数量初始化最高准确率列表
            self.highest_acc_list = [0.0] * len(self.ver_list)
            
            # 检查是否有验证集被加载
            if len(self.ver_list) == 0:
                logging.warning(f"没有找到任何验证集文件。请检查配置文件中的val_targets和rec_prefix参数，确保验证集文件存在于{rec_prefix}目录下")
                logging.info(f"配置的val_targets: {val_targets}")

        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        # 初始化当前验证轮次的最佳准确率
        current_val_best_acc = 0.0
        current_val_best_name = ""
        
        # 检查是否有验证集
        if len(self.ver_list) == 0:
            logging.warning(f"[验证][{global_step}] 没有找到有效的验证集，跳过验证")
            return results
            
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )
            if self.wandb_logger:
                import wandb
                self.wandb_logger.log({
                    f'Acc/val-Acc1 {self.ver_name_list[i]}': acc1,
                    f'Acc/val-Acc2 {self.ver_name_list[i]}': acc2,
                    # f'Acc/val-std1 {self.ver_name_list[i]}': std1,
                    # f'Acc/val-std2 {self.ver_name_list[i]}': acc2,
                })
            
            # 更新当前验证轮次的最佳准确率
            if acc2 > current_val_best_acc:
                current_val_best_acc = acc2
                current_val_best_name = self.ver_name_list[i]

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
                # 保存最佳模型
                best_model_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save(backbone.module.state_dict(), best_model_path)
                # 同时保存一个带有验证集名称和准确率的模型文件
                detailed_best_model_path = os.path.join(self.output_dir, f"best_model_{self.ver_name_list[i]}_{acc2:.4f}.pt")
                torch.save(backbone.module.state_dict(), detailed_best_model_path)
                logging.info('[%s][%d]已保存验证集最佳模型到: %s' % (self.ver_name_list[i], global_step, best_model_path))
                logging.info('[%s][%d]已保存详细验证集最佳模型到: %s' % (self.ver_name_list[i], global_step, detailed_best_model_path))
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        found_datasets = []
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                try:
                    data_set = verification.load_bin(path, image_size)
                    # data_set 是 (data_list, issame_list)，issame_list 中的每个元素代表一对图像样本
                    num_pairs = len(data_set[1])
                    num_images = num_pairs * 2
                    print(f"验证集 {name} 包含 {num_pairs} 对样本（共 {num_images} 张图像）")
                    self.ver_list.append(data_set)
                    self.ver_name_list.append(name)
                    found_datasets.append(name)
                    logging.info(f"成功加载验证集: {name} - 路径: {path}")
                except Exception as e:
                    logging.error(f"加载验证集失败: {name} - 错误: {str(e)}")
            else:
                logging.warning(f"验证集文件不存在: {path}")
                
        if not found_datasets:
            logging.warning(f"在目录 {data_dir} 中没有找到任何验证集文件")
        else:
            logging.info(f"成功加载 {len(found_datasets)} 个验证集: {found_datasets}")

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0:
            logging.info("="*60)
            logging.info(f"开始验证 - Global Step: {num_update}")
            logging.info("="*60)
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()
            logging.info("验证完成\n")


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0,writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler,
                 accuracy: float = None):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                #time_now = (time.time() - self.time_start) / 3600
                #time_total = time_now / ((global_step + 1) / self.total_step)
                #time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                    # 如果提供了准确率，添加到TensorBoard
                    if accuracy is not None:
                        self.writer.add_scalar('accuracy/train_accuracy', accuracy, global_step)
                # 准备详细的日志信息
                log_info = {
                    "速度": f"{speed_total:.2f} 样本/秒",
                    "当前批次损失": f"{loss.val:.4f}",
                    "平均损失": f"{loss.avg:.4f}",
                    "学习率": f"{learning_rate:.6f}",
                    "轮次": epoch,
                    "全局步数": global_step,
                    "剩余时间": f"{time_for_end:.1f} 小时"
                }
                
                # 如果提供了准确率，添加到日志信息中
                if accuracy is not None:
                    log_info["训练准确率"] = f"{accuracy:.4f}"
                
                if fp16:
                    log_info["Fp16梯度缩放"] = f"{grad_scaler.get_scale():.2f}"
                
                # 生成格式化的日志消息
                msg = f"[训练批次 #{global_step}] "
                for key, value in log_info.items():
                    msg += f"{key}: {value}, "
                msg = msg.rstrip(", ")
                
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()