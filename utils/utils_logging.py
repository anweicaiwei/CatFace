import logging
import os
import sys
import logging.handlers


class BatchLogFilter(logging.Filter):
    """
    过滤器类，用于过滤掉包含"[训练批次 #"或"[每50批次]"的日志
    仅应用于控制台日志处理器，文件日志不受影响
    """
    def filter(self, record):
        # 如果日志消息包含"[训练批次 #"，返回False表示不记录
        return "[训练批次 #" not in record.msg


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logging(rank, models_root):
    """
    初始化日志记录器，增强日志功能
    - 同时输出到控制台和文件
    - 更详细的日志格式（包含时间戳、日志级别、模块名、行号）
    - 设置不同的日志级别阈值
    - 过滤第三方库的DEBUG日志，只记录项目自己的DEBUG日志
    """
    # 确保输出目录存在
    os.makedirs(models_root, exist_ok=True)
    
    # 创建根日志记录器
    log_root = logging.getLogger()
    log_root.setLevel(logging.DEBUG)  # 设置根日志级别为DEBUG，允许所有级别的日志通过
    
    # 清除现有的处理器（避免重复日志）
    for handler in log_root.handlers[:]:
        log_root.removeHandler(handler)
    
    # 定义详细的日志格式
    formatter = logging.Formatter(
        f"%(asctime)s - [%(levelname)s] - %(name)s:%(lineno)d - Rank:{rank} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 设置第三方库的日志级别，避免DEBUG日志污染
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torchvision').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    
    # 文件日志处理器（保存到training.log）
    log_file = os.path.join(models_root, "training.log")
    # log_file = log_file.replace('/', '\\')  # 确保Windows路径正确
    handler_file = logging.FileHandler(log_file, encoding='utf-8')
    handler_file.setLevel(logging.DEBUG)  # 文件日志记录所有级别
    handler_file.setFormatter(formatter)
    log_root.addHandler(handler_file)
    
    # 添加调试信息
    log_root.debug(f"文件日志处理器已添加，日志文件路径: {log_file}")
    log_root.debug(f"目录是否存在: {os.path.exists(models_root)}")
    log_root.debug(f"文件是否可写: {os.access(models_root, os.W_OK)}")
    
    # 控制台日志处理器（只显示INFO及以上级别）
    # 使用stderr而不是stdout，避免与tqdm进度条冲突
    handler_stream = logging.StreamHandler(sys.stderr)
    handler_stream.setLevel(logging.INFO)  # 控制台只显示重要信息
    handler_stream.setFormatter(formatter)
    # 添加过滤器，过滤掉训练批次相关的频繁日志
    handler_stream.addFilter(BatchLogFilter())
    log_root.addHandler(handler_stream)
    
    # 记录初始化信息
    log_root.info(f"日志系统初始化完成，日志文件: {log_file}")
    log_root.info(f'当前进程rank: {rank}')
    
    return log_root