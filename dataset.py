import numbers
import os
import queue as Queue
import threading
from functools import partial
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


def get_record_info(root_dir):
    """
    获取Record文件的样本数和种类数
    
    参数：
        root_dir (str): 数据集根目录路径
    
    返回：
        tuple: (num_samples, num_classes)，样本数和种类数
    """
    import mxnet as mx
    import numpy as np
    
    # 构建Mxnet RecordIO数据文件路径
    rec_path = os.path.join(root_dir, 'train.rec')
    idx_path = os.path.join(root_dir, 'train.idx')
    
    # 读取record文件
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
    
    # 读取头部信息
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    
    num_samples = 0
    num_classes = 0
    
    if header.flag > 0:
        # 对于有头部标记的record文件
        num_samples = int(header.label[0]) - 1  # 减去1是因为索引从1开始
        num_classes = int(header.label[1])
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        # 对于没有头部标记的record文件
        imgidx = np.array(list(imgrec.keys))
        num_samples = len(imgidx)
        
        # 需要遍历所有样本获取最大标签值作为种类数
        max_label = -1
        for idx in imgidx[:10000]:  # 只遍历前10000个样本以提高速度
            s = imgrec.read_idx(idx)
            header, _ = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            if label > max_label:
                max_label = label
        num_classes = int(max_label) + 1
    
    return num_samples, num_classes


def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:
    """
    获取数据加载器函数，支持多种数据格式和加载方式

    参数：
        root_dir (str): 数据集根目录路径，或"synthetic"表示使用合成数据
        local_rank (int): 当前进程的本地GPU排名
        batch_size (int): 每个批次的样本数量
        dali (bool, optional): 是否使用DALI加速数据加载，默认False
        dali_aug (bool, optional): 是否在DALI中进行数据增强，默认False
        seed (int, optional): 随机种子，用于数据打乱和 worker 初始化，默认2048
        num_workers (int, optional): 数据加载的工作进程数，默认2

    返回：
        Iterable: 数据加载器对象，支持迭代获取批次数据
    """
    # 打印数据加载开始信息
    print("Loading Data...")
    # 构建Mxnet RecordIO数据文件路径
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    print(f"Record file: {rec}")
    print(f"Index file: {idx}")
    # 初始化训练数据集对象
    train_set = None

    # 处理合成数据
    if root_dir == "synthetic":
        # 使用SyntheticDataset生成随机合成图像数据
        train_set = SyntheticDataset()
        # 合成数据不支持DALI加速，强制关闭
        dali = False

    # 处理Mxnet RecordIO格式数据
    elif os.path.exists(rec) and os.path.exists(idx):
        # 使用MXFaceDataset加载RecordIO格式的人脸数据集
        print("Loading MXFaceDataset...")
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # 处理Image Folder格式数据
    else:
        # 定义图像变换和数据增强操作
        transform = transforms.Compose([
             transforms.Resize((112, 112), interpolation='bilinear'),  # 改为线性插值避免黑边
             transforms.RandomHorizontalFlip(),  # 随机水平翻转
             transforms.RandomRotation(10),  # 随机旋转±10度
             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机仿射变换
             transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # 随机透视变换
             transforms.ToTensor(),  # 转换为张量
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到[-1, 1]范围
             transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 随机擦除
             ])
        # 使用ImageFolder加载图像文件夹格式的数据集
        train_set = ImageFolder(root_dir, transform)
        #train_set = ImageFolder(root_dir, None)  # 注释掉的代码：不使用变换的版本

    # 如果启用DALI加速，返回DALI数据加载器
    if dali:
        # 打印DALI数据加载器信息
        print("Loading DALI DataLoader...")
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    # 获取分布式训练信息
    rank, world_size = get_dist_info()
    # 创建分布式采样器，用于多GPU训练时的数据分片
    train_sampler = DistributedSampler(
        #train_set, num_replicas=world_size, rank=rank, shuffle=False, seed=seed)  # 注释掉的代码：不打乱数据
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)  # 打乱数据版本

    # 设置worker初始化函数，用于确保不同进程的随机性
    if seed is None:
        init_fn = None
    else:
        # 使用partial创建带参数的worker初始化函数
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    # 创建DataLoaderX对象，支持异步数据加载和CUDA流预加载
    train_loader = DataLoaderX(
        local_rank=local_rank,  # 本地GPU排名
        dataset=train_set,  # 数据集对象
        batch_size=batch_size,  # 批次大小
        sampler=train_sampler,  # 采样器（分布式或普通）
        num_workers=num_workers,  # 工作进程数
        pin_memory=True,  # 是否将数据固定在内存中以加速GPU传输
        drop_last=True,  # 是否丢弃最后一个不完整的批次
        worker_init_fn=init_fn,  # worker初始化函数
    )

    # 返回创建好的数据加载器
    return train_loader

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=12):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            self.progress_bar.close()
            raise StopIteration
        self.progress_bar.update(1)
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((112, 112), interpolation='bilinear'),  # 改为线性插值避免黑边
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(10),
             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
             transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class SyntheticDataset(Dataset):
    """
    合成数据集类，用于生成随机图像数据进行测试或调试
    所有样本都返回相同的随机生成图像和固定标签
    """
    def __init__(self):
        """
        初始化合成数据集
        生成一张随机图像并进行预处理
        """
        super(SyntheticDataset, self).__init__()
        # 生成随机图像：112x112像素，3通道（RGB），像素值范围0-255
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        # 转换通道顺序：从HWC（高度、宽度、通道）转换为CHW（通道、高度、宽度）
        img = np.transpose(img, (2, 0, 1))
        # 将NumPy数组转换为PyTorch张量，并转换为float类型
        img = torch.from_numpy(img).squeeze(0).float()
        # 归一化图像：将像素值从[0, 255]范围转换为[-1, 1]范围
        img = ((img / 255) - 0.5) / 0.5
        # 保存生成的图像和固定标签
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        
        参数：
            index (int): 样本索引（此处未使用，因为所有样本都相同）
            
        返回：
            tuple: 包含图像张量和标签的元组
        """
        return self.img, self.label

    def __len__(self):
        """
        获取数据集的长度
        
        返回：
            int: 数据集的样本数量（固定为1,000,000）
        """
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        # 确保所有图像都调整为112x112大小，使用线性插值避免黑边
        images = fn.resize(images, size=(112, 112), interp_type=types.DALIInterpType.INTERP_LINEAR)
        if dali_aug:
            print("DALI Augmentation Enabled")
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)
            # 添加更多增强操作
            # 随机亮度调整
            condition_brightness = fn.random.coin_flip(probability=0.3)
            brightness = fn.random.uniform(range=(-0.2, 0.2), dtype=types.FLOAT)
            images = multiplexing(condition_brightness, fn.brightness(images, brightness), images)
            # 随机对比度调整
            condition_contrast = fn.random.coin_flip(probability=0.3)
            contrast = fn.random.uniform(range=(0.8, 1.2), dtype=types.FLOAT)
            images = multiplexing(condition_contrast, fn.contrast(images, contrast), images)
            # 随机饱和度调整
            condition_saturation = fn.random.coin_flip(probability=0.3)
            saturation = fn.random.uniform(range=(0.8, 1.2), dtype=types.FLOAT)
            images = multiplexing(condition_saturation, fn.saturation(images, saturation), images)
            # 随机仿射变换
            condition_affine = fn.random.coin_flip(probability=0.3)
            translate_x = fn.random.uniform(range=(-0.1, 0.1), dtype=types.FLOAT)
            translate_y = fn.random.uniform(range=(-0.1, 0.1), dtype=types.FLOAT)
            scale = fn.random.uniform(range=(0.9, 1.1), dtype=types.FLOAT)
            images = multiplexing(condition_affine, fn.warp_affine(images, matrix=[[scale, 0, translate_x*112], [0, scale, translate_y*112]], size=(112, 112)), images)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()