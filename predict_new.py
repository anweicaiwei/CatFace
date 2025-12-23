import argparse
import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
from backbones import get_model
from utils.utils_config import get_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path, config):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        config: 配置对象，包含网络类型、嵌入大小、FP16等参数
    
    Returns:
        加载好的模型和设备
    """
    logging.info(f"加载模型: {model_path}")
    logging.info(f"网络类型: {config.network}, 嵌入大小: {config.embedding_size}, FP16: {config.fp16}")
    
    # 创建模型
    model = get_model(
        config.network, 
        dropout=0.0, 
        fp16=config.fp16, 
        num_features=config.embedding_size
    )
    
    # 加载模型权重
    state_dict = torch.load(model_path)
    
    # 处理分布式训练保存的模型（如果需要）
    if 'module.' in list(state_dict.keys())[0]:
        # 移除 'module.' 前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    
    # 将模型移至GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    logging.info(f"模型加载完成，使用设备: {device}")
    return model, device

def preprocess_image(image_path, config):
    """
    预处理输入图像
    
    Args:
        image_path: 图像文件路径
        config: 配置对象，包含图像大小、均值和标准差等参数
    
    Returns:
        预处理后的图像张量
    """
    # 打开图像
    img = Image.open(image_path).convert('RGB')
    
    # 获取图像大小，默认112
    image_size = getattr(config, 'image_size', 112)
    # 获取归一化参数，默认[0.5, 0.5, 0.5]
    mean = getattr(config, 'mean', [0.5, 0.5, 0.5])
    std = getattr(config, 'std', [0.5, 0.5, 0.5])
    
    # 定义预处理变换（与训练时相同，但不使用随机翻转）
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # 应用变换
    img_tensor = transform(img)
    
    return img_tensor

def predict(model, image_paths, device, config, batch_size=8):
    """
    对多张图像进行批量预测
    
    Args:
        model: 加载好的模型
        image_paths: 图像文件路径列表
        device: 运行设备
        config: 配置对象
        batch_size: 批量处理大小
    
    Returns:
        特征向量列表
    """
    all_features = []
    
    # 批量处理图像
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # 预处理批量图像
        batch_tensors = []
        for path in batch_paths:
            if not os.path.exists(path):
                logging.warning(f"图像文件不存在: {path}")
                continue
            try:
                img_tensor = preprocess_image(path, config)
                batch_tensors.append(img_tensor)
            except Exception as e:
                logging.error(f"处理图像 {path} 时出错: {e}")
                continue
        
        if not batch_tensors:
            continue
        
        # 合并为批次张量
        batch_tensor = torch.stack(batch_tensors, dim=0)
        
        # 将图像移至设备
        batch_tensor = batch_tensor.to(device)
        
        # 预测
        with torch.no_grad():
            if next(model.parameters()).dtype == torch.float16:
                batch_tensor = batch_tensor.half()
            batch_features = model(batch_tensor)
        
        # 转换为numpy数组并添加到结果列表
        batch_features_np = batch_features.cpu().numpy()
        all_features.extend(batch_features_np)
        
        logging.info(f"已处理 {i+len(batch_paths)}/{len(image_paths)} 张图像")
    
    return np.array(all_features)

def compare_faces(model, image_path1, image_path2, device, config):
    """
    比较两张人脸图像的相似度
    
    Args:
        model: 加载好的模型
        image_path1: 第一张图像的路径
        image_path2: 第二张图像的路径
        device: 运行设备
        config: 配置对象
    
    Returns:
        相似度
    """
    # 获取两张图像的特征
    features = predict(model, [image_path1, image_path2], device, config)
    
    if len(features) != 2:
        logging.error("无法获取两张图像的特征")
        return None
    
    # 计算余弦相似度
    similarity = np.dot(features[0], features[1]) / (
        np.linalg.norm(features[0]) * np.linalg.norm(features[1])
    )
    
    return similarity

def save_features(features, image_paths, output_file):
    """
    保存特征向量到文件
    
    Args:
        features: 特征向量数组
        image_paths: 对应的图像路径列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w') as f:
        for path, feat in zip(image_paths, features):
            # 将特征向量转换为空格分隔的字符串
            feat_str = ' '.join([f'{x:.6f}' for x in feat])
            f.write(f"{path} {feat_str}\n")
    logging.info(f"特征向量已保存到: {output_file}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='预测使用train_v2.py训练的猫脸模型')
    
    # 添加配置文件参数（与train_v2.py相同）
    parser.add_argument('config', type=str, help='配置文件路径，如configs/ms1mv3_r50_onegpu.py')
    
    # 添加模型路径参数
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    
    # 预测功能
    parser.add_argument('--predict', type=str, nargs='+', help='预测多张图像，返回特征向量')
    
    # 比较功能
    parser.add_argument('--compare', nargs=2, type=str, help='比较两张图像的相似度')
    parser.add_argument('--threshold', type=float, default=0.5, help='相似度阈值，默认为0.5')
    
    # 批量处理参数
    parser.add_argument('--batch_size', type=int, default=8, help='批量处理大小，默认为8')
    
    # 输出参数
    parser.add_argument('--output', type=str, help='保存特征向量的输出文件路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 加载配置文件（与train_v2.py相同的方式）
    config = get_config(args.config)
    logging.info(f"加载配置文件: {args.config}")
    
    # 加载模型
    model, device = load_model(args.model_path, config)
    
    # 执行预测
    if args.predict:
        logging.info(f"预测图像数量: {len(args.predict)}")
        features = predict(model, args.predict, device, config, args.batch_size)
        
        if args.output:
            save_features(features, args.predict, args.output)
        else:
            for i, (path, feat) in enumerate(zip(args.predict, features)):
                logging.info(f"图像 {i+1}: {path}")
                logging.info(f"特征向量形状: {feat.shape}")
                logging.info(f"特征向量: {feat[:5]}...")  # 只显示前5个元素
        
    # 执行比较
    elif args.compare:
        image1, image2 = args.compare
        logging.info(f"比较图像: {image1} 和 {image2}")
        similarity = compare_faces(model, image1, image2, device, config)
        
        if similarity is not None:
            is_same = similarity > args.threshold
            logging.info(f"相似度: {similarity:.4f}, 是否为同一猫: {is_same}")
        else:
            logging.error("比较失败")
    
    else:
        logging.error("请指定 --predict 或 --compare 参数")
        parser.print_help()

if __name__ == '__main__':
    main()