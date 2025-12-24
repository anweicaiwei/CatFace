import os
import pickle
import numpy as np
import argparse
import mxnet as mx
from mxnet import ndarray as nd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_lfw_pairs(pairs_file):
    """
    解析 LFW 数据集的 pairs.txt 文件
    
    Args:
        pairs_file (str): pairs.txt 文件路径
        
    Returns:
        tuple: (same_pairs, different_pairs)
            same_pairs: 同一人的图像对列表，每个元素为 (name, idx1, idx2)
            different_pairs: 不同人的图像对列表，每个元素为 (name1, idx1, name2, idx2)
    """
    same_pairs = []
    different_pairs = []
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
    
    # 第一行是数据集参数，通常是 (number_of_pairs, number_of_pairs_per_class)
    # 跳过前几行，直到找到实际的图像对
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # 尝试解析数字行（数据集参数）
        try:
            int(line.split()[0])
            i += 1
            continue
        except ValueError:
            pass
        
        parts = line.split()
        if len(parts) == 3:
            # 同一人的图像对: Name idx1 idx2
            name, idx1, idx2 = parts[0], int(parts[1]), int(parts[2])
            same_pairs.append((name, idx1, idx2))
        elif len(parts) == 4:
            # 不同人的图像对: Name1 idx1 Name2 idx2
            name1, idx1, name2, idx2 = parts[0], int(parts[1]), parts[2], int(parts[3])
            different_pairs.append((name1, idx1, name2, idx2))
        i += 1
    
    return same_pairs, different_pairs


def parse_cfp_fp_pairs(pairs_file):
    """
    解析 CFP-FP 数据集的 pairs.txt 文件
    CFP-FP 数据集的 pairs.txt 格式可能与 LFW 类似，但图像路径结构不同
    
    Args:
        pairs_file (str): pairs.txt 文件路径
        
    Returns:
        tuple: (same_pairs, different_pairs)
            same_pairs: 同一人的图像对列表，每个元素为 (name, idx1, idx2)
            different_pairs: 不同人的图像对列表，每个元素为 (name1, idx1, name2, idx2)
    """
    # CFP-FP 的 pairs.txt 格式通常与 LFW 相同，所以这里可以复用 LFW 的解析函数
    # 如果实际格式不同，可以在这里修改
    return parse_lfw_pairs(pairs_file)


def parse_agedb_30_pairs(pairs_file):
    """
    解析 AgeDB-30 数据集的 pairs.txt 文件
    AgeDB-30 数据集的 pairs.txt 格式可能与 LFW 类似，但图像命名规则不同
    
    Args:
        pairs_file (str): pairs.txt 文件路径
        
    Returns:
        tuple: (same_pairs, different_pairs)
            same_pairs: 同一人的图像对列表，每个元素为 (name, idx1, idx2)
            different_pairs: 不同人的图像对列表，每个元素为 (name1, idx1, name2, idx2)
    """
    # AgeDB-30 的 pairs.txt 格式通常与 LFW 相同，所以这里可以复用 LFW 的解析函数
    # 如果实际格式不同，可以在这里修改
    return parse_lfw_pairs(pairs_file)

def load_image(image_path):
    """
    加载图像并转换为二进制数据
    使用MXNet进行图像加载和编码，以确保与verification.py中的解码兼容
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        bytes: 图像的二进制数据
    """
    try:
        # 使用MXNet加载图像
        img = mx.image.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        # 检测文件扩展名
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.png':
            # 将图像转换为 PNG 格式的二进制数据
            _, buffer = mx.image.imencode('.png', img)
        else:
            # 默认转换为 JPEG 格式的二进制数据
            _, buffer = mx.image.imencode('.jpg', img, quality=95)  # 设置较高的JPEG质量
        return buffer.asnumpy().tobytes()
    except Exception as e:
        # 如果MXNet加载失败，回退到OpenCV
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        # 检测文件扩展名
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.png':
            # 将图像转换为 PNG 格式的二进制数据
            _, buffer = cv2.imencode('.png', img)
        else:
            # 默认转换为 JPEG 格式的二进制数据
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()

def get_image_path(dataset_name, dataset_root, name, idx):
    """
    根据数据集名称和参数构建图像文件路径
    
    Args:
        dataset_name (str): 数据集名称
        dataset_root (str): 数据集根目录
        name (str): 图像名称或标识
        idx (int): 图像索引
        
    Returns:
        str: 完整的图像文件路径
    """
    if dataset_name == 'lfw':
        # LFW 格式: 人名/人名_xxxx.jpg
        # 尝试 PNG 和 JPEG 两种格式
        png_path = os.path.join(dataset_root, name, f"{name}_{idx:04d}.png")
        jpg_path = os.path.join(dataset_root, name, f"{name}_{idx:04d}.jpg")
        if os.path.exists(png_path):
            return png_path
        return jpg_path
    elif dataset_name == 'cfp_fp':
        # CFP-FP 格式: 通常在 Frontal/Face 子目录中，文件名格式为 000001_01.jpg
        # 尝试 PNG 和 JPEG 两种格式
        png_path = os.path.join(dataset_root, name, "frontal", f"{name}_{idx:04d}.png")
        jpg_path = os.path.join(dataset_root, name, "frontal", f"{name}_{idx:04d}.jpg")
        if os.path.exists(png_path):
            return png_path
        return jpg_path
    elif dataset_name == 'agedb_30':
        # AgeDB-30 格式: 图像直接存储在根目录，文件名格式为 00001.jpg
        # 尝试 PNG 和 JPEG 两种格式
        png_path = os.path.join(dataset_root, f"{idx:05d}.png")
        jpg_path = os.path.join(dataset_root, f"{idx:05d}.jpg")
        if os.path.exists(png_path):
            return png_path
        return jpg_path
    else:
        # 默认使用 LFW 格式，尝试 PNG 和 JPEG 两种格式
        png_path = os.path.join(dataset_root, name, f"{name}_{idx:04d}.png")
        jpg_path = os.path.join(dataset_root, name, f"{name}_{idx:04d}.jpg")
        if os.path.exists(png_path):
            return png_path
        return jpg_path


def process_image_pair(pair, dataset_name, dataset_root):
    """
    处理单个图像对，加载并返回图像二进制数据
    
    Args:
        pair (tuple): 图像对信息
        dataset_name (str): 数据集名称
        dataset_root (str): 数据集根目录
        
    Returns:
        tuple: (img1_bin, img2_bin, issame)
    """
    # 处理同一个人的情况
    if len(pair) == 3:
        name, idx1, idx2 = pair
        img_path1 = get_image_path(dataset_name, dataset_root, name, idx1)
        img_path2 = get_image_path(dataset_name, dataset_root, name, idx2)
        issame = True
    # 处理不同人的情况
    elif len(pair) == 4:
        name1, idx1, name2, idx2 = pair
        img_path1 = get_image_path(dataset_name, dataset_root, name1, idx1)
        img_path2 = get_image_path(dataset_name, dataset_root, name2, idx2)
        issame = False
    else:
        raise ValueError(f"无效的pair格式: {pair}")
    
    # 加载图像
    img1_bin = load_image(img_path1)
    img2_bin = load_image(img_path2)
    
    return img1_bin, img2_bin, issame

def create_bin_file(dataset_name, dataset_root, pairs_file, output_bin):
    """
    创建验证数据集的 bin 文件
    
    Args:
        dataset_name (str): 数据集名称 (lfw, cfp_fp, agedb_30)
        dataset_root (str): 数据集根目录
        pairs_file (str): pairs.txt 文件路径
        output_bin (str): 输出的 bin 文件路径
    """
    print(f"正在处理 {dataset_name} 数据集...")
    
    # 根据数据集名称选择对应的解析函数
    if dataset_name == 'lfw':
        parse_func = parse_lfw_pairs
    elif dataset_name == 'cfp_fp':
        parse_func = parse_cfp_fp_pairs
    elif dataset_name == 'agedb_30':
        parse_func = parse_agedb_30_pairs
    else:
        raise ValueError(f"不支持的数据集名称: {dataset_name}")
    
    # 解析 pairs.txt 文件
    same_pairs, different_pairs = parse_func(pairs_file)
    
    # 合并同一人和不同人的图像对
    all_pairs = same_pairs + different_pairs
    
    # 准备图像二进制数据和 issame 列表
    bins = []
    issame_list = []
    
    print(f"开始处理 {len(all_pairs)} 个图像对...")
    
    # 使用多线程加速图像加载和处理
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务
        futures = []
        for pair in all_pairs:
            futures.append(executor.submit(process_image_pair, pair, dataset_name, dataset_root))
        
        # 收集结果，使用tqdm显示进度条
        with tqdm(total=len(all_pairs), desc="Processing image pairs", unit="pair") as pbar:
            for i, future in enumerate(as_completed(futures)):
                try:
                    img1_bin, img2_bin, issame = future.result()
                    bins.append(img1_bin)
                    bins.append(img2_bin)
                    issame_list.append(issame)
                except Exception as e:
                    # 捕获处理单个图像对时的异常
                    pair = all_pairs[i]  # 获取原始图像对信息用于错误提示
                    if len(pair) == 3:
                        print(f"\n处理图像对 {pair[0]} {pair[1]} {pair[2]} 时出错: {e}")
                    else:
                        print(f"\n处理图像对 {pair[0]} {pair[1]} {pair[2]} {pair[3]} 时出错: {e}")
                finally:
                    pbar.update(1)  # 更新进度条
    
    # 保存为 bin 文件
    print(f"\n保存 bin 文件到 {output_bin}...")
    with open(output_bin, 'wb') as f:
        pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"{dataset_name} 数据集处理完成，共 {len(issame_list)} 对图像")
    print(f"图像数量: {len(bins)}")
    # print("\n提示: 为了提高加载速度，可以考虑:")
    # print("1. 在verification.py的load_bin函数中使用批量解码操作")
    # print("2. 使用多线程或GPU加速图像解码和转换")
    # print("3. 考虑将图像预解码并保存为numpy数组格式")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将人脸验证数据集转换为 bin 文件格式')
    
    # 添加命令行参数
    parser.add_argument('--dataset-name', type=str, required=True, choices=['lfw', 'cfp_fp', 'agedb_30'],
                        help='数据集名称：lfw, cfp_fp, 或 agedb_30')
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='数据集图像文件的根目录')
    parser.add_argument('--pairs-file', type=str, required=True,
                        help='pairs.txt 文件的路径')
    parser.add_argument('--output-bin', type=str, required=True,
                        help='输出的 bin 文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_bin)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理指定的数据集
    create_bin_file(args.dataset_name, args.dataset_root, args.pairs_file, args.output_bin)
    
    print(f"{args.dataset_name} 数据集处理完成！")

if __name__ == "__main__":
    main()