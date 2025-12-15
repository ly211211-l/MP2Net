# check_missing_airplane_sequences.py
import json
import os
from pathlib import Path
from collections import defaultdict

# 路径配置
JSON_PATH = "./dataset/SatMTB/instances_SatMTB_test_3cate.json"
IMG_DIR = "./dataset/SatMTB/test/img/airplane"

def extract_sequences_from_json(json_path, category='airplane'):
    """从JSON文件中提取指定类别的所有序列ID"""
    print(f"正在读取JSON文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sequences = set()
    category_prefix = f"img/{category}/"
    
    for image in data['images']:
        file_name = image['file_name']
        if file_name.startswith(category_prefix):
            # 提取序列ID，例如: img/airplane/10/000001.png -> 10
            parts = file_name.split('/')
            if len(parts) >= 3:
                seq_id = parts[2]  # 序列ID
                sequences.add(seq_id)
    
    return sorted(sequences, key=lambda x: int(x) if x.isdigit() else x)

def get_actual_sequences(img_dir):
    """获取实际目录中存在的序列ID"""
    print(f"正在检查实际目录: {img_dir}")
    
    if not os.path.exists(img_dir):
        print(f"警告: 目录不存在: {img_dir}")
        return []
    
    sequences = []
    for item in os.listdir(img_dir):
        item_path = os.path.join(img_dir, item)
        if os.path.isdir(item_path):
            sequences.append(item)
    
    return sorted(sequences, key=lambda x: int(x) if x.isdigit() else x)

def check_sequence_files(json_path, img_dir, seq_id, category='airplane'):
    """检查某个序列的文件是否完整"""
    # 从JSON中获取该序列的所有文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    json_files = set()
    category_prefix = f"img/{category}/{seq_id}/"
    
    for image in data['images']:
        file_name = image['file_name']
        if file_name.startswith(category_prefix):
            # 提取文件名，例如: img/airplane/10/000001.png -> 000001.png
            filename = os.path.basename(file_name)
            json_files.add(filename)
    
    # 检查实际目录中的文件
    seq_dir = os.path.join(img_dir, seq_id)
    actual_files = set()
    if os.path.exists(seq_dir):
        actual_files = set(os.listdir(seq_dir))
    
    missing_files = json_files - actual_files
    extra_files = actual_files - json_files
    
    return {
        'json_count': len(json_files),
        'actual_count': len(actual_files),
        'missing_files': sorted(missing_files),
        'extra_files': sorted(extra_files)
    }

def main():
    print("=" * 60)
    print("检查Airplane序列缺失情况")
    print("=" * 60)
    
    # 1. 从JSON提取序列
    json_sequences = extract_sequences_from_json(JSON_PATH, 'airplane')
    print(f"\nJSON中记录的序列数: {len(json_sequences)}")
    print(f"JSON中的序列ID: {json_sequences}")
    
    # 2. 获取实际目录中的序列
    actual_sequences = get_actual_sequences(IMG_DIR)
    print(f"\n实际目录中的序列数: {len(actual_sequences)}")
    print(f"实际目录中的序列ID: {actual_sequences}")
    
    # 3. 对比找出缺失的序列
    json_set = set(json_sequences)
    actual_set = set(actual_sequences)
    
    missing_sequences = json_set - actual_set
    extra_sequences = actual_set - json_set
    
    print("\n" + "=" * 60)
    print("对比结果:")
    print("=" * 60)
    
    if missing_sequences:
        print(f"\n❌ 缺失的序列 ({len(missing_sequences)}个):")
        for seq_id in sorted(missing_sequences, key=lambda x: int(x) if x.isdigit() else x):
            print(f"  - 序列 {seq_id}")
            # 检查该序列在JSON中有多少文件
            file_info = check_sequence_files(JSON_PATH, IMG_DIR, seq_id, 'airplane')
            print(f"    JSON中应有 {file_info['json_count']} 个文件，实际目录中 {file_info['actual_count']} 个")
    else:
        print("\n✅ 没有缺失的序列")
    
    if extra_sequences:
        print(f"\n⚠️  实际目录中存在但JSON中没有的序列 ({len(extra_sequences)}个):")
        for seq_id in sorted(extra_sequences, key=lambda x: int(x) if x.isdigit() else x):
            print(f"  - 序列 {seq_id}")
    
    # 4. 详细检查每个序列的文件完整性
    print("\n" + "=" * 60)
    print("详细文件检查（仅检查缺失的序列）:")
    print("=" * 60)
    
    if missing_sequences:
        for seq_id in sorted(missing_sequences, key=lambda x: int(x) if x.isdigit() else x):
            file_info = check_sequence_files(JSON_PATH, IMG_DIR, seq_id, 'airplane')
            print(f"\n序列 {seq_id}:")
            print(f"  JSON中文件数: {file_info['json_count']}")
            print(f"  实际文件数: {file_info['actual_count']}")
            if file_info['missing_files']:
                print(f"  缺失的文件数: {len(file_info['missing_files'])}")
                if len(file_info['missing_files']) <= 10:
                    print(f"  缺失的文件: {file_info['missing_files']}")
                else:
                    print(f"  缺失的文件（前10个）: {file_info['missing_files'][:10]}...")
    
    print("\n" + "=" * 60)
    print("检查完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()



