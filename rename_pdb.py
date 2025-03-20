import os
import re
import argparse

def rename_relaxed_pdb_files(root_dir, copy_to_dir=None):
    pattern = re.compile(r'^(.*)_.*_relaxed')
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pdb') and '_relaxed' in filename:
                match = pattern.match(filename)
                if match:
                    # 创建目标目录
                    target_dir = copy_to_dir if copy_to_dir else os.getcwd()
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # 源文件路径
                    src_path = os.path.join(dirpath, filename)
                    # 复制文件到目标目录
                    import shutil
                    dest_path = shutil.copy2(src_path, target_dir)
                    
                    # 在目标目录处理重命名
                    base_name = os.path.basename(dest_path)
                    new_name = f"{match.group(1)}.pdb"
                    old_path = os.path.join(target_dir, base_name)
                    new_path = os.path.join(target_dir, new_name)
                    
                    # 处理重名文件
                    counter = 1
                    while os.path.exists(new_path):
                        new_name = f"{match.group(1)}_{counter}.pdb"
                        new_path = os.path.join(dirpath, new_name)
                        counter += 1
                    
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='重命名包含relaxed的PDB文件')
    parser.add_argument('directory', help='需要处理的根目录路径')
    parser.add_argument('--copy-to', dest='copy_to', help='指定复制目标目录（默认为当前目录）')
    args = parser.parse_args()
    
    try:
        rename_relaxed_pdb_files(args.directory, args.copy_to)
        print("处理完成！")
    except Exception as e:
        print(f"发生错误: {str(e)}")