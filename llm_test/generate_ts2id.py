# 专门生成 ts2id.json 的脚本

import os
import json
from datetime import datetime, timedelta

def generate_ts2id(data_dir):
    # ICEWS14 数据集的起点时间通常是 2014-01-01
    start_dt = datetime(2014, 1, 1)
    ts_set = set()

    # 遍历所有可能包含数据的文件
    files_to_check = ['train.txt', 'test.txt', 'valid.txt']
    
    for filename in files_to_check:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # 第4列（索引为3）是时间戳
                        ts = int(parts[3])
                        ts_set.add(ts)
            print(f"已处理 {filename}")

    # 生成映射字典 (反转格式：日期字符串为键，时间戳ID字符串为值)
    ts2id = {}
    for ts in sorted(ts_set):
        # 加上经过的小时数，计算出真实日期
        current_dt = start_dt + timedelta(hours=ts)
        # 将其格式化为 "YYYY-MM-DD" 的文本
        date_str = current_dt.strftime("%Y-%m-%d")
        
        # 注意：如果有多个时间戳对应同一天（比如同一天的不同小时），
        # 这种反转格式可能会导致后面的时间戳覆盖前面的。
        # 如果模型需要精确的时间戳映射，我们只保留第一个即可。
        if date_str not in ts2id:
            ts2id[date_str] = str(ts)

    # 保存为 json 文件
    out_path = os.path.join(data_dir, 'ts2id.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ts2id, f, ensure_ascii=False, indent=4)

    print(f"\n成功生成 {out_path}！")
    print(f"共映射了 {len(ts2id)} 个唯一的日期。")
    print(f"示例: 日期 '2014-01-01' 映射为 '{ts2id.get('2014-01-01')}'")

if __name__ == "__main__":
    # 使用绝对路径指向你的 ICEWS14 文件夹
    data_dir = r"d:\知识图谱推理\pythonCode\llm_test\data\ICEWS14"
    generate_ts2id(data_dir)