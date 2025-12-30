#!/bin/bash

# 指定日志文件的根目录
log_dir="results/Checkpoints/1226_libero4in1_qwen3oft"

# 遍历指定目录下的所有日志文件
last_Folder=""
find "$log_dir" -type f -name "*.log" | while read -r log_file; do
    # 提取日志文件中最后一个 "Total success rate" 的值
    success_rate=$(grep "INFO     | >> Total success rate:" "$log_file" | tail -n 1)
    
    # 如果找到匹配的内容，则输出日志文件路径和对应的成功率
    if [ -n "$success_rate" ]; then
        echo "Folder: $(basename "$(dirname "$log_file")")"
        echo "File: $(basename "$log_file")"
        echo "$success_rate"
        echo
    fi
done