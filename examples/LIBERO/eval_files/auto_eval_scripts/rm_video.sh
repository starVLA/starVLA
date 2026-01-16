#!/bin/bash

# 删除所有 results/Checkpoints/*/videos/libero_* 目录

# 设置目标目录
TARGET_DIR="results/Checkpoints"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录 '$TARGET_DIR' 不存在"
    exit 1
fi

echo "正在查找 $TARGET_DIR/*/videos/libero_* 目录..."

# 方法1: 使用更直接的查找方式
MATCHING_DIRS=()
for checkpoints_dir in "$TARGET_DIR"/*/; do
    videos_dir="${checkpoints_dir}videos/"
    if [ -d "$videos_dir" ]; then
        for libero_dir in "$videos_dir"libero_*/; do
            if [ -d "$libero_dir" ]; then
                # 移除结尾的斜杠
                dir="${libero_dir%/}"
                MATCHING_DIRS+=("$dir")
                echo "找到: $dir"
            fi
        done
    fi
done

# 如果没有找到目录
if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
    echo "在 $TARGET_DIR/*/videos/ 下没有找到 libero_* 目录"
    exit 0
fi

echo ""
echo "总共找到 ${#MATCHING_DIRS[@]} 个目录"
echo ""

# 确认是否要删除
read -p "确认要删除以上所有目录吗？(y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始删除..."
    for dir in "${MATCHING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            echo "已删除: $dir"
        fi
    done
    echo "删除完成!"
else
    echo "取消删除操作"
    exit 0
fi