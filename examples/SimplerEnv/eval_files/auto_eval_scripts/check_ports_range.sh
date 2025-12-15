#!/bin/bash
# 这个脚本提前检查端口是否可以用，为并行化测试提供前提基础
# 检查端口范围
start_port=5400
end_port=5500

echo "检查端口范围 ${start_port}-${end_port} 是否可用..."

for port in $(seq $start_port $end_port); do
  # 检查端口是否被占用
  if lsof -iTCP:$port -sTCP:LISTEN > /dev/null 2>&1; then
    echo "端口 $port 已被占用"
  else
    echo "端口 $port 可用"
  fi
done

echo "检查完成！"