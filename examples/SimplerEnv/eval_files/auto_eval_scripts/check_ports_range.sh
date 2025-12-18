#!/bin/bash
# Check port availability ahead of time to prepare for parallel testing
# Target port range
start_port=5400
end_port=5500

echo "Checking whether ports ${start_port}-${end_port} are available..."

for port in $(seq $start_port $end_port); do
  # Determine if the port is already in use
  if lsof -iTCP:$port -sTCP:LISTEN > /dev/null 2>&1; then
    echo "Port $port is in use"
  else
    echo "Port $port is available"
  fi
done

echo "Port check complete!"