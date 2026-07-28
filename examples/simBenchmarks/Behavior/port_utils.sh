#!/bin/bash

# Utility function to find an available port
# Requires: used_ports array to be declared in the sourcing script before sourcing this file
# Usage: port=$(find_available_port $base_port)
find_available_port() {
  local base_port=$1
  local port=$base_port
  
  # First check our internal tracking
  while [[ " ${used_ports[@]} " =~ " ${port} " ]]; do
    port=$((port + 1))
  done
  
  # Then check system ports if tools are available
  if command -v netstat >/dev/null 2>&1; then
    while netstat -tuln 2>/dev/null | grep -q ":$port "; do
      port=$((port + 1))
    done
  elif command -v ss >/dev/null 2>&1; then
    while ss -tuln 2>/dev/null | grep -q ":$port "; do
      port=$((port + 1))
    done
  elif command -v lsof >/dev/null 2>&1; then
    while lsof -i :$port >/dev/null 2>&1; do
      port=$((port + 1))
    done
  else
    # Fallback: just increment port and hope for the best
    echo "⚠️ Warning: No port checking tools available, using port ${port}" >&2
  fi
  
  # Add to our tracking
  used_ports+=($port)
  echo $port
}

# Define a function to check if server is ready
wait_for_server() {
  local port=$1
  local max_attempts=30
  local attempt=0
  
  echo "⏳ Waiting for server on port ${port} to be ready..." >&2
  while [ $attempt -lt $max_attempts ]; do
    # Try different methods to check if port is in use
    local port_in_use=false
    
    # Method 1: Try to connect to the port (most reliable)
    if timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
      port_in_use=true
    # Method 2: Use system tools if available
    elif command -v netstat >/dev/null 2>&1; then
      if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        port_in_use=true
      fi
    elif command -v ss >/dev/null 2>&1; then
      if ss -tuln 2>/dev/null | grep -q ":$port "; then
        port_in_use=true
      fi
    elif command -v lsof >/dev/null 2>&1; then
      if lsof -i :$port >/dev/null 2>&1; then
        port_in_use=true
      fi
    fi
    
    if [ "$port_in_use" = true ]; then
      echo "✅ Server on port ${port} is ready" >&2
      return 0
    fi
    
    sleep 2
    attempt=$((attempt + 1))
  done
  
  echo "❌ Server on port ${port} failed to start after $((max_attempts * 2)) seconds" >&2
  return 1
}