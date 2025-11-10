#!/bin/bash

# Port Forwarding Script for Epic 8 Services
set -euo pipefail

NAMESPACE="epic8"

echo "=== Epic 8 Services Port Forwarding ==="

# Function to start port forwarding in background
start_port_forward() {
    local service=$1
    local local_port=$2
    local remote_port=$3

    echo "Starting port forward for $service: localhost:$local_port -> $service:$remote_port"
    kubectl port-forward -n "${NAMESPACE}" "svc/$service" "$local_port:$remote_port" &
    local pid=$!
    echo "$pid" > "/tmp/pf-$service.pid"
}

# Function to stop all port forwards
stop_port_forwards() {
    echo "Stopping all port forwards..."
    for pidfile in /tmp/pf-*.pid; do
        if [ -f "$pidfile" ]; then
            local pid=$(cat "$pidfile")
            kill "$pid" 2>/dev/null || true
            rm "$pidfile"
        fi
    done
}

# Trap to cleanup on exit
trap stop_port_forwards EXIT

# Check if services exist and start port forwarding
services_config=(
    "epic8-api-gateway 8080 8080"
    "epic8-query-analyzer 8082 8082"
    "epic8-generator 8081 8081"
    "epic8-retriever 8083 8083"
    "epic8-cache 8084 8084"
    "epic8-analytics 8085 8085"
)

for config in "${services_config[@]}"; do
    read -r service local_port remote_port <<< "$config"

    if kubectl get service "$service" -n "${NAMESPACE}" >/dev/null 2>&1; then
        start_port_forward "$service" "$local_port" "$remote_port"
    else
        echo "Service $service not found, skipping..."
    fi
done

echo -e "\nPort forwarding started. Access services at:"
echo "- API Gateway: http://localhost:8080"
echo "- Query Analyzer: http://localhost:8082"
echo "- Generator: http://localhost:8081"
echo "- Retriever: http://localhost:8083"
echo "- Cache: http://localhost:8084"
echo "- Analytics: http://localhost:8085"

echo -e "\nPress Ctrl+C to stop all port forwards..."
wait
