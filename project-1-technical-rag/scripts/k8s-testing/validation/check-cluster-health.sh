#!/bin/bash

# Cluster Health Check Script
set -euo pipefail

echo "=== Kubernetes Cluster Health Check ==="

# Check cluster info
echo "Cluster info:"
kubectl cluster-info

echo -e "\nNode status:"
kubectl get nodes -o wide

echo -e "\nNamespace status:"
kubectl get namespaces

echo -e "\nSystem pods status:"
kubectl get pods -n kube-system

echo -e "\nIngress controller status:"
kubectl get pods -n ingress-nginx 2>/dev/null || echo "Ingress controller not found"

echo -e "\nMonitoring status:"
kubectl get pods -n monitoring 2>/dev/null || echo "Monitoring not found"

echo -e "\nEpic 8 namespace status:"
kubectl get all -n epic8 2>/dev/null || echo "Epic 8 namespace empty"

echo -e "\n=== Health Check Complete ==="
