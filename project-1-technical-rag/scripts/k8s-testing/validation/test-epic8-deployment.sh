#!/bin/bash

# Epic 8 Deployment Test Script
set -euo pipefail

NAMESPACE="epic8"

echo "=== Epic 8 Deployment Test ==="

# Check if namespace exists
if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    echo "Error: Namespace ${NAMESPACE} does not exist"
    exit 1
fi

echo "Testing deployments in namespace: ${NAMESPACE}"

# Check deployments
echo -e "\nDeployments:"
kubectl get deployments -n "${NAMESPACE}"

# Check services
echo -e "\nServices:"
kubectl get services -n "${NAMESPACE}"

# Check pods
echo -e "\nPods:"
kubectl get pods -n "${NAMESPACE}"

# Check ingress
echo -e "\nIngress:"
kubectl get ingress -n "${NAMESPACE}" 2>/dev/null || echo "No ingress found"

# Test service connectivity
echo -e "\nTesting service connectivity..."

services=$(kubectl get services -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}')
for service in $services; do
    if [ "$service" != "kubernetes" ]; then
        echo "Testing service: $service"
        kubectl run test-pod-$service --image=curlimages/curl --rm -i --restart=Never -n "${NAMESPACE}" -- \
            curl -s -o /dev/null -w "%{http_code}" "http://$service" || echo "Connection test failed for $service"
    fi
done

echo -e "\n=== Deployment Test Complete ==="
