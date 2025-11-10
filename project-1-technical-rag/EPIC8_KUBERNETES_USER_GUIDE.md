# Epic 8 Kubernetes User Guide

## Overview

Epic 8 is a cloud-native RAG (Retrieval-Augmented Generation) platform deployed on Kubernetes, featuring intelligent query analysis, multi-model generation, and comprehensive caching.

**Current Status**: 4/6 services operational (API Gateway, Generator, Query Analyzer, Cache)

---

## Quick Start

### Prerequisites

- Kubernetes cluster (Kind, EKS, GKE, or AKS)
- `kubectl` configured
- Docker installed
- 8GB+ RAM available

### Deploy the System

```bash
# 1. Build all service images
./scripts/deployment/build-services.sh build all

# 2. Load images into Kind cluster
./scripts/deployment/load-images-kind.sh load

# 3. Apply Kubernetes manifests
kubectl apply -f k8s/namespaces/
kubectl apply -f k8s/config/
kubectl apply -f k8s/storage/
kubectl apply -f k8s/services/

# 4. Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=epic8 -n epic8-dev --timeout=300s
```

### Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n epic8-dev

# Expected output: All pods 1/1 Running with 0 restarts
# - api-gateway
# - cache
# - generator
# - query-analyzer
```

---

## Using the System

### Making RAG Queries

#### Via Port Forward

```bash
# Forward API Gateway port
kubectl port-forward -n epic8-dev svc/api-gateway 8080:8080

# Make a query (in another terminal)
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "strategy": "balanced"
  }'
```

#### Query Strategies

- **`cost_optimized`**: Uses cheapest models (ollama/llama3.2:3b)
- **`balanced`**: Mix of quality and cost (default)
- **`quality_first`**: Best models (GPT-4, Mistral Large)

### Checking System Health

```bash
# Overall system status
kubectl get pods -n epic8-dev

# Individual service health
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/status

# Service-specific health checks
kubectl exec -n epic8-dev <pod-name> -- curl http://localhost:<port>/health/ready
```

### Viewing Logs

```bash
# API Gateway logs
kubectl logs -f deployment/api-gateway -n epic8-dev

# Generator logs
kubectl logs -f deployment/generator -n epic8-dev

# Query Analyzer logs
kubectl logs -f deployment/query-analyzer -n epic8-dev

# Cache logs
kubectl logs -f deployment/cache -n epic8-dev

# All services (combined)
kubectl logs -f -l app=epic8 -n epic8-dev --all-containers=true
```

---

## Monitoring

### Prometheus Metrics

Each service exposes Prometheus metrics on port 9090:

```bash
# Port forward to metrics endpoint
kubectl port-forward -n epic8-dev svc/api-gateway 9090:9090

# View metrics
curl http://localhost:9090/metrics
```

**Key Metrics:**
- `gateway_requests_total` - Total API requests
- `gateway_request_duration_seconds` - Request latency
- `gateway_service_health` - Service health status
- `gateway_connected_services` - Number of healthy services

### Resource Usage

```bash
# CPU and memory usage
kubectl top pods -n epic8-dev

# Resource quotas
kubectl describe resourcequota -n epic8-dev
```

---

## Common Operations

### Scaling Services

```bash
# Scale a service
kubectl scale deployment/generator --replicas=2 -n epic8-dev

# Note: Current cluster quota supports 1 replica per service
```

### Updating Services

```bash
# 1. Build new image
./scripts/deployment/build-services.sh build generator

# 2. Load into Kind
kind load docker-image epic8/generator:latest --name epic8-testing

# 3. Restart deployment
kubectl rollout restart deployment/generator -n epic8-dev

# 4. Monitor rollout
kubectl rollout status deployment/generator -n epic8-dev
```

### Viewing Configuration

```bash
# View ConfigMaps
kubectl get configmap -n epic8-dev
kubectl describe configmap epic8-common-config -n epic8-dev

# View Secrets (base64 encoded)
kubectl get secrets -n epic8-dev
```

### Restarting Services

```bash
# Restart a specific service
kubectl rollout restart deployment/api-gateway -n epic8-dev

# Restart all services
kubectl rollout restart deployment -n epic8-dev

# Delete a pod (will be recreated automatically)
kubectl delete pod <pod-name> -n epic8-dev
```

---

## Troubleshooting

### Pod Not Starting

**Symptoms**: Pod stuck in `Pending`, `CrashLoopBackOff`, or `ImagePullBackOff`

**Diagnosis:**
```bash
# Check pod status and events
kubectl describe pod <pod-name> -n epic8-dev

# Check logs
kubectl logs <pod-name> -n epic8-dev
kubectl logs <pod-name> -n epic8-dev --previous  # Previous crash logs
```

**Common Causes:**
1. **Resource Quota Exceeded**: Scale down other services
2. **Image Not Found**: Rebuild and reload image
3. **Configuration Error**: Check ConfigMaps and Secrets
4. **Health Check Failure**: Check `/health/startup`, `/health/ready`, `/health/live` endpoints

### ValidationError: Port Parsing

**Symptoms**: Services crash with "unable to parse string as an integer" for port field

**Cause**: Kubernetes sets environment variables like `GENERATOR_PORT=tcp://10.96.85.6:8081`

**Solution**: Field validators are implemented in `config.py` to handle this. If you see this error:
```bash
# Verify the field validator is present
grep -A 20 "@field_validator('port'" services/generator/generator_app/core/config.py
```

### Health Check Failures

**Symptoms**: Pod shows 0/1 Ready, health probe failures in logs

**Diagnosis:**
```bash
# Check which probe is failing
kubectl describe pod <pod-name> -n epic8-dev | grep -A 10 "Events:"

# Test endpoint manually
kubectl exec -n epic8-dev <pod-name> -- curl http://localhost:<port>/health/startup
```

**Common Endpoints:**
- API Gateway: Port 8080
- Generator: Port 8081
- Query Analyzer: Port 8082
- Cache: Port 8084

### Service Unavailable (503)

**Symptoms**: API Gateway returns 503 or "Service not initialized"

**Diagnosis:**
```bash
# Check if dependent services are healthy
kubectl get pods -n epic8-dev

# Check API Gateway logs for connection errors
kubectl logs deployment/api-gateway -n epic8-dev | grep -i error
```

**Solution:**
1. Ensure all services are Running and Ready (1/1)
2. Check service discovery is working
3. Verify network policies allow communication

### Out of Resources

**Symptoms**: Pods stuck in Pending with "exceeded quota" error

**Solution:**
```bash
# Check current resource usage
kubectl describe resourcequota epic8-dev-quota -n epic8-dev

# Scale down non-essential services
kubectl scale deployment/analytics --replicas=0 -n epic8-dev
kubectl scale deployment/retriever --replicas=0 -n epic8-dev
```

---

## Configuration

### Environment Variables

Services read configuration from:
1. **ConfigMaps**: `epic8-common-config`, `<service>-config`
2. **Secrets**: `epic8-secrets`, `llm-api-keys`
3. **Environment Variables**: Set via deployment manifests

### Key Configuration Files

- `config/default.yaml` - Default application config
- `k8s/config/` - Kubernetes ConfigMaps and Secrets
- `services/<service>/config.yaml` - Service-specific config

### LLM API Keys

To use external LLM providers:

```bash
# Edit the secret
kubectl edit secret llm-api-keys -n epic8-dev

# Add base64-encoded keys:
# OPENAI_API_KEY: <base64-encoded-key>
# MISTRAL_API_KEY: <base64-encoded-key>

# Restart services to pick up changes
kubectl rollout restart deployment/generator -n epic8-dev
```

---

## Performance Tuning

### Resource Limits

Current settings per service:
- **CPU**: 500m request, 1 limit
- **Memory**: 1Gi request, 2Gi limit

To adjust:
```bash
# Edit deployment
kubectl edit deployment generator -n epic8-dev

# Update resources section
resources:
  requests:
    cpu: "1"
    memory: "2Gi"
  limits:
    cpu: "2"
    memory: "4Gi"
```

### Cache Configuration

Redis cache configuration in `cache-config` ConfigMap:
- Max memory: 512MB
- Eviction policy: allkeys-lru
- TTL: 3600s (1 hour)

---

## Cleanup

### Remove Epic 8 Deployment

```bash
# Delete all resources in namespace
kubectl delete namespace epic8-dev

# Verify removal
kubectl get namespaces | grep epic8
```

### Clean Docker Images

```bash
# Remove local images
docker rmi epic8/api-gateway:latest
docker rmi epic8/generator:latest
docker rmi epic8/query-analyzer:latest
docker rmi epic8/cache:latest

# Clean Kind cluster images
kind load docker-image --help  # No direct cleanup; recreate cluster if needed
```

---

## Support and References

### Log Locations

- **Application Logs**: `/app/logs/` (inside containers)
- **Kubernetes Logs**: `kubectl logs` command
- **Persistent Logs**: Not configured (future enhancement)

### Health Check Endpoints

All services expose:
- `/health` - Basic health check
- `/health/live` - Liveness probe (K8s uses this)
- `/health/ready` - Readiness probe (K8s uses this)
- `/health/startup` - Startup probe (K8s uses this)

### Service Ports

| Service | HTTP Port | Metrics Port |
|---------|-----------|--------------|
| API Gateway | 8080 | 9090 |
| Generator | 8081 | 9090 |
| Query Analyzer | 8082 | 9090 |
| Retriever | 8083 | 9090 |
| Cache | 8084 | 9090 |
| Analytics | 8085 | 9090 |

---

## Next Steps

1. **Enable Analytics and Retriever**: Scale up remaining services once resource quota increased
2. **Add Monitoring**: Deploy Prometheus and Grafana for full observability
3. **Configure Ingress**: Set up Ingress controller for external access
4. **Enable TLS**: Add cert-manager and TLS certificates
5. **Set up CI/CD**: Automate build and deployment pipeline

For architecture details, see `EPIC8_KUBERNETES_ARCHITECTURE.md`