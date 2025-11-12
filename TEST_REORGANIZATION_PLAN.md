# Test Reorganization Plan - Detailed Implementation Strategy

## Root Cause Analysis: Why Tests Are Failing

### Problem 1: Wrong Dependencies in Wrong Places

**Scenario**: Unit test runner executing tests that require ML models

```python
# tests/unit/components/query_processors/analyzers/ml_views/test_semantic_complexity_view.py
import torch  # ❌ FAILS: ModuleNotFoundError: No module named 'torch'
from src.components.query_processors.analyzers.ml_models.model_manager import ModelManager

def test_semantic_analysis_with_ml():
    view = SemanticComplexityView()  # Tries to load Sentence-BERT model
    result = view.analyze("complex query")  # ❌ FAILS: torch not installed
```

**Why it fails**:
- Test runner: `pytest tests/unit/` 
- Environment: Minimal Python + core packages (no torch, no transformers)
- Expectation: Unit tests should use mocks, not real ML models
- Reality: Test loads actual ModelManager → tries to load torch models → crashes

**Impact**: ~40-60 tests in `tests/unit/components/query_processors/analyzers/ml_views/`

---

### Problem 2: External Services Required

**Scenario**: Component test expecting Ollama to be running

```python
# tests/component/test_modular_answer_generator.py
def _check_ollama_available() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200  # ❌ FAILS: Connection refused
    except:
        return False

def test_generate_with_ollama(self):
    if not _check_ollama_available():
        pytest.skip("Ollama not available")  # Test skipped or fails
```

**Why it fails**:
- Test runner: Expects Ollama service running on localhost:11434
- CI environment: No Ollama installed, no service running
- Local environment: Ollama may not be running
- Result: Test skipped or times out trying to connect

**Impact**: ~20-30 tests requiring Ollama in `tests/component/test_modular_answer_generator.py`

---

### Problem 3: Heavy Dependencies Loading

**Scenario**: Embedding test loading full models

```python
# tests/component/test_embeddings.py
from src.shared_utils.embeddings.generator import generate_embeddings

def test_embedding_performance():
    # This function loads sentence-transformers model (400MB+)
    embeddings = generate_embeddings(["text1", "text2"])  
    # ❌ FAILS: torch not installed OR takes 30+ seconds to load
```

**Why it fails/is problematic**:
- Model download: sentence-transformers/all-MiniLM-L6-v2 (~90MB download, ~400MB in memory)
- Dependencies: torch, transformers, sentence-transformers all required
- Performance: 10-30 seconds just to initialize model
- Memory: ~500MB-1GB RAM per test process

**Impact**: ~10-15 tests in `tests/component/test_embeddings.py`

---

## Solution: Three-Tier Test Strategy

### Tier 1: Pure Unit Tests (Fast, Always Run)
**Location**: `tests/unit/`
**Runtime**: <1 minute total
**Dependencies**: Python + core packages only (pytest, unittest, mock)
**Failures if miscategorized**: ModuleNotFoundError, ImportError

### Tier 2: Component Tests (Moderate, Regular Runs)
**Location**: `tests/component/`
**Runtime**: 2-5 minutes total
**Dependencies**: + lightweight libraries (pdfplumber, beautifulsoup4)
**Failures if miscategorized**: Slower tests, but still runnable

### Tier 3: Integration Tests (Slow, Selective Runs)
**Location**: `tests/integration/`
**Runtime**: 10-30 minutes total
**Dependencies**: + ML stack (torch, transformers, sentence-transformers)
**Failures if miscategorized**: Test suite times out, memory issues

---

## Detailed Implementation Plan

### Phase 1: Add Pytest Markers (Low Risk, Immediate Value)

**Objective**: Label tests without moving files, enable selective execution

**Implementation**:

1. **Update `pytest.ini`**:
```ini
[pytest]
markers =
    unit: Pure unit tests with mocks only (fast, no external deps)
    component: Component tests with lightweight deps (moderate speed)
    integration: Integration tests with real deps (slow, heavy)
    requires_ml: Requires ML dependencies (torch, transformers, sentence-transformers)
    requires_ollama: Requires Ollama service running on localhost:11434
    requires_postgres: Requires PostgreSQL database
    requires_redis: Requires Redis cache
    slow: Tests taking >5 seconds
```

2. **Tag Miscategorized Unit Tests**:
```python
# tests/unit/test_platform_orchestrator_phase2.py
import pytest

@pytest.mark.integration  # Tag as integration, not unit
@pytest.mark.component    # Uses real components
class TestPlatformOrchestratorPhase2:
    def test_phase2_features(self):
        # ... existing test code
```

```python
# tests/unit/components/query_processors/analyzers/ml_views/test_semantic_complexity_view.py
import pytest

@pytest.mark.integration  # Tag as integration
@pytest.mark.requires_ml   # Needs torch
class TestSemanticComplexityView:
    def test_semantic_analysis(self):
        # ... existing test code
```

3. **Tag Component Tests**:
```python
# tests/component/test_modular_answer_generator.py
import pytest

@pytest.mark.integration     # Actually integration test
@pytest.mark.requires_ollama  # Needs Ollama service
class TestModularAnswerGenerator:
    def test_generate_with_ollama(self):
        # ... existing test code
```

```python
# tests/component/test_embeddings.py
import pytest

@pytest.mark.integration  # Actually integration test
@pytest.mark.requires_ml   # Needs torch, sentence-transformers
@pytest.mark.slow          # Takes 10+ seconds
class TestEmbeddingGeneration:
    def test_generate_embeddings(self):
        # ... existing test code
```

**Benefits**:
- ✅ No files moved (zero risk of breaking imports)
- ✅ Immediate selective execution
- ✅ Documents what each test actually needs
- ✅ CI/CD can use markers to run appropriate tests

**Execution Commands**:
```bash
# Fast unit tests only (no ML, no services)
pytest -m "unit and not requires_ml and not requires_ollama" --maxfail=50

# Component tests (lightweight deps okay)
pytest tests/component -m "not requires_ml and not requires_ollama" --maxfail=50

# Integration tests with ML (requires torch installed)
pytest -m "integration and requires_ml" --maxfail=50

# Integration tests with Ollama (requires service running)
pytest -m "integration and requires_ollama" --maxfail=20
```

**Expected Impact**:
- **Before**: 537 failures when running all tests with minimal environment
- **After Phase 1**: 
  - Unit tests only: ~200-300 failures (skip ML/service tests)
  - Can isolate actual code issues from dependency issues

---

### Phase 2: Create Dependency Files (Medium Risk)

**Objective**: Document exact dependencies for each test tier

**Implementation**:

1. **Create `requirements-test-unit.txt`**:
```txt
# Minimal dependencies for unit tests
pytest==9.0.0
pytest-cov==7.0.0
pytest-mock==3.14.0
pydantic==2.6.0
pyyaml==6.0.1
```

2. **Create `requirements-test-component.txt`**:
```txt
# Component test dependencies (extends unit)
-r requirements-test-unit.txt

# Lightweight parsing/processing
pdfplumber==0.11.0
beautifulsoup4==4.12.3
lxml==5.1.0

# Basic NLP (no ML models)
nltk==3.8.1
spacy==3.7.2  # Base only, no models
```

3. **Create `requirements-test-integration.txt`**:
```txt
# Full integration test dependencies
-r requirements-test-component.txt

# ML Stack
torch==2.2.0
transformers==4.38.0
sentence-transformers==2.5.0

# Additional ML tools
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.4.0

# API clients
requests==2.31.0
httpx==0.27.0

# Database clients (optional)
psycopg2-binary==2.9.9  # PostgreSQL
redis==5.0.1            # Redis
```

**Installation Strategy**:
```bash
# Developer working on unit tests only
pip install -r requirements-test-unit.txt

# Developer working on components
pip install -r requirements-test-component.txt

# Full test suite / CI integration tier
pip install -r requirements-test-integration.txt

# Plus services need to be running:
# - Ollama: ollama serve (provides localhost:11434)
# - PostgreSQL: docker run -p 5432:5432 postgres
# - Redis: docker run -p 6379:6379 redis
```

**Expected Impact**:
- Clear dependency boundaries
- Faster developer setup (install only what's needed)
- Smaller CI containers for fast tests

---

### Phase 3: Reorganize Files (Higher Risk, Maximum Clarity)

**Objective**: Physical file organization matches test categories

**Implementation**:

1. **Move Integration Tests from `tests/unit/`**:

```bash
# Create integration directories
mkdir -p tests/integration/platform
mkdir -p tests/integration/ml_infrastructure

# Move miscategorized files
git mv tests/unit/test_platform_orchestrator_phase2.py \
       tests/integration/platform/test_platform_orchestrator_phase2.py

git mv tests/unit/test_fusion_rerankers_comprehensive.py \
       tests/integration/retrieval/test_fusion_rerankers_comprehensive.py

# Move ML view tests
git mv tests/unit/components/query_processors/analyzers/ml_views/*.py \
       tests/integration/ml_infrastructure/
```

**Files to Move** (18 files total):

**From `tests/unit/` to `tests/integration/`**:
- `test_platform_orchestrator_phase2.py` → `integration/platform/`
- `test_fusion_rerankers_comprehensive.py` → `integration/retrieval/`
- `test_epic1_ml_analyzer_comprehensive.py` → `integration/ml_infrastructure/`
- `components/query_processors/analyzers/ml_views/test_semantic_complexity_view.py` → `integration/ml_infrastructure/`
- `components/query_processors/analyzers/ml_views/test_computational_complexity_view.py` → `integration/ml_infrastructure/`
- `components/query_processors/analyzers/ml_views/test_linguistic_complexity_view.py` → `integration/ml_infrastructure/`
- `components/query_processors/analyzers/ml_views/test_task_complexity_view.py` → `integration/ml_infrastructure/`
- `components/query_processors/analyzers/ml_views/test_technical_complexity_view.py` → `integration/ml_infrastructure/`

**From `tests/component/` to `tests/integration/`**:
- `test_modular_answer_generator.py` → `integration/generation/`
- `test_embeddings.py` → `integration/embeddings/`
- `test_graph_components.py` → `integration/graph/`

2. **Update Import Paths**:

```python
# Before (in tests/unit/test_platform_orchestrator_phase2.py)
from src.core.platform_orchestrator import PlatformOrchestrator

# After (in tests/integration/platform/test_platform_orchestrator_phase2.py)
# Same import - Python path resolution handles it
from src.core.platform_orchestrator import PlatformOrchestrator
```

3. **Update conftest.py Files**:

```python
# tests/integration/conftest.py
import pytest
import sys
from pathlib import Path

# Ensure project root in path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def check_ml_dependencies():
    """Verify ML dependencies are available."""
    try:
        import torch
        import transformers
        import sentence_transformers
        return True
    except ImportError as e:
        pytest.fail(f"Integration tests require ML dependencies: {e}")

@pytest.fixture(scope="session")
def check_ollama_service():
    """Verify Ollama service is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            pytest.skip("Ollama service not available (required for generation tests)")
    except Exception:
        pytest.skip("Ollama service not running on localhost:11434")
```

**Expected Impact**:
- **tests/unit/**: ~200 tests, all fast, no external deps, 90%+ pass rate
- **tests/component/**: ~150 tests, moderate speed, lightweight deps, 85%+ pass rate
- **tests/integration/**: ~187 tests, slow, all deps, 60-70% pass rate (actual code issues)

---

## CI/CD Strategy

### Option 1: Separate Stages (Recommended)

```yaml
# .github/workflows/tests.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install unit test dependencies
        run: pip install -r requirements-test-unit.txt
      - name: Run unit tests
        run: pytest tests/unit -m "not requires_ml and not requires_ollama" --cov
      - name: Upload coverage
        uses: codecov/codecov-action@v3
    # This job MUST pass for PR to merge

  component-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - name: Install component test dependencies
        run: pip install -r requirements-test-component.txt
      - name: Run component tests
        run: pytest tests/component -m "not requires_ml and not requires_ollama"
    # This job SHOULD pass (warning if fails)

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5432:5432
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - name: Install integration test dependencies
        run: pip install -r requirements-test-integration.txt
      - name: Setup Ollama
        run: |
          curl -fsSL https://ollama.com/install.sh | sh
          ollama serve &
          sleep 5
          ollama pull llama3.2:3b
      - name: Run integration tests
        run: pytest tests/integration -m "integration" --timeout=300
    # This job is OPTIONAL (informational)
```

### Option 2: Marker-Based Execution (Current Structure)

```yaml
# .github/workflows/tests.yml
name: Test Suite

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - name: Install minimal dependencies
        run: pip install -r requirements-test-unit.txt
      - name: Run fast tests only
        run: |
          pytest -m "unit and not requires_ml and not requires_ollama" \
                 --maxfail=50 --tb=short
      - name: Run component tests (no ML/services)
        run: |
          pip install pdfplumber beautifulsoup4
          pytest tests/component -m "not requires_ml and not requires_ollama" \
                 --maxfail=50 --tb=short

  slow-tests:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # Only on main branch
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - name: Install full dependencies
        run: pip install -r requirements-test-integration.txt
      - name: Setup services
        run: |
          # Start Ollama, PostgreSQL, Redis
      - name: Run integration tests
        run: pytest -m "integration" --timeout=300
```

---

## Execution Plan: Step-by-Step

### Week 1: Phase 1 Implementation (Low Risk)

**Day 1-2: Add Markers**
```bash
# 1. Update pytest.ini with marker definitions
vim pytest.ini

# 2. Tag ~50 miscategorized tests
# Find candidates:
grep -r "integration test\|Integration test" tests/unit --include="*.py"
grep -r "import torch\|from torch\|import transformers" tests/unit --include="*.py"

# 3. Add markers to each file
@pytest.mark.integration
@pytest.mark.requires_ml
```

**Day 3: Test Selective Execution**
```bash
# Run unit tests only (should have far fewer failures)
pytest -m "unit and not requires_ml and not requires_ollama" -v

# Expected: ~200-300 failures instead of 537
# Failures should be actual code issues, not dependency issues
```

**Day 4: Document Results**
- Count failures per category
- Identify remaining high-payout issues
- Update CLAUDE.md with baseline

**Day 5: Create dependency files**
```bash
# Extract actual dependencies from imports
grep -r "^import\|^from" tests/unit --include="*.py" | sort | uniq
grep -r "^import\|^from" tests/component --include="*.py" | sort | uniq
grep -r "^import\|^from" tests/integration --include="*.py" | sort | uniq

# Create requirements-test-*.txt files
```

**Expected Outcome**:
- ✅ Clear separation of test categories via markers
- ✅ Baseline of ~200-300 actual failures (down from 537)
- ✅ Documentation of dependency requirements

### Week 2: Phase 2-3 Implementation (If Needed)

**Only proceed if user wants physical reorganization**

**Day 6-8: Move Files**
```bash
# Create directory structure
mkdir -p tests/integration/{platform,ml_infrastructure,generation,embeddings,graph,retrieval}

# Move files with git (preserves history)
git mv tests/unit/test_platform_orchestrator_phase2.py tests/integration/platform/

# Repeat for all 18 miscategorized files
```

**Day 9-10: Fix Imports and Test**
```bash
# Run tests in new locations
pytest tests/unit --maxfail=50
pytest tests/component --maxfail=50
pytest tests/integration --maxfail=100

# Fix any import issues
```

---

## Risk Mitigation

### Risks with Mitigation Strategies

**Risk 1: Breaking imports when moving files**
- Mitigation: Use `git mv` (preserves history)
- Mitigation: Test after each batch of moves
- Mitigation: Python path resolution should handle it (project root in sys.path)
- Rollback: `git revert` if issues arise

**Risk 2: CI/CD doesn't have ML dependencies**
- Mitigation: Start with Phase 1 (markers only, no CI changes needed)
- Mitigation: Update CI gradually after testing locally
- Mitigation: Make integration tests optional initially

**Risk 3: Breaking existing workflows**
- Mitigation: Add markers first, don't remove old test commands
- Mitigation: Both `pytest tests/unit` and `pytest -m unit` work
- Mitigation: Communicate changes to team

**Risk 4: Markers not comprehensive enough**
- Mitigation: Start with key markers (requires_ml, requires_ollama)
- Mitigation: Add more markers as needed
- Mitigation: Document marker usage in pytest.ini

---

## Expected Results

### Before Reorganization
```
$ pytest tests/ --tb=short
===================== 537 failed, 1373 passed, 57 skipped =====================
```

**Failure breakdown**:
- ~100-150: Missing ML dependencies (torch, transformers)
- ~50-100: Missing services (Ollama, PostgreSQL)
- ~287-387: Actual code issues

### After Phase 1 (Markers Added)
```
$ pytest -m "unit and not requires_ml and not requires_ollama" --tb=short
===================== 200-300 failed, 800-900 passed =====================

$ pytest -m "requires_ml" --tb=short
ERROR: torch not installed
(Skip all ML tests - as expected)

$ pytest -m "requires_ollama" --tb=short
50-100 skipped (Ollama not running - as expected)
```

### After Phase 2-3 (Full Reorganization)
```
$ pytest tests/unit/ --tb=short
===================== ~150 failed, ~650 passed =====================
(Fast, reliable unit tests)

$ pytest tests/component/ --tb=short  
===================== ~50 failed, ~120 passed =====================
(Moderate speed, lightweight deps)

$ pytest tests/integration/ --tb=short
===================== ~150 failed, ~100 passed =====================  
(Slow, but properly categorized)
```

**Total**: Same ~350-400 failures, but now clearly categorized and understood

---

## Success Metrics

### Phase 1 Success Criteria
- ✅ All tests tagged with appropriate markers
- ✅ Unit tests run without ML dependency errors
- ✅ Documented failure count per category
- ✅ Can run `pytest -m unit` successfully

### Phase 2 Success Criteria
- ✅ Dependency files created and tested
- ✅ CI can install dependencies per tier
- ✅ Local dev environment matches CI

### Phase 3 Success Criteria  
- ✅ All files in correct directories
- ✅ No import errors after moves
- ✅ Test counts match expected (unit: ~200, component: ~150, integration: ~187)
- ✅ CI/CD runs each tier appropriately

---

## Questions Answered

**Q: How does this fix the failures?**
A: It doesn't fix code issues, but it:
- Isolates ~150-250 dependency-related "failures" that aren't code bugs
- Makes actual code issues visible (not hidden by import errors)
- Enables running appropriate tests with appropriate environments

**Q: What if we just install all dependencies everywhere?**
A: Problems with that approach:
- Slower CI (10-30 min vs 2-5 min for fast tests)
- Heavier Docker images (~5GB vs ~500MB)
- Still need external services (Ollama, databases)
- Can't run fast tests locally without full ML stack
- Masks which tests actually need which dependencies

**Q: Can we start with markers and decide on file moves later?**
A: Yes! That's exactly Phase 1. Zero risk, immediate value.

**Q: How long will this take?**
A: Phase 1: 3-5 days (adding markers, testing selective execution)
A: Phase 2-3: Additional 5-7 days (if you want physical reorganization)

---

## Recommendation

**Start with Phase 1 only** (markers):
1. Low risk (no file moves)
2. Immediate value (isolate dependency issues)
3. Reversible (can remove markers if approach doesn't work)
4. Test the strategy before committing to file reorganization

**After Phase 1 succeeds, evaluate**:
- Are markers sufficient? (may be!)
- Do we still have ~200-300 actual code failures to fix?
- Do we want cleaner directory structure (Phase 2-3)?

