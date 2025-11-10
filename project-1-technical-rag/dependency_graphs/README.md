# Dependency Graphs for RAG Portfolio Project

This directory contains comprehensive dependency graphs for all 6 main components in the `src/` directory, provided in both visual (Mermaid) and structured (JSON) formats.

## Overview

The RAG portfolio project is organized into 6 main components with well-defined internal structures and clear dependency relationships:

1. **🏗️ Core** - Foundation interfaces and orchestration
2. **🧩 Components** - Main component implementations  
3. **🔧 Shared Utils** - Shared utility functions and helpers
4. **📊 Evaluation** - Evaluation and metrics collection
5. **🎯 Training** - Machine learning training pipeline
6. **🧪 Testing** - Testing infrastructure and orchestration

## Dependency Graph Files

### JSON Format (Structured Data)
- [`overall_system_dependencies.json`](./overall_system_dependencies.json) - High-level system overview
- [`core_dependencies.json`](./core_dependencies.json) - Core module internal structure
- [`components_dependencies.json`](./components_dependencies.json) - Components module internal structure
- [`shared_utils_dependencies.json`](./shared_utils_dependencies.json) - Shared utils module internal structure
- [`training_dependencies.json`](./training_dependencies.json) - Training module internal structure
- [`evaluation_dependencies.json`](./evaluation_dependencies.json) - Evaluation module internal structure
- [`testing_dependencies.json`](./testing_dependencies.json) - Testing module internal structure

### Mermaid Format (Visual Diagrams)
The visual diagrams were created during the analysis and show:
1. Overall system dependency relationships
2. Individual component internal structures
3. Cross-component dependency flows
4. Architectural patterns and principles

## Key Architectural Insights

### Dependency Principles
- **One-way Dependencies**: Components → Core (no circular dependencies)
- **Interface-driven Architecture**: All components implement core interfaces
- **Shared Utilities**: Common functionality extracted to avoid duplication
- **Factory Pattern**: Centralized component creation and lifecycle management
- **Clean Architecture**: Clear separation of concerns and responsibilities

### Component Relationships

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Training  │───▶│ Components  │───▶│    Core     │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   ▲
                           ▼                   │
┌─────────────┐    ┌─────────────┐            │
│ Evaluation  │    │Shared Utils │────────────┘
└─────────────┘    └─────────────┘
       │                   ▲
       ▼                   │
┌─────────────┐            │
│   Testing   │────────────┘
└─────────────┘
```

### Cross-Component Dependencies
- **All Components** → `core.interfaces` (base types and contracts)
- **Components** → `shared_utils` (utility functions for specific domains)
- **Training** → `components.query_processors` (ML integration)
- **Evaluation** → `core.platform_orchestrator` (system coordination)
- **Testing** → Various testing adapters and utilities

## Usage

### JSON Files
The JSON files provide structured data that can be used for:
- Automated dependency analysis
- Documentation generation
- Architecture validation
- Impact analysis for changes
- Integration with development tools

### Example JSON Structure
```json
{
  "name": "Component Name",
  "description": "Component description",
  "type": "component_detail",
  "files": {...},
  "internal_dependencies": [...],
  "external_dependencies": [...],
  "architectural_patterns": [...]
}
```

### Visual Diagrams
The Mermaid diagrams provide visual representation for:
- Architecture documentation
- Onboarding new developers
- System design reviews
- Communication with stakeholders

## Component Details

### 🏗️ Core (`src/core/`)
- **interfaces.py**: Base types and contracts
- **component_factory.py**: Type-safe component creation
- **platform_orchestrator.py**: System lifecycle management
- **config.py**: Configuration management
- **query_processor.py**: Query processing coordination

### 🧩 Components (`src/components/`)
- **embedders/**: Text embedding (modular, sentence transformers)
- **generators/**: Answer generation (adaptive, Epic 1 multi-model)
- **processors/**: Document processing (PDF, hybrid parsing)
- **query_processors/**: Query analysis (modular, domain-aware)
- **retrievers/**: Information retrieval (unified, modular)
- **vector_stores/**: Vector storage (largely moved to retriever backends)

### 🔧 Shared Utils (`src/shared_utils/`)
- **document_processing/**: Parsing and chunking utilities
- **embeddings/**: Embedding generation utilities
- **generation/**: Generation providers and templates
- **query_processing/**: Query enhancement utilities
- **retrieval/**: Hybrid search and vocabulary indexing
- **vector_stores/**: Vector-specific utility implementations

### 📊 Evaluation (`src/evaluation/`)
- **retrieval_evaluator.py**: RAGAS-based scientific evaluation metrics

### 🎯 Training (`src/training/`)
- **epic1_training_orchestrator.py**: ML training pipeline coordinator
- **data_loader.py**: Training data management
- **view_trainer.py**: Individual ML model training
- **evaluation_framework.py**: ML model evaluation

### 🧪 Testing (`src/testing/`)
- **core/test_orchestrator.py**: Unified test execution engine
- **cli/test_cli.py**: Command-line interface for testing

## Maintenance

These dependency graphs should be updated when:
- New components are added
- Component interfaces change
- Dependencies are modified
- Architecture patterns evolve

The JSON format makes it easy to programmatically validate and update these relationships as the system evolves.
