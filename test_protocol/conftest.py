"""
Shared test configuration and fixtures for RAG System Test Protocol.
"""

import sys
from pathlib import Path
import pytest

# Add project root to path (go up from test_protocol to project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also add the rag-portfolio root for shared_utils
rag_root = project_root.parent
sys.path.insert(0, str(rag_root))


@pytest.fixture
def config_path():
    """Provide path to test configuration file."""
    return Path("../config/default.yaml")


@pytest.fixture
def sample_embedder():
    """Provide a sample embedder for tests that need it."""
    from src.core.component_factory import ComponentFactory

    return ComponentFactory.create_embedder("sentence_transformer")
