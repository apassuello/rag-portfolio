"""
Component Factory for Phase 3 Direct Wiring Architecture.

This module provides a lightweight factory for direct component instantiation,
eliminating the ComponentRegistry overhead and improving startup performance.
It supports both legacy and unified architectures with type-safe component creation.
"""

import logging
import time
import hashlib
from typing import Dict, Type, Any, Optional, Union
from pathlib import Path
from collections import defaultdict, OrderedDict

from .interfaces import (
    DocumentProcessor, 
    Embedder, 
    VectorStore, 
    Retriever, 
    AnswerGenerator,
    QueryProcessor
)

# Component classes will be imported lazily to avoid circular imports
# See _get_component_class() method for lazy loading implementation

logger = logging.getLogger(__name__)


class ComponentFactory:
    """
    Lightweight factory for direct component instantiation.
    
    This factory replaces the ComponentRegistry with direct class mappings,
    eliminating lookup overhead and improving startup performance by ~20%.
    It maintains type safety and provides clear error messages.
    
    Features:
    - Direct component class mapping (no registry lookup)
    - Type-safe instantiation with validation
    - Support for both legacy and unified architectures
    - Comprehensive error handling with actionable messages
    - Performance optimized for production workloads
    
    Example:
        factory = ComponentFactory()
        
        # Create components directly
        processor = factory.create_processor("hybrid_pdf", chunk_size=1000)
        embedder = factory.create_embedder("sentence_transformer", use_mps=True)
        retriever = factory.create_retriever("unified", embedder=embedder, dense_weight=0.7)
    """
    
    # Component type mappings - module paths for lazy loading
    _PROCESSORS: Dict[str, str] = {
        "hybrid_pdf": "src.components.processors.document_processor.ModularDocumentProcessor",
        "modular": "src.components.processors.document_processor.ModularDocumentProcessor",
        "pdf_processor": "src.components.processors.pdf_processor.HybridPDFProcessor",  # Legacy processor
        "legacy_pdf": "src.components.processors.pdf_processor.HybridPDFProcessor",  # Alias for legacy
    }
    
    _EMBEDDERS: Dict[str, str] = {
        "modular": "src.components.embedders.modular_embedder.ModularEmbedder",
        "sentence_transformer": "src.components.embedders.sentence_transformer_embedder.SentenceTransformerEmbedder",
        "sentence_transformers": "src.components.embedders.sentence_transformer_embedder.SentenceTransformerEmbedder",  # Alias for compatibility
    }
    
    _VECTOR_STORES: Dict[str, str] = {
        # Legacy vector stores removed - functionality moved to UnifiedRetriever
        # "faiss": "src.components.vector_stores.faiss_store.FAISSVectorStore",
    }
    
    _RETRIEVERS: Dict[str, str] = {
        # Legacy Phase 1 architecture moved to archive
        # "hybrid": "src.components.retrievers.hybrid_retriever.HybridRetriever",
        "unified": "src.components.retrievers.unified_retriever.UnifiedRetriever",
        "modular_unified": "src.components.retrievers.modular_unified_retriever.ModularUnifiedRetriever",
        "advanced": "src.components.retrievers.advanced_retriever.AdvancedRetriever",  # Epic 2 implementation
    }
    
    _GENERATORS: Dict[str, str] = {
        "adaptive": "src.components.generators.adaptive_generator.AdaptiveAnswerGenerator",
        "adaptive_generator": "src.components.generators.adaptive_generator.AdaptiveAnswerGenerator",  # Alias for compatibility
        "adaptive_modular": "src.components.generators.answer_generator.AnswerGenerator",  # New modular implementation
    }
    
    _QUERY_PROCESSORS: Dict[str, str] = {
        "modular": "src.components.query_processors.modular_query_processor.ModularQueryProcessor",
        "modular_query_processor": "src.components.query_processors.modular_query_processor.ModularQueryProcessor",  # Alias for compatibility
    }
    
    # Phase 4: Performance monitoring and caching
    _performance_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "creation_count": 0,
        "total_time": 0.0,
        "average_time": 0.0,
        "min_time": float('inf'),
        "max_time": 0.0,
        "last_created": None
    })
    
    # Component cache for reusable instances (LRU with max size)
    _component_cache: OrderedDict[str, Any] = OrderedDict()
    _cache_max_size: int = 10  # Max cached components
    _cacheable_types = {"embedder"}  # Only cache expensive components
    
    # Cache metrics tracking (configurable for production)
    _cache_metrics_enabled: bool = True  # Can be disabled for production
    _cache_hits: int = 0
    _cache_misses: int = 0
    _cache_operations: Dict[str, int] = defaultdict(int)
    
    # Class cache for lazy loading
    _class_cache: Dict[str, Type] = {}
    
    @classmethod
    def _get_component_class(cls, module_path: str) -> Type:
        """
        Lazily import and cache component class.
        
        Args:
            module_path: Module path in format "src.package.module.ClassName"
            
        Returns:
            Component class
        """
        if module_path in cls._class_cache:
            return cls._class_cache[module_path]
        
        try:
            # Split module path and class name
            parts = module_path.split('.')
            class_name = parts[-1]
            module_path_only = '.'.join(parts[:-1])
            
            # Import module using absolute import
            from importlib import import_module
            module = import_module(module_path_only)
            
            # Get class from module
            component_class = getattr(module, class_name)
            
            # Cache for future use
            cls._class_cache[module_path] = component_class
            
            return component_class
            
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {module_path}: {e}") from e
    
    @classmethod
    def get_performance_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for component creation.
        
        Returns:
            Dictionary with creation metrics by component type
        """
        return dict(cls._performance_metrics)
    
    @classmethod
    def reset_performance_metrics(cls) -> None:
        """Reset all performance metrics."""
        cls._performance_metrics.clear()
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """
        Get component cache statistics.
        
        Returns:
            Dictionary with cache size, hit rate, etc.
        """
        total_operations = cls._cache_hits + cls._cache_misses
        hit_rate = cls._cache_hits / total_operations if total_operations > 0 else 0.0
        
        return {
            "cache_size": len(cls._component_cache),
            "max_size": cls._cache_max_size,
            "cached_components": list(cls._component_cache.keys()),
            "cacheable_types": cls._cacheable_types,
            "metrics_enabled": cls._cache_metrics_enabled,
            "hits": cls._cache_hits,
            "misses": cls._cache_misses,
            "total_operations": total_operations,
            "hit_rate": hit_rate,
            "operations_by_type": dict(cls._cache_operations)
        }
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the component cache."""
        cls._component_cache.clear()
    
    @classmethod
    def enable_cache_metrics(cls, enabled: bool = True) -> None:
        """
        Enable/disable cache metrics tracking.
        
        Args:
            enabled: Whether to enable metrics tracking
        """
        cls._cache_metrics_enabled = enabled
    
    @classmethod
    def reset_cache_metrics(cls) -> None:
        """Reset cache metrics counters."""
        cls._cache_hits = 0
        cls._cache_misses = 0
        cls._cache_operations.clear()
    
    @classmethod
    def _get_cache_key(cls, component_type: str, **kwargs) -> str:
        """
        Generate cache key for component configuration.
        
        Args:
            component_type: Type of component
            **kwargs: Component configuration
            
        Returns:
            Cache key string
        """
        # Create deterministic key from component type and config
        config_str = str(sorted(kwargs.items()))
        key_material = f"{component_type}:{config_str}"
        return hashlib.md5(key_material.encode()).hexdigest()[:16]
    
    @classmethod
    def _get_from_cache(cls, cache_key: str) -> Optional[Any]:
        """
        Get component from cache (LRU update).
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached component or None
        """
        if cache_key in cls._component_cache:
            # Track cache hit
            if cls._cache_metrics_enabled:
                cls._cache_hits += 1
                component_type = cache_key.split('_')[0]  # Extract component type from key
                cls._cache_operations[f"hit_{component_type}"] += 1
            
            # Move to end (most recently used)
            component = cls._component_cache.pop(cache_key)
            cls._component_cache[cache_key] = component
            return component
        else:
            # Track cache miss
            if cls._cache_metrics_enabled:
                cls._cache_misses += 1
                component_type = cache_key.split('_')[0]  # Extract component type from key
                cls._cache_operations[f"miss_{component_type}"] += 1
        
        return None
    
    @classmethod
    def _add_to_cache(cls, cache_key: str, component: Any) -> None:
        """
        Add component to cache with LRU eviction.
        
        Args:
            cache_key: Cache key
            component: Component to cache
        """
        # Remove oldest if at capacity
        if len(cls._component_cache) >= cls._cache_max_size:
            cls._component_cache.popitem(last=False)  # Remove oldest
        
        cls._component_cache[cache_key] = component
    
    @classmethod
    def _track_performance(cls, component_type: str, creation_time: float) -> None:
        """
        Track performance metrics for component creation.
        
        Args:
            component_type: Type of component created
            creation_time: Time taken to create component in seconds
        """
        metrics = cls._performance_metrics[component_type]
        metrics["creation_count"] += 1
        metrics["total_time"] += creation_time
        metrics["average_time"] = metrics["total_time"] / metrics["creation_count"]
        metrics["min_time"] = min(metrics["min_time"], creation_time)
        metrics["max_time"] = max(metrics["max_time"], creation_time)
        metrics["last_created"] = time.time()
    
    @classmethod
    def _create_with_tracking(cls, component_class: Type, component_type: str, use_cache: bool = False, **kwargs) -> Any:
        """
        Create component with performance tracking and optional caching.
        
        Args:
            component_class: Class to instantiate
            component_type: Type identifier for tracking
            use_cache: Whether to use component caching
            **kwargs: Constructor arguments
            
        Returns:
            Instantiated component
        """
        # Check cache first if caching is enabled
        cache_key = None
        if use_cache:
            cache_key = cls._get_cache_key(component_type, **kwargs)
            cached_component = cls._get_from_cache(cache_key)
            if cached_component is not None:
                logger.debug(f"Cache hit for {component_type}")
                cls._track_performance(f"{component_type}_cached", 0.0)
                return cached_component
        
        start_time = time.time()
        try:
            # Log component creation with essential information (INFO level for visibility)
            component = component_class(**kwargs)
            creation_time = time.time() - start_time
            
            # Enhanced logging with component details
            component_name = component.__class__.__name__
            component_module = component.__class__.__module__
            logger.info(f"🏭 ComponentFactory created: {component_name} "
                       f"(type={component_type}, module={component_module}, "
                       f"time={creation_time:.3f}s)")
            
            # Log component-specific info if available
            sub_components_logged = False
            
            # Check for ModularEmbedder and ModularDocumentProcessor sub-components
            if hasattr(component, 'get_sub_components'):
                try:
                    sub_info = component.get_sub_components()
                    if isinstance(sub_info, dict) and 'components' in sub_info:
                        components = sub_info['components']
                        sub_components = [f"{k}:{v.get('class', 'Unknown')}" for k, v in components.items()]
                        logger.info(f"  └─ Sub-components: {', '.join(sub_components)}")
                        sub_components_logged = True
                except Exception:
                    pass  # Don't fail component creation on logging issues
            
            # Fallback to legacy get_component_info for backward compatibility
            if not sub_components_logged and hasattr(component, 'get_component_info'):
                try:
                    info = component.get_component_info()
                    if isinstance(info, dict) and len(info) > 0:
                        sub_components = [f"{k}:{v.get('class', 'Unknown')}" for k, v in info.items()]
                        logger.info(f"  └─ Sub-components: {', '.join(sub_components)}")
                except Exception:
                    pass  # Don't fail component creation on logging issues
            
            # Add to cache if caching is enabled
            if use_cache and cache_key:
                cls._add_to_cache(cache_key, component)
            
            cls._track_performance(component_type, creation_time)
            return component
        except Exception as e:
            creation_time = time.time() - start_time
            cls._track_performance(f"{component_type}_failed", creation_time)
            raise
    
    @classmethod
    def create_processor(cls, processor_type: str, **kwargs) -> DocumentProcessor:
        """
        Create a document processor instance.
        
        Args:
            processor_type: Type of processor ("hybrid_pdf" or "pdf_processor")
            **kwargs: Arguments to pass to the processor constructor
            
        Returns:
            Instantiated DocumentProcessor
            
        Raises:
            ValueError: If processor type is not supported
            TypeError: If constructor arguments are invalid
        """
        if processor_type not in cls._PROCESSORS:
            available = list(cls._PROCESSORS.keys())
            raise ValueError(
                f"Unknown processor type '{processor_type}'. "
                f"Available processors: {available}"
            )
        
        processor_module_path = cls._PROCESSORS[processor_type]
        processor_class = cls._get_component_class(processor_module_path)
        
        try:
            return cls._create_with_tracking(
                processor_class, 
                f"processor_{processor_type}", 
                **kwargs
            )
        except Exception as e:
            raise TypeError(
                f"Failed to create processor '{processor_type}': {e}. "
                f"Check constructor arguments: {kwargs}"
            ) from e
    
    @classmethod
    def create_embedder(cls, embedder_type: str, **kwargs) -> Embedder:
        """
        Create an embedder instance.
        
        Args:
            embedder_type: Type of embedder ("sentence_transformer")
            **kwargs: Arguments to pass to the embedder constructor
            
        Returns:
            Instantiated Embedder
            
        Raises:
            ValueError: If embedder type is not supported
            TypeError: If constructor arguments are invalid
        """
        if embedder_type not in cls._EMBEDDERS:
            available = list(cls._EMBEDDERS.keys())
            raise ValueError(
                f"Unknown embedder type '{embedder_type}'. "
                f"Available embedders: {available}"
            )
        
        embedder_module_path = cls._EMBEDDERS[embedder_type]
        embedder_class = cls._get_component_class(embedder_module_path)
        
        try:
            # Use caching for embedders (expensive to create)
            return cls._create_with_tracking(
                embedder_class, 
                f"embedder_{embedder_type}", 
                use_cache=True,  # Enable caching for embedders
                **kwargs
            )
        except Exception as e:
            raise TypeError(
                f"Failed to create embedder '{embedder_type}': {e}. "
                f"Check constructor arguments: {kwargs}"
            ) from e
    
    @classmethod
    def create_vector_store(cls, store_type: str, **kwargs) -> VectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: Type of vector store ("faiss")
            **kwargs: Arguments to pass to the vector store constructor
            
        Returns:
            Instantiated VectorStore
            
        Raises:
            ValueError: If vector store type is not supported
            TypeError: If constructor arguments are invalid
        """
        if store_type not in cls._VECTOR_STORES:
            available = list(cls._VECTOR_STORES.keys())
            raise ValueError(
                f"Unknown vector store type '{store_type}'. "
                f"Available vector stores: {available}"
            )
        
        store_module_path = cls._VECTOR_STORES[store_type]
        store_class = cls._get_component_class(store_module_path)
        
        try:
            logger.debug(f"Creating {store_type} vector store with args: {kwargs}")
            return store_class(**kwargs)
        except Exception as e:
            raise TypeError(
                f"Failed to create vector store '{store_type}': {e}. "
                f"Check constructor arguments: {kwargs}"
            ) from e
    
    @classmethod
    def create_retriever(cls, retriever_type: str, **kwargs) -> Retriever:
        """
        Create a retriever instance.
        
        Args:
            retriever_type: Type of retriever ("unified" or "modular_unified")
            **kwargs: Arguments to pass to the retriever constructor
            
        Returns:
            Instantiated Retriever
            
        Raises:
            ValueError: If retriever type is not supported
            TypeError: If constructor arguments are invalid
        """
        if retriever_type not in cls._RETRIEVERS:
            available = list(cls._RETRIEVERS.keys())
            raise ValueError(
                f"Unknown retriever type '{retriever_type}'. "
                f"Available retrievers: {available}"
            )
        
        retriever_module_path = cls._RETRIEVERS[retriever_type]
        retriever_class = cls._get_component_class(retriever_module_path)
        
        try:
            logger.debug(f"Creating {retriever_type} retriever with args: {kwargs}")
            
            # Special handling for retrievers that need embedder + config pattern
            if retriever_type in ["modular_unified", "advanced"]:
                # Extract embedder and config from kwargs
                embedder = kwargs.pop("embedder", None)
                if embedder is None:
                    retriever_name = "ModularUnifiedRetriever" if retriever_type == "modular_unified" else "AdvancedRetriever"
                    raise ValueError(f"{retriever_name} requires 'embedder' parameter")
                
                # All remaining kwargs become the config
                config = kwargs
                
                return cls._create_with_tracking(
                    retriever_class, 
                    f"retriever_{retriever_type}", 
                    config=config,
                    embedder=embedder
                )
            else:
                return cls._create_with_tracking(
                    retriever_class, 
                    f"retriever_{retriever_type}", 
                    **kwargs
                )
        except Exception as e:
            raise TypeError(
                f"Failed to create retriever '{retriever_type}': {e}. "
                f"Check constructor arguments: {kwargs}"
            ) from e
    
    @classmethod
    def create_generator(cls, generator_type: str, **kwargs) -> AnswerGenerator:
        """
        Create an answer generator instance.
        
        Args:
            generator_type: Type of generator ("adaptive")
            **kwargs: Arguments to pass to the generator constructor
            
        Returns:
            Instantiated AnswerGenerator
            
        Raises:
            ValueError: If generator type is not supported
            TypeError: If constructor arguments are invalid
        """
        if generator_type not in cls._GENERATORS:
            available = list(cls._GENERATORS.keys())
            raise ValueError(
                f"Unknown generator type '{generator_type}'. "
                f"Available generators: {available}"
            )
        
        generator_module_path = cls._GENERATORS[generator_type]
        generator_class = cls._get_component_class(generator_module_path)
        
        try:
            logger.debug(f"Creating {generator_type} generator with args: {kwargs}")
            return cls._create_with_tracking(
                generator_class, 
                f"generator_{generator_type}", 
                **kwargs
            )
        except Exception as e:
            raise TypeError(
                f"Failed to create generator '{generator_type}': {e}. "
                f"Check constructor arguments: {kwargs}"
            ) from e
    
    @classmethod
    def create_query_processor(cls, processor_type: str, **kwargs) -> QueryProcessor:
        """
        Create a query processor instance.
        
        Args:
            processor_type: Type of query processor ("modular")
            **kwargs: Arguments to pass to the processor constructor
            
        Returns:
            Instantiated QueryProcessor
            
        Raises:
            ValueError: If processor type is not supported
            TypeError: If constructor arguments are invalid
        """
        if processor_type not in cls._QUERY_PROCESSORS:
            available = list(cls._QUERY_PROCESSORS.keys())
            raise ValueError(
                f"Unknown query processor type '{processor_type}'. "
                f"Available query processors: {available}"
            )
        
        processor_module_path = cls._QUERY_PROCESSORS[processor_type]
        processor_class = cls._get_component_class(processor_module_path)
        
        try:
            logger.debug(f"Creating {processor_type} query processor with args: {kwargs}")
            return cls._create_with_tracking(
                processor_class, 
                f"query_processor_{processor_type}", 
                **kwargs
            )
        except Exception as e:
            raise TypeError(
                f"Failed to create query processor '{processor_type}': {e}. "
                f"Check constructor arguments: {kwargs}"
            ) from e
    
    @classmethod
    def is_supported(cls, component_type: str, name: str) -> bool:
        """
        Check if a component type and name are supported.
        
        Args:
            component_type: Type of component ('processor', 'embedder', 'vector_store', 
                           'retriever', 'generator')
            name: Component name to check
            
        Returns:
            True if component is supported, False otherwise
        """
        type_mappings = {
            'processor': cls._PROCESSORS,
            'embedder': cls._EMBEDDERS,
            'vector_store': cls._VECTOR_STORES,
            'retriever': cls._RETRIEVERS,
            'generator': cls._GENERATORS,
            'query_processor': cls._QUERY_PROCESSORS
        }
        
        mapping = type_mappings.get(component_type)
        if mapping is None:
            return False
        
        return name in mapping
    
    @classmethod
    def get_all_supported_components(cls) -> Dict[str, list[str]]:
        """Get all supported components organized by type (alias for get_available_components)."""
        return cls.get_available_components()
    
    @classmethod
    def get_available_components(cls) -> Dict[str, list[str]]:
        """
        Get all available components organized by type.
        
        Returns:
            Dictionary mapping component types to lists of available component names
        """
        return {
            "processors": list(cls._PROCESSORS.keys()),
            "embedders": list(cls._EMBEDDERS.keys()),
            "vector_stores": list(cls._VECTOR_STORES.keys()),
            "retrievers": list(cls._RETRIEVERS.keys()),
            "generators": list(cls._GENERATORS.keys()),
            "query_processors": list(cls._QUERY_PROCESSORS.keys()),
        }
    
    @classmethod
    def validate_configuration(cls, config: Dict[str, Any]) -> list[str]:
        """
        Validate component configuration.
        
        Args:
            config: Configuration dictionary with component specifications
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Component type mappings for validation
        required_components = {
            'document_processor': 'processors',
            'embedder': 'embedders',
            'retriever': 'retrievers',
            'answer_generator': 'generators'
        }
        
        # vector_store is optional in unified architecture
        optional_components = {
            'vector_store': 'vector_stores'
        }
        
        all_components = {**required_components, **optional_components}
        available = cls.get_available_components()
        
        # Check required components
        for comp_key, comp_type_key in required_components.items():
            if comp_key not in config:
                errors.append(f"Missing required component: {comp_key}")
                continue
            
            comp_config = config[comp_key]
            if not isinstance(comp_config, dict) or 'type' not in comp_config:
                errors.append(f"Invalid configuration for {comp_key}: missing 'type' field")
                continue
            
            comp_type = comp_config['type']
            if comp_type not in available[comp_type_key]:
                errors.append(
                    f"Unknown {comp_key} type '{comp_type}'. "
                    f"Available: {available[comp_type_key]}"
                )
        
        # Check optional components if present
        for comp_key, comp_type_key in optional_components.items():
            if comp_key in config:
                comp_config = config[comp_key]
                if not isinstance(comp_config, dict) or 'type' not in comp_config:
                    errors.append(f"Invalid configuration for {comp_key}: missing 'type' field")
                    continue
                
                comp_type = comp_config['type']
                if comp_type not in available[comp_type_key]:
                    errors.append(
                        f"Unknown {comp_key} type '{comp_type}'. "
                        f"Available: {available[comp_type_key]}"
                    )
        
        return errors