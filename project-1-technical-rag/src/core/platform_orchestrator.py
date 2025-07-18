"""
Platform Orchestrator - System lifecycle and platform integration.

This component manages the system lifecycle, component initialization,
and platform-specific adaptations with factory-based architecture.
It uses ComponentFactory for direct component instantiation with optimal performance.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .interfaces import Document, Answer, RetrievalResult
from .config import ConfigManager, PipelineConfig
from .component_factory import ComponentFactory

logger = logging.getLogger(__name__)


class PlatformOrchestrator:
    """
    Platform Orchestrator manages system lifecycle and platform integration.
    
    Responsibilities:
    - Component initialization and dependency injection
    - Configuration management
    - Platform-specific adaptations (cloud, on-premise, edge)
    - System health monitoring
    - Resource management
    - API exposure and routing
    
    This class uses the ComponentFactory for direct component instantiation
    during Phase 3, providing improved performance while maintaining backward compatibility.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize platform orchestrator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        # Convert string to Path if needed
        if isinstance(config_path, str):
            config_path = Path(config_path)
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Component storage
        self._components: Dict[str, Any] = {}
        self._initialized = False
        
        # Phase 2: Track architecture type for compatibility
        self._using_unified_retriever = False
        self._retriever_type = None
        
        # Initialize system
        self._initialize_system()
        
        logger.info(f"Platform Orchestrator initialized with config: {config_path}")
    
    def _initialize_system(self) -> None:
        """
        Initialize all system components with Phase 3 direct wiring.
        
        This method uses ComponentFactory for direct component instantiation,
        supporting both legacy and unified architectures with improved performance.
        """
        logger.info("Initializing RAG system components...")
        
        try:
            # Create document processor using factory
            proc_config = self.config.document_processor
            self._components['document_processor'] = ComponentFactory.create_processor(
                proc_config.type,
                **proc_config.config
            )
            logger.debug(f"Document processor initialized: {proc_config.type}")
            
            # Create embedder using factory
            emb_config = self.config.embedder
            self._components['embedder'] = ComponentFactory.create_embedder(
                emb_config.type,
                **emb_config.config
            )
            logger.debug(f"Embedder initialized: {emb_config.type}")
            
            # Phase 3: Architecture detection with factory-based instantiation
            ret_config = self.config.retriever
            if ret_config.type in ["unified", "modular_unified", "advanced"]:
                # Phase 2: Use unified retriever (no separate vector store needed)
                self._components['retriever'] = ComponentFactory.create_retriever(
                    ret_config.type,
                    embedder=self._components['embedder'],
                    **ret_config.config
                )
                logger.info(f"Phase 3: Unified retriever initialized: {ret_config.type}")
                
                # Mark that we're using unified architecture
                self._using_unified_retriever = True
                self._retriever_type = ret_config.type
                
            else:
                # Phase 1: Legacy architecture with separate vector store and retriever
                vs_config = self.config.vector_store
                if vs_config is None:
                    raise RuntimeError("Legacy architecture requires vector_store configuration")
                
                self._components['vector_store'] = ComponentFactory.create_vector_store(
                    vs_config.type,
                    **vs_config.config
                )
                logger.debug(f"Vector store initialized: {vs_config.type}")
                
                self._components['retriever'] = ComponentFactory.create_retriever(
                    ret_config.type,
                    vector_store=self._components['vector_store'],
                    embedder=self._components['embedder'],
                    **ret_config.config
                )
                logger.debug(f"Retriever initialized: {ret_config.type}")
                
                # Mark that we're using legacy architecture
                self._using_unified_retriever = False
                self._retriever_type = ret_config.type
            
            # Create answer generator using factory
            gen_config = self.config.answer_generator
            self._components['answer_generator'] = ComponentFactory.create_generator(
                gen_config.type,
                **gen_config.config
            )
            logger.debug(f"Answer generator initialized: {gen_config.type}")
            
            # Note: Query processor will be created in the next step
            # For now, we'll handle query processing directly
            
            self._initialized = True
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise RuntimeError(f"System initialization failed: {str(e)}") from e
    
    def process_document(self, file_path: Path) -> int:
        """
        Process and index a document.
        
        This method orchestrates the document processing workflow:
        1. Process document into chunks
        2. Generate embeddings for chunks
        3. Store chunks and embeddings in vector store
        
        Args:
            file_path: Path to document file
            
        Returns:
            Number of document chunks created
            
        Raises:
            FileNotFoundError: If document file doesn't exist
            RuntimeError: If processing fails
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Process document using existing component
            processor = self._components['document_processor']
            documents = processor.process(file_path)
            
            if not documents:
                logger.warning(f"No documents created from {file_path}")
                return 0
            
            logger.debug(f"Created {len(documents)} document chunks")
            
            # Generate embeddings using existing component
            embedder = self._components['embedder']
            texts = [doc.content for doc in documents]
            embeddings = embedder.embed(texts)
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            # Phase 2: Handle unified vs legacy architecture
            retriever = self._components['retriever']
            
            if self._using_unified_retriever:
                # Phase 2: Direct indexing in unified retriever
                retriever.index_documents(documents)
                logger.debug(f"Indexed documents in unified retriever")
            else:
                # Phase 1: Legacy architecture - store in vector store first
                vector_store = self._components['vector_store']
                vector_store.add(documents)
                
                # Then index in retriever if it supports it
                if hasattr(retriever, 'index_documents'):
                    retriever.index_documents(documents)
                
                logger.debug(f"Indexed documents in legacy vector store + retriever")
            
            logger.info(f"Successfully indexed {len(documents)} chunks from {file_path}")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            raise RuntimeError(f"Document processing failed: {str(e)}") from e
    
    def process_documents(self, file_paths: List[Path]) -> Dict[str, int]:
        """
        Process multiple documents and return chunk counts.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            Dictionary mapping file paths to chunk counts
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        results = {}
        failed_files = []
        
        logger.info(f"Processing {len(file_paths)} documents...")
        
        for file_path in file_paths:
            try:
                chunk_count = self.process_document(file_path)
                results[str(file_path)] = chunk_count
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                failed_files.append(str(file_path))
                results[str(file_path)] = 0
        
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
        
        total_chunks = sum(results.values())
        logger.info(f"Batch processing complete: {total_chunks} total chunks from {len(file_paths)} files")
        
        return results
    
    def index_documents(self, documents: List[Document]) -> int:
        """
        Directly index pre-created Document objects into the retrieval system.
        
        This method allows bypassing file processing when you already have
        Document objects (e.g., from testing, external processing, etc.).
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Number of documents successfully indexed
            
        Raises:
            RuntimeError: If system not initialized or indexing fails
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        if not documents:
            logger.warning("No documents provided for indexing")
            return 0
        
        logger.info(f"Indexing {len(documents)} pre-created documents...")
        
        try:
            # Generate embeddings for documents that don't have them
            embedder = self._components['embedder']
            documents_needing_embeddings = [doc for doc in documents if doc.embedding is None]
            
            if documents_needing_embeddings:
                logger.debug(f"Generating embeddings for {len(documents_needing_embeddings)} documents")
                texts = [doc.content for doc in documents_needing_embeddings]
                embeddings = embedder.embed(texts)
                
                # Add embeddings to documents
                for doc, embedding in zip(documents_needing_embeddings, embeddings):
                    doc.embedding = embedding
            
            # Index documents based on architecture
            retriever = self._components['retriever']
            
            if self._using_unified_retriever:
                # Phase 4: Direct indexing in unified retriever
                retriever.index_documents(documents)
                logger.debug(f"Indexed documents in unified retriever")
            else:
                # Legacy architecture - store in vector store first
                vector_store = self._components['vector_store']
                vector_store.add(documents)
                
                # Then index in retriever if it supports it
                if hasattr(retriever, 'index_documents'):
                    retriever.index_documents(documents)
                
                logger.debug(f"Indexed documents in legacy vector store + retriever")
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            raise RuntimeError(f"Document indexing failed: {str(e)}") from e
    
    def process_query(self, query: str, k: int = 5) -> Answer:
        """
        Process a query and return an answer.
        
        This method orchestrates the query processing workflow:
        1. Retrieve relevant documents
        2. Generate answer from query and context
        
        Note: In Phase 1, this directly implements query processing.
        In Phase 3, this will delegate to the QueryProcessor component.
        
        Args:
            query: User query string
            k: Number of documents to retrieve for context
            
        Returns:
            Answer object with generated text, sources, and metadata
            
        Raises:
            ValueError: If query is empty or k is invalid
            RuntimeError: If query processing fails
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        if k <= 0:
            raise ValueError("k must be positive")
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Retrieve relevant documents using existing component
            retriever = self._components['retriever']
            results = retriever.retrieve(query, k)
            
            if not results:
                logger.warning(f"No relevant documents found for query: {query}")
                return Answer(
                    text="No relevant information found for your query.",
                    sources=[],
                    confidence=0.0,
                    metadata={
                        "query": query,
                        "retrieved_docs": 0,
                        "orchestrator": "PlatformOrchestrator"
                    }
                )
            
            logger.debug(f"Retrieved {len(results)} relevant documents")
            
            # Extract documents from results
            context_docs = [r.document for r in results]
            
            # Generate answer using existing component
            generator = self._components['answer_generator']
            answer = generator.generate(query, context_docs)
            
            # Add orchestrator metadata
            answer.metadata.update({
                "query": query,
                "retrieved_docs": len(results),
                "retrieval_scores": [r.score for r in results],
                "retrieval_methods": [r.retrieval_method for r in results],
                "orchestrator": "PlatformOrchestrator"
            })
            
            logger.info(f"Generated answer with confidence: {answer.confidence}")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            raise RuntimeError(f"Query processing failed: {str(e)}") from e
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health information.
        
        Returns:
            Dictionary with system health metrics and component status
        """
        health = {
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "architecture": self._determine_system_architecture(),
            "config_path": str(self.config_path),
            "components": {},
            "platform": self.config.global_settings.get("platform", {})
        }
        
        # Phase 4: Enhanced factory and performance monitoring
        try:
            from .component_factory import ComponentFactory
            health["factory_info"] = ComponentFactory.get_available_components()
            health["performance_metrics"] = ComponentFactory.get_performance_metrics()
            health["cache_stats"] = ComponentFactory.get_cache_stats()
        except ImportError:
            pass  # Factory not available
        
        if self._initialized:
            # Get component status
            for name, component in self._components.items():
                component_info = {
                    "type": type(component).__name__,
                    "module": type(component).__module__,
                    "healthy": True,  # Will be updated by health checks
                    "factory_managed": True,  # Phase 3: All components now factory-managed
                    "created_at": getattr(component, '_created_at', None),
                    "last_used": getattr(component, '_last_used', None),
                    "health_checks": {}
                }
                
                # Phase 4: Enhanced component health validation
                component_info["healthy"] = self._validate_component_health(component, component_info["health_checks"])
                
                # Add component-specific health info if available
                if hasattr(component, 'get_stats'):
                    try:
                        component_info["stats"] = component.get_stats()
                    except Exception as e:
                        component_info["healthy"] = False
                        component_info["error"] = str(e)
                elif hasattr(component, 'get_configuration'):
                    component_info["config"] = component.get_configuration()
                elif hasattr(component, 'get_model_info'):
                    component_info["config"] = component.get_model_info()
                
                health["components"][name] = component_info
        
        # Phase 4: Add deployment readiness check
        health["deployment_readiness"] = self._check_deployment_readiness()
        
        return health
    
    def _validate_component_health(self, component: Any, health_checks: Dict[str, Any]) -> bool:
        """
        Validate component health with comprehensive checks.
        
        Args:
            component: Component to validate
            health_checks: Dictionary to store health check results
            
        Returns:
            True if component is healthy, False otherwise
        """
        overall_healthy = True
        
        # Check 1: Required methods exist
        required_methods = {
            "DocumentProcessor": ["process"],
            "Embedder": ["embed", "embedding_dim"],
            "VectorStore": ["add", "search"],
            "Retriever": ["retrieve"],
            "AnswerGenerator": ["generate"]
        }
        
        component_type = type(component).__name__
        if component_type in required_methods:
            missing_methods = []
            for method in required_methods[component_type]:
                if not hasattr(component, method):
                    missing_methods.append(method)
                    overall_healthy = False
            
            health_checks["required_methods"] = {
                "passed": len(missing_methods) == 0,
                "missing": missing_methods
            }
        
        # Check 2: Component-specific health validation
        if hasattr(component, 'health_check'):
            try:
                component_health = component.health_check()
                health_checks["component_specific"] = {
                    "passed": component_health.get("healthy", True),
                    "details": component_health
                }
                if not component_health.get("healthy", True):
                    overall_healthy = False
            except Exception as e:
                health_checks["component_specific"] = {
                    "passed": False,
                    "error": str(e)
                }
                overall_healthy = False
        
        # Check 3: Memory usage validation (if available)
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            health_checks["memory"] = {
                "current_mb": round(memory_mb, 1),
                "within_limits": memory_mb < 2048  # 2GB limit
            }
            if memory_mb > 2048:
                overall_healthy = False
        except ImportError:
            health_checks["memory"] = {"available": False}
        
        # Check 4: Configuration validation
        if hasattr(component, 'get_configuration'):
            try:
                config = component.get_configuration()
                health_checks["configuration"] = {
                    "passed": isinstance(config, dict),
                    "config_size": len(config) if isinstance(config, dict) else 0
                }
            except Exception as e:
                health_checks["configuration"] = {
                    "passed": False,
                    "error": str(e)
                }
                overall_healthy = False
        
        return overall_healthy
    
    def _check_deployment_readiness(self) -> Dict[str, Any]:
        """
        Check system readiness for cloud deployment.
        
        Returns:
            Dictionary with deployment readiness assessment
        """
        readiness = {
            "ready": True,
            "score": 100,
            "checks": {},
            "recommendations": []
        }
        
        # Check 1: All components initialized
        if not self._initialized:
            readiness["ready"] = False
            readiness["score"] -= 50
            readiness["checks"]["initialization"] = {
                "passed": False,
                "message": "System not initialized"
            }
        else:
            readiness["checks"]["initialization"] = {
                "passed": True,
                "message": "System fully initialized"
            }
        
        # Check 2: Memory usage within limits
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > 1024:  # 1GB warning threshold
                readiness["score"] -= 20
                readiness["recommendations"].append("Memory usage high - consider optimization")
            
            readiness["checks"]["memory"] = {
                "passed": memory_mb < 2048,  # 2GB hard limit
                "current_mb": round(memory_mb, 1),
                "limit_mb": 2048
            }
            
            if memory_mb >= 2048:
                readiness["ready"] = False
                readiness["score"] -= 30
                
        except ImportError:
            readiness["checks"]["memory"] = {
                "passed": False,
                "message": "Memory monitoring not available"
            }
            readiness["score"] -= 10
        
        # Check 3: Configuration validation
        try:
            errors = self.validate_configuration()
            if errors:
                readiness["ready"] = False
                readiness["score"] -= 40
                readiness["checks"]["configuration"] = {
                    "passed": False,
                    "errors": errors
                }
            else:
                readiness["checks"]["configuration"] = {
                    "passed": True,
                    "message": "Configuration valid"
                }
        except Exception as e:
            readiness["ready"] = False
            readiness["score"] -= 40
            readiness["checks"]["configuration"] = {
                "passed": False,
                "error": str(e)
            }
        
        # Check 4: Performance metrics
        try:
            from .component_factory import ComponentFactory
            metrics = ComponentFactory.get_performance_metrics()
            cache_stats = ComponentFactory.get_cache_stats()
            
            # Check if we have reasonable performance
            slow_components = []
            for comp_type, stats in metrics.items():
                if stats.get("average_time", 0) > 5.0:  # 5 second threshold
                    slow_components.append(comp_type)
            
            if slow_components:
                readiness["score"] -= 15
                readiness["recommendations"].append(f"Slow components detected: {slow_components}")
            
            readiness["checks"]["performance"] = {
                "passed": len(slow_components) == 0,
                "metrics": {
                    "tracked_components": len(metrics),
                    "cache_usage": f"{cache_stats['cache_size']}/{cache_stats['max_size']}",
                    "slow_components": slow_components
                }
            }
            
        except ImportError:
            readiness["checks"]["performance"] = {
                "passed": False,
                "message": "Performance monitoring not available"
            }
            readiness["score"] -= 10
        
        # Check 5: Environment variables for production
        prod_env_vars = ["RAG_ENV", "RAG_LOG_LEVEL"]
        missing_env_vars = [var for var in prod_env_vars if not os.getenv(var)]
        
        if missing_env_vars:
            readiness["score"] -= 5
            readiness["recommendations"].append(f"Consider setting: {missing_env_vars}")
        
        readiness["checks"]["environment"] = {
            "passed": len(missing_env_vars) == 0,
            "missing_vars": missing_env_vars,
            "current_env": self.config.global_settings.get("platform", "unknown")
        }
        
        # Final score adjustment
        readiness["score"] = max(0, readiness["score"])
        
        # Add readiness level
        if readiness["score"] >= 90:
            readiness["level"] = "production_ready"
        elif readiness["score"] >= 70:
            readiness["level"] = "staging_ready"
        elif readiness["score"] >= 50:
            readiness["level"] = "development_ready"
        else:
            readiness["level"] = "not_ready"
        
        return readiness
    
    def _determine_system_architecture(self) -> str:
        """
        Determine the overall system architecture based on component types.
        
        Returns:
            String describing the current system architecture
        """
        if not self._initialized:
            return "uninitialized"
        
        # Check component types to determine architecture level
        component_types = {}
        for name, component in self._components.items():
            component_types[name] = type(component).__name__
        
        # Determine architecture based on modular component usage
        modular_components = 0
        total_components = 0
        
        # Check each major component for modular architecture
        if 'document_processor' in component_types:
            total_components += 1
            if component_types['document_processor'] == 'ModularDocumentProcessor':
                modular_components += 1
        
        if 'embedder' in component_types:
            total_components += 1
            if component_types['embedder'] == 'ModularEmbedder':
                modular_components += 1
        
        if 'retriever' in component_types:
            total_components += 1
            if component_types['retriever'] == 'ModularUnifiedRetriever':
                modular_components += 1
        
        if 'answer_generator' in component_types:
            total_components += 1
            if component_types['answer_generator'] == 'AnswerGenerator':  # This is already modular
                modular_components += 1
        
        # Determine architecture level
        if total_components == 0:
            return "no_components"
        
        modular_percentage = modular_components / total_components
        
        if modular_percentage >= 1.0:
            return "modular"  # All components are modular
        elif modular_percentage >= 0.75:
            return "mostly_modular"  # 3/4 or more components are modular
        elif modular_percentage >= 0.5:
            return "hybrid"  # Half of components are modular
        elif self._using_unified_retriever:
            return "unified"  # Using unified retriever but not fully modular
        else:
            return "legacy"  # Legacy architecture
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a specific component for testing/debugging.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        return self._components.get(name)
    
    def clear_index(self) -> None:
        """
        Clear all indexed documents from the vector store.
        
        This method resets the vector store to its initial state.
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        vector_store = self._components['vector_store']
        vector_store.clear()
        
        # Also clear retriever if it has separate state
        retriever = self._components['retriever']
        if hasattr(retriever, 'clear'):
            retriever.clear()
        
        logger.info("System index cleared")
    
    def reload_config(self) -> None:
        """
        Reload configuration and reinitialize components.
        
        This method allows for dynamic configuration changes without
        creating a new orchestrator instance.
        
        Raises:
            RuntimeError: If reinitialization fails
        """
        logger.info("Reloading system configuration...")
        
        try:
            # Reload configuration
            self.config_manager = ConfigManager(self.config_path)
            self.config = self.config_manager.config
            
            # Clear existing components
            self._components.clear()
            self._initialized = False
            
            # Reinitialize components
            self._initialize_system()
            
            logger.info("System configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {str(e)}")
            raise RuntimeError(f"Configuration reload failed: {str(e)}") from e
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Create configuration dict for factory validation
            config_dict = {
                'document_processor': {
                    'type': self.config.document_processor.type,
                    'config': self.config.document_processor.config
                },
                'embedder': {
                    'type': self.config.embedder.type,
                    'config': self.config.embedder.config
                },
                'retriever': {
                    'type': self.config.retriever.type,
                    'config': self.config.retriever.config
                },
                'answer_generator': {
                    'type': self.config.answer_generator.type,
                    'config': self.config.answer_generator.config
                }
            }
            
            # Add vector_store if present (optional for unified architecture)
            if self.config.vector_store is not None:
                config_dict['vector_store'] = {
                    'type': self.config.vector_store.type,
                    'config': self.config.vector_store.config
                }
            
            # Use factory validation
            errors = ComponentFactory.validate_configuration(config_dict)
            
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    def __str__(self) -> str:
        """String representation of the orchestrator."""
        return f"PlatformOrchestrator(config={self.config_path}, initialized={self._initialized})"
    
    def __repr__(self) -> str:
        """Detailed representation of the orchestrator."""
        return (f"PlatformOrchestrator(config_path={self.config_path}, "
                f"initialized={self._initialized}, "
                f"components={list(self._components.keys())})")