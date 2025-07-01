"""
Comparative evaluation system for RAG methods.

Provides statistical analysis and benchmarking capabilities to determine
which retrieval method produces objectively better results.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict
import json
from pathlib import Path
import time

from .content_analysis import ContentAnalyzer


class ComparativeEvaluator:
    """
    Evaluates and compares different RAG retrieval methods.
    
    Provides statistical analysis to determine which method
    produces genuinely better results with confidence intervals.
    """
    
    def __init__(self):
        """Initialize comparative evaluator."""
        self.content_analyzer = ContentAnalyzer()
        self.evaluation_history = []
    
    def evaluate_single_query(self, 
                             query: str,
                             basic_results: List[Dict],
                             enhanced_results: List[Dict],
                             method_names: Tuple[str, str] = ("Basic", "Enhanced")) -> Dict[str, Any]:
        """
        Evaluate a single query across multiple methods.
        
        Args:
            query: Search query
            basic_results: Results from first method
            enhanced_results: Results from second method
            method_names: Names for the two methods
            
        Returns:
            Detailed evaluation results
        """
        start_time = time.perf_counter()
        
        # Perform comparative analysis
        comparison = self.content_analyzer.compare_result_sets(
            query, basic_results, enhanced_results
        )
        
        # Add method names
        comparison['method_names'] = method_names
        comparison['evaluation_time'] = time.perf_counter() - start_time
        
        # Store in history
        self.evaluation_history.append(comparison)
        
        return comparison
    
    def evaluate_query_set(self,
                          test_queries: List[str],
                          rag_system,
                          method_names: Tuple[str, str] = ("Basic", "Enhanced")) -> Dict[str, Any]:
        """
        Evaluate multiple queries for statistical significance.
        
        Args:
            test_queries: List of test queries
            rag_system: RAG system with query methods
            method_names: Names for comparison methods
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"🧪 Evaluating {len(test_queries)} queries...")
        
        query_results = []
        improvements = {
            'coverage': [],
            'coherence': [],
            'overall': []
        }
        
        for i, query in enumerate(test_queries, 1):
            print(f"   Query {i}/{len(test_queries)}: '{query}'")
            
            try:
                # Get results from both methods
                basic_results = rag_system.query(query, top_k=5)
                enhanced_results = rag_system.enhanced_hybrid_query(query, top_k=5)
                
                # Evaluate this query
                evaluation = self.evaluate_single_query(
                    query,
                    basic_results.get('chunks', []),
                    enhanced_results.get('chunks', []),
                    method_names
                )
                
                query_results.append(evaluation)
                
                # Collect improvement metrics
                improvements['coverage'].append(evaluation['improvements']['coverage_improvement_pct'])
                improvements['coherence'].append(evaluation['improvements']['coherence_improvement_pct'])
                improvements['overall'].append(evaluation['improvements']['overall_improvement_pct'])
                
            except Exception as e:
                print(f"   ⚠️  Error evaluating query '{query}': {e}")
                continue
        
        # Statistical analysis
        stats_analysis = self._perform_statistical_analysis(improvements)
        
        # Summary metrics
        successful_evaluations = len(query_results)
        enhanced_better_count = sum(1 for r in query_results if r['is_enhanced_better'])
        
        return {
            'query_count': len(test_queries),
            'successful_evaluations': successful_evaluations,
            'enhanced_better_percentage': enhanced_better_count / successful_evaluations * 100 if successful_evaluations > 0 else 0,
            'statistical_analysis': stats_analysis,
            'individual_results': query_results,
            'method_names': method_names,
            'summary_improvements': {
                'avg_coverage_improvement': np.mean(improvements['coverage']),
                'avg_coherence_improvement': np.mean(improvements['coherence']),
                'avg_overall_improvement': np.mean(improvements['overall'])
            }
        }
    
    def _perform_statistical_analysis(self, improvements: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        analysis = {}
        
        for metric, values in improvements.items():
            if len(values) < 2:
                analysis[metric] = {'insufficient_data': True}
                continue
            
            # Basic statistics
            mean_improvement = np.mean(values)
            std_improvement = np.std(values)
            median_improvement = np.median(values)
            
            # One-sample t-test against 0 (no improvement)
            t_stat, p_value = stats.ttest_1samp(values, 0)
            
            # Confidence interval (95%)
            n = len(values)
            se = std_improvement / np.sqrt(n)
            margin_error = stats.t.ppf(0.975, n-1) * se
            ci_lower = mean_improvement - margin_error
            ci_upper = mean_improvement + margin_error
            
            # Effect size (Cohen's d)
            cohens_d = mean_improvement / std_improvement if std_improvement > 0 else 0
            
            # Practical significance
            positive_improvements = sum(1 for v in values if v > 0)
            
            analysis[metric] = {
                'mean_improvement': mean_improvement,
                'median_improvement': median_improvement,
                'std_improvement': std_improvement,
                'sample_size': n,
                'positive_improvements': positive_improvements,
                'positive_percentage': positive_improvements / n * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'confidence_interval': (ci_lower, ci_upper),
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(cohens_d)
            }
        
        return analysis
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        method1, method2 = evaluation_results['method_names']
        
        report = []
        report.append("🔬 RAG Method Comparison Report")
        report.append("=" * 50)
        
        # Overview
        report.append(f"\n📊 Overview:")
        report.append(f"   • Methods compared: {method1} vs {method2}")
        report.append(f"   • Queries evaluated: {evaluation_results['successful_evaluations']}")
        report.append(f"   • {method2} better in: {evaluation_results['enhanced_better_percentage']:.1f}% of cases")
        
        # Statistical results
        stats = evaluation_results['statistical_analysis']
        
        report.append(f"\n📈 Statistical Analysis:")
        for metric, analysis in stats.items():
            if analysis.get('insufficient_data'):
                continue
                
            report.append(f"\n   {metric.title()} Improvement:")
            report.append(f"      • Mean: {analysis['mean_improvement']:+.1f}%")
            report.append(f"      • Median: {analysis['median_improvement']:+.1f}%")
            report.append(f"      • Positive improvements: {analysis['positive_percentage']:.1f}%")
            report.append(f"      • Statistical significance: {'✅ Yes' if analysis['is_significant'] else '❌ No'} (p={analysis['p_value']:.3f})")
            report.append(f"      • Effect size: {analysis['effect_size_interpretation']} (d={analysis['cohens_d']:.2f})")
            report.append(f"      • 95% CI: [{analysis['confidence_interval'][0]:+.1f}%, {analysis['confidence_interval'][1]:+.1f}%]")
        
        # Quality assessment
        report.append(f"\n✅ Quality Assessment:")
        
        overall_stats = stats.get('overall', {})
        if not overall_stats.get('insufficient_data'):
            if overall_stats['is_significant'] and overall_stats['mean_improvement'] > 0:
                report.append(f"   • {method2} shows statistically significant improvement")
                report.append(f"   • Effect size: {overall_stats['effect_size_interpretation']}")
                if overall_stats['positive_percentage'] > 70:
                    report.append(f"   • Improvement is consistent across queries")
                else:
                    report.append(f"   • Improvement varies significantly by query")
            else:
                report.append(f"   • No statistically significant improvement detected")
                report.append(f"   • May not be worth the additional complexity")
        
        # Recommendations
        report.append(f"\n🎯 Recommendations:")
        if overall_stats.get('is_significant') and overall_stats.get('mean_improvement', 0) > 5:
            report.append(f"   ✅ Use {method2} - shows clear improvement")
        elif overall_stats.get('mean_improvement', 0) > 0:
            report.append(f"   ⚠️  {method2} shows improvement but may not be significant")
            report.append(f"   → Consider A/B testing with real users")
        else:
            report.append(f"   ❌ Stick with {method1} - no clear benefit from {method2}")
        
        return "\n".join(report)
    
    def benchmark_methods(self,
                         rag_system,
                         test_queries: List[str],
                         methods: List[Tuple[str, callable]]) -> Dict[str, Any]:
        """
        Benchmark multiple retrieval methods.
        
        Args:
            rag_system: RAG system instance
            test_queries: List of test queries
            methods: List of (method_name, method_function) tuples
            
        Returns:
            Comprehensive benchmark results
        """
        print(f"🏁 Benchmarking {len(methods)} methods on {len(test_queries)} queries...")
        
        method_results = {}
        performance_metrics = defaultdict(list)
        
        # Evaluate each method
        for method_name, method_func in methods:
            print(f"\n📊 Evaluating {method_name}...")
            method_results[method_name] = []
            
            for query in test_queries:
                start_time = time.perf_counter()
                
                try:
                    # Get results from this method
                    results = method_func(query, top_k=5)
                    chunks = results.get('chunks', [])
                    
                    # Analyze quality
                    analysis = self.content_analyzer.comprehensive_analysis(query, chunks)
                    
                    # Record performance
                    execution_time = time.perf_counter() - start_time
                    performance_metrics[method_name].append(execution_time * 1000)  # ms
                    
                    method_results[method_name].append({
                        'query': query,
                        'analysis': analysis,
                        'execution_time_ms': execution_time * 1000
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  Error with {method_name} on '{query}': {e}")
                    continue
        
        # Calculate average quality scores
        quality_averages = {}
        for method_name, results in method_results.items():
            if results:
                avg_quality = np.mean([r['analysis']['overall_quality_score'] for r in results])
                avg_coverage = np.mean([r['analysis']['term_coverage']['avg_coverage'] for r in results])
                avg_coherence = np.mean([r['analysis']['semantic_coherence']['avg_coherence'] for r in results])
                avg_time = np.mean(performance_metrics[method_name])
                
                quality_averages[method_name] = {
                    'overall_quality': avg_quality,
                    'avg_coverage': avg_coverage,
                    'avg_coherence': avg_coherence,
                    'avg_execution_time_ms': avg_time
                }
        
        return {
            'methods_evaluated': len(methods),
            'queries_per_method': len(test_queries),
            'method_results': method_results,
            'quality_averages': quality_averages,
            'performance_metrics': dict(performance_metrics),
            'best_method': max(quality_averages.keys(), key=lambda k: quality_averages[k]['overall_quality']) if quality_averages else None
        }
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: Path) -> None:
        """Save evaluation results to JSON file."""
        # Convert numpy types for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj