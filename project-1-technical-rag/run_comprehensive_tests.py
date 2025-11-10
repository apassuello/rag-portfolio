#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner with Coverage and Reporting

This script runs all test categories and generates a single comprehensive report
including coverage analysis, performance metrics, and detailed test results.
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import os

def run_command(command, description):
    """Run a command and capture results."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=1800  # 30 minute timeout
        )
        end_time = time.time()
        
        return {
            'command': command,
            'description': description,
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': end_time - start_time
        }
    except subprocess.TimeoutExpired:
        return {
            'command': command,
            'description': description,
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Command timed out after 30 minutes',
            'execution_time': 1800
        }

def parse_pytest_output(stdout):
    """Parse pytest output to extract test statistics."""
    lines = stdout.split('\n')
    stats = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': 0,
        'warnings': 0
    }
    
    for line in lines:
        if 'passed' in line and 'failed' in line:
            # Look for summary line like "123 passed, 45 failed, 67 skipped"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'passed,' and i > 0:
                    stats['passed'] = int(parts[i-1])
                elif part == 'failed,' and i > 0:
                    stats['failed'] = int(parts[i-1])
                elif part == 'skipped' and i > 0:
                    stats['skipped'] = int(parts[i-1])
                elif part == 'error' and i > 0:
                    stats['errors'] = int(parts[i-1])
        elif ' passed in ' in line:
            # Single category result like "123 passed in 45.67s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'passed' and i > 0:
                    stats['passed'] = int(parts[i-1])
                    break
    
    stats['total_tests'] = stats['passed'] + stats['failed'] + stats['skipped'] + stats['errors']
    return stats

def generate_html_report(results, coverage_data):
    """Generate comprehensive HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Portfolio - Comprehensive Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
            .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
            .failure {{ background-color: #f8d7da; border-color: #f5c6cb; }}
            .stats {{ display: flex; gap: 20px; }}
            .stat-box {{ padding: 10px; border-radius: 5px; text-align: center; }}
            .stat-passed {{ background-color: #d4edda; }}
            .stat-failed {{ background-color: #f8d7da; }}
            .stat-skipped {{ background-color: #fff3cd; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            pre {{ background-color: #f8f9fa; padding: 10px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 RAG Portfolio - Comprehensive Test Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Epic 8 Architectural Compliance Assessment</strong></p>
        </div>
    """
    
    # Test Summary Section
    total_tests = sum(r.get('stats', {}).get('total_tests', 0) for r in results.values())
    total_passed = sum(r.get('stats', {}).get('passed', 0) for r in results.values())
    total_failed = sum(r.get('stats', {}).get('failed', 0) for r in results.values())
    total_skipped = sum(r.get('stats', {}).get('skipped', 0) for r in results.values())
    
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    html_content += f"""
        <div class="section success">
            <h2>📊 Test Execution Summary</h2>
            <div class="stats">
                <div class="stat-box stat-passed">
                    <h3>{total_passed}</h3>
                    <p>Passed</p>
                </div>
                <div class="stat-box stat-failed">
                    <h3>{total_failed}</h3>
                    <p>Failed</p>
                </div>
                <div class="stat-box stat-skipped">
                    <h3>{total_skipped}</h3>
                    <p>Skipped</p>
                </div>
                <div class="stat-box">
                    <h3>{pass_rate:.1f}%</h3>
                    <p>Pass Rate</p>
                </div>
            </div>
        </div>
    """
    
    # Test Category Results
    html_content += """
        <div class="section">
            <h2>🧪 Test Category Results</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Status</th>
                    <th>Passed</th>
                    <th>Failed</th>
                    <th>Skipped</th>
                    <th>Pass Rate</th>
                    <th>Duration</th>
                </tr>
    """
    
    for category, result in results.items():
        stats = result.get('stats', {})
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        status_class = "success" if result['success'] else "failure"
        
        total = stats.get('total_tests', 0)
        passed = stats.get('passed', 0)
        failed = stats.get('failed', 0)
        skipped = stats.get('skipped', 0)
        pass_rate = (passed / total * 100) if total > 0 else 0
        duration = f"{result['execution_time']:.1f}s"
        
        html_content += f"""
                <tr class="{status_class}">
                    <td><strong>{category}</strong></td>
                    <td>{status}</td>
                    <td>{passed}</td>
                    <td>{failed}</td>
                    <td>{skipped}</td>
                    <td>{pass_rate:.1f}%</td>
                    <td>{duration}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Coverage Section
    if coverage_data:
        html_content += f"""
        <div class="section">
            <h2>📈 Coverage Analysis</h2>
            <p><strong>Coverage Report:</strong> <a href="htmlcov/index.html">View Detailed Coverage Report</a></p>
            <p><strong>Coverage Data:</strong> Available in coverage.json</p>
        </div>
        """
    
    # Detailed Results
    html_content += """
        <div class="section">
            <h2>📋 Detailed Test Results</h2>
    """
    
    for category, result in results.items():
        status_class = "success" if result['success'] else "failure"
        html_content += f"""
            <div class="section {status_class}">
                <h3>{category}</h3>
                <p><strong>Command:</strong> <code>{result['command']}</code></p>
                <p><strong>Status:</strong> {'✅ SUCCESS' if result['success'] else '❌ FAILED'}</p>
                <p><strong>Duration:</strong> {result['execution_time']:.1f} seconds</p>
                
                <h4>Output:</h4>
                <pre>{result['stdout'][:2000]}</pre>
                
                {f'<h4>Errors:</h4><pre>{result["stderr"][:1000]}</pre>' if result['stderr'] else ''}
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    """Main execution function."""
    print("🚀 RAG Portfolio - Comprehensive Test Suite Runner")
    print("=" * 60)
    
    # Test categories to run
    test_categories = {
        'Epic 8 Tests': 'pytest tests/epic8/ -v --tb=short',
        'Epic 1 Tests': 'pytest tests/epic1/ -v --tb=short -x',
        'Epic 2 Tests': 'pytest tests/epic2_validation/ -v --tb=short',
        'Unit Tests': 'pytest tests/unit/ -v --tb=short',
        'Integration Tests': 'pytest tests/integration/ -v --tb=short', 
        'Component Tests': 'pytest tests/component/ -v --tb=short',
        'System Tests': 'pytest tests/system/ -v --tb=short',
        'Epic 2 New Tests': 'pytest tests/epic2_epic8_integration.py tests/epic2_calibration_validation.py tests/epic2_production_validation.py -v --tb=short'
    }
    
    # Run comprehensive test with coverage
    coverage_command = 'pytest --cov=src --cov-report=html --cov-report=json --cov-report=term-missing'
    
    results = {}
    
    # Run each test category
    for category, command in test_categories.items():
        result = run_command(command, f"Running {category}")
        result['stats'] = parse_pytest_output(result['stdout'])
        results[category] = result
    
    # Run comprehensive coverage analysis
    print(f"\n{'='*60}")
    print("📊 Generating Coverage Report")
    print(f"{'='*60}")
    
    coverage_result = run_command(coverage_command, "Comprehensive Coverage Analysis")
    results['Coverage Analysis'] = coverage_result
    
    # Load coverage data if available
    coverage_data = None
    try:
        if Path('coverage.json').exists():
            with open('coverage.json', 'r') as f:
                coverage_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load coverage data: {e}")
    
    # Generate comprehensive report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path('test_reports')
    report_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    json_path = report_dir / f'comprehensive_test_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results,
            'coverage': coverage_data,
            'summary': {
                'total_categories': len(test_categories),
                'successful_categories': sum(1 for r in results.values() if r['success']),
                'total_tests': sum(r.get('stats', {}).get('total_tests', 0) for r in results.values()),
                'total_passed': sum(r.get('stats', {}).get('passed', 0) for r in results.values()),
                'total_failed': sum(r.get('stats', {}).get('failed', 0) for r in results.values()),
            }
        }, f, indent=2)
    
    # Generate HTML report
    html_content = generate_html_report(results, coverage_data)
    html_path = report_dir / f'comprehensive_test_report_{timestamp}.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 COMPREHENSIVE TEST EXECUTION COMPLETE")
    print(f"{'='*60}")
    
    total_tests = sum(r.get('stats', {}).get('total_tests', 0) for r in results.values())
    total_passed = sum(r.get('stats', {}).get('passed', 0) for r in results.values())
    total_failed = sum(r.get('stats', {}).get('failed', 0) for r in results.values())
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"🎯 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_passed}")
    print(f"   Failed: {total_failed}")
    print(f"   Pass Rate: {pass_rate:.1f}%")
    print(f"   Categories Run: {len(test_categories)}")
    
    print(f"\n📊 REPORTS GENERATED:")
    print(f"   JSON Results: {json_path}")
    print(f"   HTML Report: {html_path}")
    if Path('htmlcov/index.html').exists():
        print(f"   Coverage Report: htmlcov/index.html")
    if Path('coverage.json').exists():
        print(f"   Coverage Data: coverage.json")
    
    print(f"\n🔗 QUICK ACCESS:")
    print(f"   Open HTML Report: open {html_path}")
    print(f"   Open Coverage: open htmlcov/index.html")
    
    return pass_rate >= 70  # Return success if pass rate is acceptable

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)