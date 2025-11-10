#!/bin/bash
# Fixed Test Runner Shell Wrapper
# Swiss Engineering Quality Standards

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 RAG Portfolio - Fixed Test Runner"
echo "======================================"

# Function to show usage
show_usage() {
    echo "Usage: $0 [LEVEL] [OPTIONS]"
    echo ""
    echo "LEVELS:"
    echo "  focused      - Core component tests only (~50-100 tests)"
    echo "  basic        - Priority 1 tests (~100-150 tests)"
    echo "  working      - Priority 1-3 tests (~200-300 tests)"
    echo ""
    echo "OPTIONS:"
    echo "  --no-coverage    Skip coverage analysis"
    echo "  --epic8          Filter for Epic 8 tests only"
    echo "  --epic1          Filter for Epic 1 tests only"
    echo "  --save FILE      Save results to JSON file"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 focused                    # Quick focused test run with coverage"
    echo "  $0 working --no-coverage      # Working tests without coverage"
    echo "  $0 basic --epic8              # Basic Epic 8 tests only"
    echo ""
}

# Parse arguments
LEVEL="focused"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        focused|basic|working)
            LEVEL="$1"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        --no-coverage|--epic8|--epic1|--save)
            EXTRA_ARGS+=("$1")
            shift
            # Handle --save argument value
            if [[ "$1" == "--save" && $# -gt 1 ]]; then
                EXTRA_ARGS+=("$2")
                shift
            fi
            ;;
        *)
            echo "⚠️  Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "📋 Configuration:"
echo "   Level: $LEVEL"
echo "   Extra Args: ${EXTRA_ARGS[*]:-none}"
echo "   Fixed Coverage: ✅ Using .coveragerc"
echo "   Fixed Parsing: ✅ Enhanced error handling"
echo ""

# Ensure we have the coverage tool
if ! command -v coverage &> /dev/null; then
    echo "⚠️  Coverage tool not found. Installing..."
    pip install coverage
fi

# Run the fixed test runner
echo "🎯 Starting fixed test execution..."
python run_unified_tests_fixed.py --level "$LEVEL" "${EXTRA_ARGS[@]}"

# Show quick access commands
echo ""
echo "🔗 Quick Access Commands:"
if [[ -f "reports/coverage/html/index.html" ]]; then
    echo "   open reports/coverage/html/index.html  # Coverage report"
fi

if ls test_report_*.html 1> /dev/null 2>&1; then
    LATEST_REPORT=$(ls -t test_report_*.html | head -1)
    echo "   open \"$LATEST_REPORT\"              # Latest test report"
fi

echo ""
echo "✅ Fixed test execution complete!"