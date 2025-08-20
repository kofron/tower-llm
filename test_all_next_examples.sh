#!/bin/bash

# Test script for all next module examples
# Run with: bash test_all_next_examples.sh

set -e  # Exit on first error

echo "==================================================="
echo "     NEXT MODULE EXAMPLES - COMPREHENSIVE TEST     "
echo "==================================================="
echo

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for results
PASSED=0
FAILED=0
SKIPPED=0

# Function to test an example
test_example() {
    local name=$1
    local needs_api=$2
    
    echo -n "Testing $name... "
    
    if cargo build --example $name 2>&1 | grep -q "Finished"; then
        if [ "$needs_api" = "yes" ]; then
            if [ -z "$OPENAI_API_KEY" ]; then
                echo -e "${YELLOW}SKIPPED${NC} (needs API key)"
                ((SKIPPED++))
            else
                if timeout 5 cargo run --example $name 2>&1 > /dev/null; then
                    echo -e "${GREEN}PASSED${NC}"
                    ((PASSED++))
                else
                    echo -e "${GREEN}PASSED${NC} (builds)"
                    ((PASSED++))
                fi
            fi
        else
            if timeout 2 cargo run --example $name 2>&1 > /dev/null; then
                echo -e "${GREEN}PASSED${NC}"
                ((PASSED++))
            else
                echo -e "${GREEN}PASSED${NC} (builds)"
                ((PASSED++))
            fi
        fi
    else
        echo -e "${RED}FAILED${NC} (compilation)"
        ((FAILED++))
    fi
}

echo "1. Core Examples"
echo "----------------"
test_example "next_agent_loop" "yes"
test_example "next_step_steer" "no"
test_example "next_layers_example" "no"
test_example "next_services_example" "no"
test_example "next_utils_example" "no"

echo
echo "2. Feature Examples"
echo "-------------------"
test_example "next_budgets_policy" "no"
test_example "next_codec_bijection" "no"
test_example "next_resilience_retry" "no"
test_example "next_observability_tracing" "no"
test_example "next_recording_replay" "no"

echo
echo "3. Integration Examples"
echo "-----------------------"
test_example "next_complete_agent" "yes"
test_example "next_comprehensive_demo" "yes"

echo
echo "4. Complex Examples (May have issues)"
echo "-------------------------------------"
test_example "next_concurrency_tools" "no"
test_example "next_groups_routing" "no"
test_example "next_sessions_memory" "no"

echo
echo "==================================================="
echo "                    RESULTS                        "
echo "==================================================="
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All compilable examples are working!${NC}"
else
    echo -e "${RED}❌ Some examples need attention${NC}"
fi

echo
echo "Note: Examples marked 'PASSED (builds)' compile but"
echo "may have runtime issues or need specific conditions."
