#!/bin/bash

set -e

echo "=== ML Pipeline API Test Suite ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
HEALTH=$(curl -s ${API_URL}/health)
STATUS=$(echo $HEALTH | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")

if [ "$STATUS" = "healthy" ]; then
    echo -e "${GREEN}✓ Health check passed (status: $STATUS)${NC}"
    MODEL_LOADED=$(echo $HEALTH | python3 -c "import sys, json; print(json.load(sys.stdin)['model_loaded'])")
    DB_CONNECTED=$(echo $HEALTH | python3 -c "import sys, json; print(json.load(sys.stdin)['database_connected'])")
    echo "  - Model loaded: $MODEL_LOADED"
    echo "  - Database connected: $DB_CONNECTED"
else
    echo -e "${RED}✗ Health check failed: $STATUS${NC}"
    exit 1
fi
echo ""

# Test 2: Model Info
echo -e "${YELLOW}Test 2: Model Information${NC}"
MODEL_INFO=$(curl -s ${API_URL}/)
MODEL_NAME=$(echo $MODEL_INFO | python3 -c "import sys, json; print(json.load(sys.stdin)['model_info']['model_name'])")

if [ "$MODEL_NAME" = "distilbert-base-uncased" ]; then
    echo -e "${GREEN}✓ Model info retrieved${NC}"
    echo "  - Model: $MODEL_NAME"
else
    echo -e "${RED}✗ Model info failed${NC}"
    exit 1
fi
echo ""

# Test 3: Single Prediction
echo -e "${YELLOW}Test 3: Single Prediction${NC}"
echo '{"text": "This movie is amazing!"}' > /tmp/test_request.json
PREDICTION=$(curl -s -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/test_request.json)

SENTIMENT=$(echo $PREDICTION | python3 -c "import sys, json; print(json.load(sys.stdin).get('predicted_sentiment', 'ERROR'))")
CONFIDENCE=$(echo $PREDICTION | python3 -c "import sys, json; print(json.load(sys.stdin).get('confidence', 0))")

if [ -n "$SENTIMENT" ] && [ "$SENTIMENT" != "ERROR" ]; then
    echo -e "${GREEN}✓ Single prediction passed${NC}"
    echo "  - Sentiment: $SENTIMENT"
    echo "  - Confidence: $CONFIDENCE"
else
    echo -e "${RED}✗ Single prediction failed${NC}"
    exit 1
fi
echo ""

# Test 4: Batch Prediction
echo -e "${YELLOW}Test 4: Batch Prediction${NC}"
cat > /tmp/batch_request.json << 'EOF'
{
  "texts": ["Great movie!", "Terrible film.", "It was okay."]
}
EOF

BATCH=$(curl -s -X POST ${API_URL}/predict/batch \
  -H "Content-Type: application/json" \
  -d @/tmp/batch_request.json)

TOTAL=$(echo $BATCH | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_predictions', 0))")

if [ "$TOTAL" = "3" ]; then
    echo -e "${GREEN}✓ Batch prediction passed${NC}"
    echo "  - Total predictions: $TOTAL"
else
    echo -e "${RED}✗ Batch prediction failed${NC}"
    exit 1
fi
echo ""

# Test 5: Error Handling
echo -e "${YELLOW}Test 5: Error Handling${NC}"
ERROR=$(curl -s -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{"wrong_field": "value"}' \
  -o /dev/null -w "%{http_code}")

if [ "$ERROR" = "422" ]; then
    echo -e "${GREEN}✓ Error handling passed${NC}"
    echo "  - Returns HTTP 422 for invalid input"
else
    echo -e "${RED}✗ Error handling failed: HTTP $ERROR${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}=== All Tests Passed ===${NC}"
echo ""
echo "Summary:"
echo "  ✓ Health check"
echo "  ✓ Model info"
echo "  ✓ Single prediction"
echo "  ✓ Batch prediction"
echo "  ✓ Error handling"
echo ""
echo "API is ready for use!"

# Cleanup
rm -f /tmp/test_request.json /tmp/batch_request.json
