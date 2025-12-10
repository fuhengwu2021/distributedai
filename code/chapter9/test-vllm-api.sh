#!/bin/bash
# Test script for vLLM OpenAI-compatible API

set -e

API_URL="${1:-http://localhost:8000}"
echo "Testing vLLM API at: $API_URL"
echo "=================================="
echo ""

# Test health
echo "1. Testing health endpoint..."
echo "   GET $API_URL/health"
HEALTH_RESPONSE=$(curl --max-time 5 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/health" || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$HEALTH_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Health check passed"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ❌ Health check failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi
echo ""

# List models
echo "2. Listing available models..."
echo "   GET $API_URL/v1/models"
MODELS_RESPONSE=$(curl --max-time 5 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/v1/models" || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$MODELS_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$MODELS_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Models listed successfully"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ❌ List models failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi
echo ""

# Test completion
echo "3. Testing completion endpoint..."
echo "   POST $API_URL/v1/completions"
COMPLETION_RESPONSE=$(curl --max-time 30 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 50,
    "temperature": 0.7
  }' || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$COMPLETION_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$COMPLETION_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Completion successful"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ❌ Completion failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi
echo ""

# Test chat (if supported)
echo "4. Testing chat completion endpoint..."
echo "   POST $API_URL/v1/chat/completions"
CHAT_RESPONSE=$(curl --max-time 30 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [
      {"role": "user", "content": "Hello! Can you explain what AI is in one sentence?"}
    ],
    "max_tokens": 50
  }' || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$CHAT_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$CHAT_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Chat completion successful"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ⚠️  Chat completion failed (HTTP $HTTP_CODE) - may not be supported by this model"
    echo "$BODY"
fi
echo ""

echo "=================================="
echo "Testing complete!"
