#!/usr/bin/env python3
from rag_service.rag_pipeline import RAGPipeline
from rag_service.exceptions import ConfigurationError

# Test invalid configuration
try:
    pipeline = RAGPipeline(config={'vector_store': {'dimension': -1}})
    print('❌ No error raised for invalid dimension')
except ConfigurationError as e:
    print(f'✅ ConfigurationError raised: {e}')
except Exception as e:
    print(f'❌ Wrong error type: {type(e).__name__}: {e}')

# Test invalid k value
try:
    pipeline = RAGPipeline(config={'retriever': {'k': 0}})
    print('❌ No error raised for invalid k')
except ConfigurationError as e:
    print(f'✅ ConfigurationError raised: {e}')
except Exception as e:
    print(f'❌ Wrong error type: {type(e).__name__}: {e}')

# Test invalid temperature
try:
    pipeline = RAGPipeline(config={'qa_service': {'temperature': 2.0}})
    print('❌ No error raised for invalid temperature')
except ConfigurationError as e:
    print(f'✅ ConfigurationError raised: {e}')
except Exception as e:
    print(f'❌ Wrong error type: {type(e).__name__}: {e}')

print('Configuration validation test complete')
