#!/usr/bin/env python
"""
Run the FastAPI server for LSTM Traffic Prediction
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uvicorn

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Starting LSTM Traffic Prediction API")
    print("="*70)
    print("\n📍 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n✅ Ready for Postman testing!")
    print("="*70 + "\n")
    
    uvicorn.run(
        "predict_load_model:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
