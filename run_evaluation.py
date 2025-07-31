#!/usr/bin/env python3
"""
Enhanced ECHR QA evaluation script with command line arguments
"""

import argparse
import sys
import os
import requests
import json
import subprocess
from test_echr_qa import ECHRTestEvaluator

def check_dependencies():
    """Check required Python dependencies"""
    required_packages = ['sentence_transformers', 'pandas', 'transformers', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        install = input("Install missing packages? (y/n): ").lower() == 'y'
        if install:
            for package in missing_packages:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package])
        else:
            print("❌ Cannot proceed without required packages")
            sys.exit(1)

def check_retrieval_server(server_url):
    """Check if retrieval server is running"""
    try:
        response = requests.get(f"{server_url}/info", timeout=5)
        if response.status_code == 200:
            print("✅ Retrieval server is running")
            server_info = response.json()
            print("📊 Retriever info:")
            print(json.dumps(server_info, indent=2))
            return True
    except:
        pass
    
    try:
        test_payload = {"queries": ["test"], "topk": 1}
        response = requests.post(f"{server_url}/retrieve", json=test_payload, timeout=10)
        if response.status_code == 200:
            print("✅ Retrieval server is running (basic check)")
            return True
    except:
        pass
    
    print("❌ Retrieval server is not running!")
    print("Please start the retrieval server first:")
    print("  bash retrieval_launch.sh")
    return False

def check_required_files():
    """Check if required files exist"""
    required_files = [
        "data/echr_qa/echr_qa_compact.json",
        "data/echr_guide.jsonl",
        "inference_engine.py",
        "test_echr_qa.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    print("✅ All required files found")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run ECHR QA Evaluation")
    
    # Model configuration
    parser.add_argument("--model", type=str, 
                       default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3",
                       help="Model ID to use for evaluation")
    
    # Test configuration  
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of samples to test (default: 3)")
    parser.add_argument("--topk", type=int, default=15,
                       help="TopK for retrieval (default: 15)")
    
    # Data paths
    parser.add_argument("--qa-data", type=str, 
                       default="data/echr_qa/echr_qa_compact.json",
                       help="Path to ECHR QA data")
    parser.add_argument("--guide-data", type=str,
                       default="data/echr_guide.jsonl", 
                       help="Path to ECHR guide data")
    parser.add_argument("--results-dir", type=str,
                       default="test_results",
                       help="Directory to save results")
    
    # Server configuration
    parser.add_argument("--server-url", type=str,
                       default="http://127.0.0.1:8000",
                       help="Retrieval server URL")
    
    # Quick test presets
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 2 samples")
    parser.add_argument("--full", action="store_true", 
                       help="Full test with all samples")
    parser.add_argument("--medium", action="store_true",
                       help="Medium test with 10 samples")
    
    # Model presets
    parser.add_argument("--model-14b", action="store_true",
                       help="Use 14B model instead of 32B")
    
    # Model optimization options
    parser.add_argument("--no-preload", action="store_true",
                       help="Disable model preloading (load model for each sample)")
    parser.add_argument("--no-quantization", action="store_true",
                       help="Disable model quantization")
    parser.add_argument("--quantization-bits", type=int, choices=[4, 8], default=8,
                       help="Quantization bits (4 or 8, default: 8)")
    
    # System checks
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip system checks")
    
    # Evaluation configuration
    parser.add_argument("--evaluation-topk", type=int, default=200,
                       help="TopK for evaluation (finding target paragraphs, default: 200)")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.quick:
        args.samples = 2
    elif args.medium:
        args.samples = 10  
    elif args.full:
        args.samples = -1  # Will use all available samples
        
    if args.model_14b:
        args.model = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3"
    
    print("=== ECHR QA Evaluation Configuration ===")
    print(f"Model: {args.model}")
    print(f"Samples: {'All' if args.samples == -1 else args.samples}")
    print(f"TopK for model: {args.topk}")
    print(f"TopK for evaluation: {args.evaluation_topk}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Server URL: {args.server_url}")
    print("=" * 50)
    
    # System checks
    if not args.skip_checks:
        print("\n🔍 Running system checks...")
        
        # Check dependencies
        check_dependencies()
        
        # Check required files
        if not check_required_files():
            sys.exit(1)
        
        # Check retrieval server
        if not check_retrieval_server(args.server_url):
            sys.exit(1)
        
        print("✅ All system checks passed")
    
    # Initialize evaluator
    try:
        evaluator = ECHRTestEvaluator(
            echr_qa_path=args.qa_data,
            echr_guide_path=args.guide_data,
            results_dir=args.results_dir,
            preload_model=not args.no_preload,
            model_id=args.model,
            use_quantization=not args.no_quantization,
            quantization_bits=args.quantization_bits,
            evaluation_topk=args.evaluation_topk
        )
        
        # Determine actual sample count
        total_samples = len(evaluator.qa_data)
        if args.samples == -1:
            max_samples = total_samples
        else:
            max_samples = min(args.samples, total_samples)
            
        print(f"\n📊 Total available samples: {total_samples}")
        print(f"📝 Will test: {max_samples} samples")
        print()
        
        # Run evaluation
        results = evaluator.run_evaluation(
            model_id=args.model,
            max_samples=max_samples,
            topk=args.topk,
            retrieval_server_url=args.server_url
        )
        
        print("\n✅ Evaluation completed successfully!")
        
        # Show latest results files
        print(f"\n📁 Results saved in: {args.results_dir}")
        result_files = [f for f in os.listdir(args.results_dir) if f.endswith('.csv') or f.endswith('.json')]
        if result_files:
            print("📄 Generated files:")
            for f in sorted(result_files)[-5:]:  # Show last 5 files
                print(f"  - {f}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
