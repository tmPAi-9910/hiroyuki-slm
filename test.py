#!/usr/bin/env python3
"""
Test script for Hiroyuki SLM and API
"""

import requests
import json
import sys
import time
import argparse
import subprocess


API_BASE = "http://localhost:8080"


def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health ===")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        print(f"✓ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_models_info():
    """Test models info endpoint"""
    print("\n=== Testing /models/info ===")
    try:
        response = requests.get(f"{API_BASE}/models/info", timeout=5)
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Models info: {data}")
        return True
    except Exception as e:
        print(f"✗ Models info failed: {e}")
        return False


def test_exact_match():
    """Test exact match responses"""
    print("\n=== Testing Exact Match Responses ===")
    test_cases = [
        ("嘘", "嘘 related"),
        ("データ", "データ related"),
        ("学校", "学校 related"),
        ("Programming", None),  # Should use SLM
    ]
    
    success = True
    for message, expected_key in test_cases:
        try:
            response = requests.post(
                f"{API_BASE}/chat",
                json={"message": message},
                timeout=10
            )
            assert response.status_code == 200
            data = response.json()
            print(f"  Input: '{message}' -> Response: '{data['response'][:50]}...'")
        except Exception as e:
            print(f"  ✗ Failed for '{message}': {e}")
            success = False
    
    return success


def test_slm_generation():
    """Test SLM generation"""
    print("\n=== Testing SLM Generation ===")
    test_messages = [
        "こんにちは",
        "どう思いますか？",
        "プログラミングについて",
        "頭の悪い人について",
        "お前は馬鹿なのか？",
    ]
    
    success = True
    for message in test_messages:
        try:
            response = requests.post(
                f"{API_BASE}/chat",
                json={"message": message},
                timeout=15
            )
            assert response.status_code == 200
            data = response.json()
            print(f"  Input: '{message}'")
            print(f"  Output: '{data['response'][:60]}...'")
            print()
        except Exception as e:
            print(f"  ✗ Failed for '{message}': {e}")
            success = False
    
    return success


def test_error_cases():
    """Test error handling"""
    print("\n=== Testing Error Cases ===")
    
    # Missing message
    try:
        response = requests.post(f"{API_BASE}/chat", json={}, timeout=5)
        assert response.status_code == 400
        print("✓ Missing message correctly rejected")
    except Exception as e:
        print(f"✗ Missing message test failed: {e}")
        return False
    
    # Invalid message type
    try:
        response = requests.post(f"{API_BASE}/chat", json={"message": 123}, timeout=5)
        assert response.status_code == 400
        print("✓ Invalid message type correctly rejected")
    except Exception as e:
        print(f"✗ Invalid message type test failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Hiroyuki SLM API Test Suite")
    print("=" * 50)
    
    # Wait for server to be ready
    print("\nWaiting for API server...")
    max_retries = 10
    for i in range(max_retries):
        try:
            requests.get(f"{API_BASE}/health", timeout=2)
            print("API server is ready!")
            break
        except:
            if i < max_retries - 1:
                time.sleep(1)
            else:
                print("ERROR: API server not responding")
                print("Make sure the server is running: python api.py")
                sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Health", test_health()))
    results.append(("Models Info", test_models_info()))
    results.append(("Exact Match", test_exact_match()))
    results.append(("SLM Generation", test_slm_generation()))
    results.append(("Error Cases", test_error_cases()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test script for Hiroyuki SLM and API')
    parser.add_argument('--standalone', action='store_true', help='Run API server, test, and shutdown')
    parser.add_argument('--api-base', default='http://localhost:8080', help='API base URL')
    args = parser.parse_args()
    
    API_BASE = args.api_base
    
    if args.standalone:
        print("Starting API server in standalone mode...")
        server_process = subprocess.Popen([sys.executable, 'api.py'])
        time.sleep(5)  # Wait for server to start
        try:
            success = run_all_tests()
        finally:
            print("Shutting down API server...")
            server_process.terminate()
            server_process.wait()
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
