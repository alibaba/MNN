#!/usr/bin/env python3
"""
MNN LLM Chat API Test Script
Supports multiple test scenarios
"""

import requests
import json
import sys
import time
import argparse
from typing import Dict, Any, List

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

class MnnApiTester:
    def __init__(self, host: str = "localhost", port: int = 8080, token: str = "mnn-llm-chat"):
        self.host = host
        self.port = port
        self.token = token
        self.base_url = f"http://{host}:{port}"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def test_models(self) -> Dict[str, Any]:
        """Test the /v1/models endpoint"""
        print("Testing /v1/models endpoint...")
        url = f"{self.base_url}/v1/models"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return {"success": True, "data": data}
            else:
                print(f"Error: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"Exception: {e}")
            return {"success": False, "error": str(e)}
    
    def test_chat(self, model: str = "qwen2.5-7b-instruct", message: str = "Hello") -> Dict[str, Any]:
        """Test the /v1/chat/completions endpoint"""
        print(f"Testing /v1/chat/completions endpoint (Model: {model})...")
        url = f"{self.base_url}/v1/chat/completions"
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return {"success": True, "data": result}
            else:
                print(f"Error: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"Exception: {e}")
            return {"success": False, "error": str(e)}
    
    def test_stream_chat(self, model: str = "qwen2.5-7b-instruct", message: str = "Hello") -> Dict[str, Any]:
        """Test the streaming chat endpoint"""
        print(f"Testing streaming chat endpoint (Model: {model})...")
        url = f"{self.base_url}/v1/chat/completions"
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": True
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30, stream=True)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                print("Streaming response:")
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data_str)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        print(delta['content'], end='', flush=True)
                            except json.JSONDecodeError:
                                pass
                print("\n")
                return {"success": True}
            else:
                print(f"Error: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"Exception: {e}")
            return {"success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(
        description='MNN LLM Chat API Test Tool',
        epilog=(
            "Examples:\n"
            "  1) PC -> phone over LAN (no adb forward):\n"
            "     python3 test_api.py --host 192.168.1.23 --port 8080 --test all\n"
            "  2) PC -> phone with adb forward:\n"
            "     adb forward tcp:8080 tcp:8080\n"
            "     python3 test_api.py --adb-forward --host 127.0.0.1 --port 8080 --test all"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Server address. Without --adb-forward, use phone LAN IP (for example 192.168.x.x).'
    )
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--token', default='mnn-llm-chat', help='Authentication token')
    parser.add_argument('--test', choices=['models', 'chat', 'stream', 'all'], default='all', help='Test type')
    parser.add_argument('--model', default='qwen2.5-7b-instruct', help='Test model')
    parser.add_argument('--message', default='Hello, please briefly introduce yourself', help='Test message')
    parser.add_argument(
        '--adb-forward',
        action='store_true',
        help='Set this when using adb forward and connecting to 127.0.0.1/localhost.'
    )
    
    args = parser.parse_args()
    
    print("MNN LLM Chat API Test Tool")
    print("=" * 50)

    normalized_host = args.host.strip().lower()
    is_loopback_host = normalized_host in LOOPBACK_HOSTS
    if not args.adb_forward and is_loopback_host:
        print("ERROR: localhost/127.0.0.1 requires adb forward.")
        print("Use phone LAN IP when testing from PC without adb forward.")
        print(f"Current host: {args.host}")
        print()
        print("Fix options:")
        print("  1) No adb forward: --host <PHONE_LAN_IP>")
        print(f"  2) Use adb forward: adb forward tcp:{args.port} tcp:{args.port} and add --adb-forward")
        return 2

    if args.adb_forward and not is_loopback_host:
        print(f"WARNING: --adb-forward is set but host is non-loopback ({args.host}).")
        print("This still works, but --adb-forward is usually used with 127.0.0.1/localhost.")

    connection_mode = "ADB forward (loopback)" if args.adb_forward else "LAN direct (no adb forward)"
    print(f"Connection mode: {connection_mode}")
    print(f"Server: {args.host}:{args.port}")
    print(f"Token: {args.token}")
    print(f"Test type: {args.test}")
    print()
    
    tester = MnnApiTester(args.host, args.port, args.token)
    
    results = []
    
    if args.test in ['models', 'all']:
        print("1. Test models endpoint")
        print("-" * 30)
        result = tester.test_models()
        results.append(('models', result))
        print()
    
    if args.test in ['chat', 'all']:
        print("2. Test chat endpoint")
        print("-" * 30)
        result = tester.test_chat(args.model, args.message)
        results.append(('chat', result))
        print()
    
    if args.test in ['stream', 'all']:
        print("3. Test streaming chat endpoint")
        print("-" * 30)
        result = tester.test_stream_chat(args.model, args.message)
        results.append(('stream', result))
        print()
    
    # Summary
    print("Test Summary:")
    for test_name, result in results:
        status = "Success" if result['success'] else "Failure"
        print(f"  {test_name}: {status}")
    
    all_success = all(result['success'] for _, result in results)
    if all_success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
