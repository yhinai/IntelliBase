#!/usr/bin/env python3
"""
Test FriendliAI Token Authentication
"""
import os
from dotenv import load_dotenv

def test_friendli_token():
    """Test FriendliAI token authentication"""
    try:
        from friendli import Friendli
        
        # Load environment variables
        load_dotenv('config.env')
        
        # Get token
        token = os.getenv("FRIENDLI_TOKEN")
        print(f"🔍 Testing FriendliAI token: {token[:10]}...")
        
        # Initialize client
        client = Friendli(token=token)
        
        # Test simple completion
        response = client.chat.completions.create(
            model="meta-llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        content = response.choices[0].message.content
        print(f"✅ FriendliAI working! Response: {content}")
        return True
        
    except Exception as e:
        print(f"❌ FriendliAI test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_friendli_token()
    exit(0 if success else 1) 