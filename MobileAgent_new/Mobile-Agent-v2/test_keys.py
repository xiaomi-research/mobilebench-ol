import sys
sys.path.insert(0, '/home/mi/MyProj/mobilebench-test-copy/MobileAgent/Mobile-Agent-v2')

from MobileAgent.api import inference_chat

# Azure OpenAI configuration
API_url = "https://ui-agent-exp.openai.azure.com"
key1 =  ""
key2 =  ""

# Test chat messages
test_chat = [
    ("user", "Hello, please respond with 'Key working!' if you receive this message.")
]

print("=" * 60)
print("Testing Key 1 with gpt-4o")
print("=" * 60)
try:
    response1 = inference_chat(test_chat, "gpt-4o", API_url, key1)
    print(f"✅ Key 1 Success!")
    print(f"Response: {response1}")
except Exception as e:
    print(f"❌ Key 1 Failed: {e}")

print("\n" + "=" * 60)
print("Testing Key 2 with gpt-4o")
print("=" * 60)
try:
    response2 = inference_chat(test_chat, "gpt-4o", API_url, key2)
    print(f"✅ Key 2 Success!")
    print(f"Response: {response2}")
except Exception as e:
    print(f"❌ Key 2 Failed: {e}")

print("\n" + "=" * 60)
print("Testing Key 1 with gpt-4o-mini")
print("=" * 60)
try:
    response3 = inference_chat(test_chat, "gpt-4o-mini", API_url, key1)
    print(f"✅ Key 1 with gpt-4o-mini Success!")
    print(f"Response: {response3}")
except Exception as e:
    print(f"❌ Key 1 with gpt-4o-mini Failed: {e}")

print("\n" + "=" * 60)
print("Testing Key 2 with gpt-4o-mini")
print("=" * 60)
try:
    response4 = inference_chat(test_chat, "gpt-4o-mini", API_url, key2)
    print(f"✅ Key 2 with gpt-4o-mini Success!")
    print(f"Response: {response4}")
except Exception as e:
    print(f"❌ Key 2 with gpt-4o-mini Failed: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
