# IntelliBase Complete Setup Guide

## ⚠️ **Current Status: PARTIALLY WORKING**

The system is **NOT** working 100% due to missing API keys and external services. Here's what needs to be fixed:

## 🔑 **Missing API Keys & Configuration**

### 1. **FriendliAI Setup**
```bash
# Get your FriendliAI token from: https://console.friendli.ai/
# Add to config.env:
FRIENDLI_TOKEN=your_actual_friendli_token_here
```

### 2. **OpenAI Setup (Fallback)**
```bash
# Get your OpenAI API key from: https://platform.openai.com/api-keys
# Add to config.env:
OPENAI_API_KEY=your_actual_openai_key_here
```

### 3. **Weaviate Setup**
You have two options:

#### Option A: Use Weaviate Cloud (Recommended)
```bash
# Sign up at: https://console.weaviate.cloud/
# Get your cluster URL and API key
# Add to config.env:
WEAVIATE_CLUSTER_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your_actual_weaviate_api_key_here
```

#### Option B: Run Weaviate Locally
```bash
# Install Docker if not already installed
# Run Weaviate locally:
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e DEFAULT_VECTORIZER_MODULE='none' \
  -e ENABLE_MODULES='text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai' \
  -e CLUSTER_HOSTNAME='node1' \
  semitechnologies/weaviate:1.22.4

# Then use in config.env:
WEAVIATE_CLUSTER_URL=http://localhost:8080
WEAVIATE_API_KEY=your_local_api_key_here
```

### 4. **Hypermode Setup (Optional)**
```bash
# Get your Hypermode API key from: https://hypermode.com/
# Add to config.env:
HYPERMODE_API_KEY=your_actual_hypermode_key_here
```

## 🚀 **Complete Setup Steps**

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
# Copy and edit config.env with your actual API keys
cp config.env config.env.backup
# Edit config.env with your real API keys
```

### Step 3: Start Weaviate (if using local)
```bash
# Start Weaviate server
docker run -d --name weaviate -p 8080:8080 semitechnologies/weaviate:1.22.4
```

### Step 4: Test Configuration
```bash
# Run integration tests
python test_integration.py

# Run comprehensive system test
python comprehensive_system_test.py
```

### Step 5: Start Streamlit UI
```bash
streamlit run streamlit_ui.py
```

## 🔍 **Current Issues & Solutions**

### Issue 1: Weaviate Connection Refused
**Problem**: `Connection to Weaviate failed. Details: Error: [Errno 61] Connection refused`

**Solutions**:
1. **Use Weaviate Cloud** (easiest):
   - Sign up at https://console.weaviate.cloud/
   - Get free cluster with API key
   - Update config.env

2. **Run Weaviate Locally**:
   ```bash
   docker run -d --name weaviate -p 8080:8080 semitechnologies/weaviate:1.22.4
   ```

### Issue 2: Phoenix Port Conflicts
**Problem**: `bind: Address already in use (48)`

**Solutions**:
1. **Kill existing Phoenix processes**:
   ```bash
   pkill -f phoenix
   ```

2. **Use different ports**:
   ```bash
   export PHOENIX_PORT=6007
   ```

### Issue 3: Missing API Keys
**Problem**: All AI services using placeholder tokens

**Solutions**:
1. **Get FriendliAI token**: https://console.friendli.ai/
2. **Get OpenAI key**: https://platform.openai.com/api-keys
3. **Get Weaviate access**: https://console.weaviate.cloud/

## 📊 **What's Actually Working vs. Not Working**

### ✅ **Working Components**:
- Plotly visualizations
- Streamlit UI framework
- Daft data processing
- System testing framework
- Basic component structure

### ❌ **Not Working Components**:
- **Weaviate vector database** (no connection)
- **FriendliAI LLM** (no API key)
- **OpenAI fallback** (no API key)
- **Phoenix observability** (port conflicts)
- **Persistent storage** (falling back to in-memory)

### ⚠️ **Partially Working**:
- **System tests** (passing but using mock data)
- **UI components** (showing but not functional)
- **Data processing** (working but no persistent storage)

## 🧪 **Testing the Real System**

### Before API Keys (Current State):
```bash
# This works but uses mock data
python comprehensive_system_test.py
# Result: 100% success rate (but with mock components)
```

### After API Keys (Target State):
```bash
# This will test real components
python comprehensive_system_test.py
# Expected: Real AI responses, persistent storage, actual vector search
```

## 💰 **Cost Estimates for API Keys**

### Free Tiers Available:
- **Weaviate Cloud**: Free tier with 25GB storage
- **OpenAI**: Free tier with $5 credit
- **FriendliAI**: Free tier available
- **Hypermode**: Free tier available

### Total Cost for Full Setup: ~$0-20/month

## 🎯 **Next Steps to Get 100% Working**

1. **Get Weaviate Cloud account** (free)
2. **Get FriendliAI token** (free tier)
3. **Get OpenAI API key** (free $5 credit)
4. **Update config.env** with real keys
5. **Test with real data**
6. **Verify all components working**

## 📞 **Support Resources**

- **Weaviate Docs**: https://weaviate.io/developers/weaviate
- **FriendliAI Docs**: https://docs.friendli.ai/
- **OpenAI Docs**: https://platform.openai.com/docs
- **Phoenix Docs**: https://docs.arize.com/phoenix/

---

**Current Status**: System framework is complete, but needs API keys and external services to be fully functional. 