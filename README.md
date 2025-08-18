# 🎯 Find My Intent

An AI-powered intent classification app using state-of-the-art BGE (BAAI General Embedding) models for superior semantic understanding.

## 🚀 What's New

### BGE-large-en-v1.5 Integration
- **State-of-the-art Performance**: Now powered by BGE-large-en-v1.5, consistently ranking #1 on MTEB leaderboard
- **Superior Semantic Understanding**: 1024-dimensional embeddings capture nuanced intent differences
- **Multi-Model Support**: Choose between BGE-large, BGE-base, MPNet, or MiniLM based on your needs
- **Optimized Thresholds**: Fine-tuned confidence levels for BGE's superior discrimination

## ✨ Key Features

- **Advanced Intent Classification**: Uses BGE, E5, or other state-of-the-art models for best-in-class semantic similarity
- **Multi-Model Support**: Choose from 5 models based on your speed/accuracy requirements
- **250+ Pre-loaded Intents**: Comprehensive Vanguard financial services intent library
- **Intent Management**: Add, edit, delete intents directly in the UI
- **Batch Processing**: Upload files for bulk classification with progress tracking
- **Intent Similarity Analysis**: Find and resolve confusing intent pairs
- **Coverage Analysis**: Test multiple utterances to identify gaps
- **Real-time Testing**: Test intents and see confidence scores instantly
- **Visual Analytics**: Confidence distribution bars and color-coded results
- **Export/Import**: Full intent library management with merge options

## 🤖 Available Models

| Model | Quality | Speed | Size | Best For |
|-------|---------|-------|------|----------|
| **BGE-large-en-v1.5** | 🟢🟢 State-of-the-Art | 🟠 Moderate | ~1.3GB | Production, highest accuracy |
| **BGE-base-en-v1.5** | 🟢 Excellent | 🟡 Good | ~440MB | Balanced performance |
| **E5-base-v2** | 🟢 Excellent | 🟡 Moderate | ~440MB | Semantic search, retrieval |
| **MPNet-base-v2** | 🟢 Very Good | 🟡 Moderate | ~420MB | General purpose, proven reliability |
| **MiniLM-L6-v2** | 🟡 Good | ⚡ Very Fast | ~90MB | Resource-constrained environments |

## 🛠️ Technology Stack

- **Frontend**: Streamlit with enhanced UI/UX
- **AI/ML**: Sentence Transformers with BGE models
- **Backend**: Python 3.11+
- **Deployment**: Streamlit Community Cloud / GitHub Codespaces
- **Model Hub**: Hugging Face Model Hub

## 📋 Requirements

- Python 3.9+
- 8GB RAM recommended (for BGE-large)
- Internet connection (for initial model download)
- ~2GB disk space for models

## 🏃‍♂️ Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd find-my-intent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **First run**: BGE-large will download (~1.3GB), this is a one-time process

### GitHub Codespaces

1. Click "Code" → "Codespaces" → "Create codespace"
2. Select at least 4-core machine for optimal performance
3. Wait for setup to complete (~3-5 minutes including model download)
4. App will automatically start on port 8501

### Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from your GitHub repo
4. Note: First deployment may take longer due to model download

## 🎯 Usage Guide

### Single Classification
1. Navigate to "🎯 Classify" tab
2. Try example utterances or enter your own text
3. Click "Classify" to see results with confidence distribution
4. View top 5 matching intents with visual confidence bars

### Batch Processing
1. Prepare a .txt or .csv file (one utterance per line)
2. Upload in the "Batch Upload" section
3. Click "Process Batch" to classify all utterances
4. Download results as CSV

### Model Selection
1. Open the sidebar
2. Choose from available models based on your needs
3. BGE-large recommended for best accuracy
4. BGE-base for balanced performance
5. MiniLM for fastest inference

### Intent Management
1. Go to "⚙️ Manage" tab
2. **Add Intent**: Click "Add Intent" expander, enter name and description
3. **Edit Intent**: Click edit mode checkbox, modify description, save changes
4. **Delete Intent**: Enable edit mode, click delete button, confirm deletion
5. **Search**: Use search bar to filter intents
6. **Test**: Enter utterances to test against specific intents
7. **Export/Import**: Save intent library or merge with new intents

## 📊 Performance Metrics

### BGE-large-en-v1.5
- **First load**: 60-90 seconds (1.3GB download)
- **Subsequent loads**: 5-10 seconds (cached)
- **Classification speed**: ~100-200ms per utterance
- **Accuracy**: State-of-the-art on MTEB benchmarks
- **Memory usage**: ~2-3GB RAM

### BGE-base-en-v1.5 / E5-base-v2
- **First load**: 30-45 seconds (~440MB download)
- **Subsequent loads**: 3-5 seconds (cached)
- **Classification speed**: ~50-100ms per utterance
- **Accuracy**: Excellent, top-tier performance
- **Memory usage**: ~1-1.5GB RAM

### MPNet-base-v2
- **First load**: 30-45 seconds (~420MB download)
- **Subsequent loads**: 3-5 seconds (cached)
- **Classification speed**: ~50-100ms per utterance
- **Accuracy**: Very good, proven reliability
- **Memory usage**: ~1-1.5GB RAM

### MiniLM-L6-v2
- **First load**: 10-15 seconds (~90MB download)
- **Subsequent loads**: 1-2 seconds (cached)
- **Classification speed**: ~10-30ms per utterance
- **Accuracy**: Good for lightweight applications
- **Memory usage**: ~400-600MB RAM

### Optimization Features
- Model caching with `@st.cache_resource`
- Pre-computed intent embeddings
- Normalized embeddings for faster similarity
- Batch processing with progress tracking

## 🔧 Configuration

### Classification Thresholds (BGE-optimized)
```python
{
    "high_confidence": 0.6,    # Very confident match
    "medium_confidence": 0.4,   # Reasonable match
    "low_confidence": 0.2,      # Possible match
    "no_match": 0.2            # Below threshold
}
```

### Model-Specific Settings
- **BGE Models**: 
  - Max sequence length: 512 tokens
  - Query prefix: "Represent this sentence for retrieval:"
  - Normalized embeddings: Enabled
  - Dimensions: 1024 (large) or 768 (base)

- **E5 Models**:
  - Max sequence length: 512 tokens
  - Query prefix: "query:"
  - Passage prefix: "passage:"
  - Normalized embeddings: Enabled
  - Dimensions: 768

- **MPNet/MiniLM**:
  - No special prefixes required
  - Standard sentence transformer settings
  - Dimensions: 768 (MPNet) or 384 (MiniLM)

## 📁 File Structure

```
├── app.py                     # Main application with BGE support
├── requirements.txt           # Python dependencies
├── vanguard_intents.json     # Intent definitions
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── .devcontainer/
│   └── devcontainer.json     # Development container setup
└── README.md                 # This file
```

## 🚀 Deployment Tips

### For Streamlit Cloud
- Ensure adequate resources (at least 1GB RAM)
- First deployment will be slow (model download)
- Consider using secrets.toml for API keys

### For Production
- Use BGE-large for best accuracy
- Pre-download models to avoid startup delay
- Consider GPU deployment for high volume
- Implement caching layer for repeated queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test with different models
5. Submit a pull request

## 📈 Benchmarks

Model performance on MTEB (Massive Text Embedding Benchmark):
- **BGE-large-en-v1.5**: #1 for retrieval tasks, state-of-the-art performance
- **BGE-base-en-v1.5**: Top 5 performer, excellent balance
- **E5-base-v2**: Top tier for semantic search, excels in zero-shot scenarios
- **MPNet-base-v2**: Consistent top 10, recommended by SBERT
- **MiniLM-L6-v2**: Good performance for its size, 5x faster than larger models

Expected improvements over MiniLM baseline:
- **BGE-large**: 20-25% accuracy improvement
- **BGE-base/E5**: 15-20% accuracy improvement  
- **MPNet**: 10-15% accuracy improvement

## 🆘 Troubleshooting

### Model Download Issues
- Check internet connection
- Ensure sufficient disk space (2GB+ for BGE-large, 500MB+ for others)
- Try BGE-base or E5-base if large model fails
- Clear Hugging Face cache if needed: `~/.cache/huggingface/`

### Memory Issues
- Use BGE-base instead of large
- Increase swap space on Linux
- Close other applications
- Consider batch processing for large datasets

### Slow Performance
- Model loading is slow only on first run
- Use MiniLM for testing/development
- Enable GPU if available
- Reduce batch size for processing

## 🔗 Resources

- [BGE Models on Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [E5 Models on Hugging Face](https://huggingface.co/intfloat/e5-base-v2)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence Transformers Docs](https://www.sbert.net)
- [Streamlit Documentation](https://docs.streamlit.io)

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- BAAI for BGE models
- Microsoft for E5 models
- Sentence Transformers community
- Streamlit for the web framework
- Hugging Face for model hosting

---

**Note**: BGE and E5 models require more resources but provide significantly better intent classification accuracy. For resource-constrained environments, consider using MPNet for a good balance or MiniLM for fastest performance.