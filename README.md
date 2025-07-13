# 🎯 Find My Intent

An AI-powered intent classification app using semantic similarity with Sentence Transformers.

## 🚀 Features

- **Smart Intent Classification**: Uses state-of-the-art sentence transformers for semantic understanding
- **250+ Pre-loaded Intents**: Comprehensive Vanguard financial services intent library
- **Batch Processing**: Upload files for bulk classification
- **Intent Management**: Add, edit, and remove intents through the UI
- **Performance Optimized**: Cached models and embeddings for fast response times

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Sentence Transformers (all-MiniLM-L6-v2)
- **Backend**: Python 3.11+
- **Deployment**: Streamlit Community Cloud / GitHub Codespaces

## 📋 Requirements

- Python 3.9+
- 4GB RAM recommended
- Internet connection (for initial model download)

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

### GitHub Codespaces

1. Click "Code" → "Codespaces" → "Create codespace"
2. Wait for setup to complete (~2-3 minutes)
3. App will automatically start on port 8501

### Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from your GitHub repo

## 🎯 Usage

### Single Classification
1. Navigate to "Classify Utterances" tab
2. Enter your text (e.g., "What's my IRA balance?")
3. Click "Classify" to see the best matching intent

### Batch Classification
1. Select "Batch Upload" 
2. Upload a .txt or .csv file (one utterance per line)
3. Process multiple utterances at once

### Intent Management
1. Go to "Manage Intents" tab
2. Search, edit, or add new intents
3. Changes are automatically saved

## 📊 Performance

- **First load**: 30-60 seconds (model download)
- **Subsequent loads**: 2-5 seconds (cached)
- **Classifications**: Near-instant
- **Supported**: 250+ intents, unlimited utterances

## 🔧 Configuration

The app uses smart caching to optimize performance:
- Model loading is cached globally
- Intent embeddings are pre-computed and cached
- UI state is efficiently managed

## 📁 File Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── vanguard_intents.json # Intent definitions
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── .devcontainer/
│   └── devcontainer.json # Development container setup
└── README.md             # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Slow Initial Load
- Normal on first run (downloading ~90MB model)
- Subsequent loads are much faster due to caching

### Memory Issues
- Ensure at least 4GB RAM available
- Close other applications if needed

### Model Loading Errors
- Check internet connection
- Verify requirements.txt dependencies

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Sentence Transformers](https://www.sbert.net)
- [Hugging Face Models](https://huggingface.co/sentence-transformers)