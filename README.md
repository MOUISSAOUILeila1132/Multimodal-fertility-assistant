# 🌸 Tanit - Multimodal Fertility Assistant

**Tanit** is an AI-powered medical assistant designed to help interpret fertility-related documents (Spermograms, Hormonal Analysis, Ultrasounds) and answer questions in both **Arabic** and **French**.

It leverages a multimodal architecture combining Computer Vision, RAG (Retrieval-Augmented Generation), and Voice Interaction.

##  Key Features
- **Multimodal Analysis:** Understands medical images (charts, tables) using `Qwen2-VL-7B`.
- **Bilingual Support:** Switch natively between Arabic (with dialect support) and French.
- **RAG Engine:** Contextual answers based on verified medical PDFs using `LlamaIndex`.
- **Voice Interface:** Full Speech-to-Text (`Faster-Whisper`) and Text-to-Speech (`Edge-TTS`).
- **Medical Guardrails:** Strict anti-hallucination prompts and deterministic generation settings.

---

##  Architecture & Tech Stack

| Component | Technology | Details |
|-----------|------------|---------|
| **VLM (Vision-LLM)** | **Qwen2-VL-7B-Instruct** | 4-bit Quantization (NF4) via `bitsandbytes` |
| **Embeddings** | **Multilingual-E5-Base** | `intfloat/multilingual-e5-base` |
| **RAG Framework** | **LlamaIndex** | Vector Store Index |
| **STT (Input)** | **Faster-Whisper** | GPU-accelerated transcription |
| **TTS (Output)** | **Edge-TTS** | Neural voices (Azure) |
| **UI** | **Gradio** | Web Interface with Chatbot & Audio |

---

##  Exact Kaggle Run Instructions

Follow these steps to run the prototype on a **Kaggle Notebook**.

### 1. Environment Setup
*   Open a new Notebook on Kaggle.
*   **IMPORTANT:** Go to the **Settings** panel (right side) and set **Accelerator** to **GPU T4 x2** (or P100). The model requires GPU VRAM.
*   Internet: **On**.

### 2. Install Dependencies
Copy and run this code in the first cell to install all required libraries:

```python
!pip install transformers accelerate qwen-vl-utils bitsandbytes safetensors
!pip install faster-whisper
!pip install edge-tts nest_asyncio
!pip install gradio
!pip install llama-index llama-index-embeddings-huggingface sentence-transformers
!pip install pydub
# FFmpeg is usually pre-installed on Kaggle/Colab, if not:
# !apt-get install -y ffmpeg



pip install -r requirements.txt

python build_index.py

python app.py
