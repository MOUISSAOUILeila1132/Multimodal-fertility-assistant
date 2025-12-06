#  Modèles Utilisés

**Ce projet télécharge automatiquement les modèles suivants depuis Hugging Face lors de la première exécution.**

## 1. Vision & Langage (VLM)
- **Nom:** `Qwen/Qwen2-VL-7B-Instruct`
- **Type:** Vision-Language Model
- **Usage:** Analyse des images (spermogrammes, courbes) et génération de texte.
- **Configuration:** Chargé en 4-bit (NF4) via BitsAndBytes.

## 2. Embeddings (RAG)
- **Nom:** `intfloat/multilingual-e5-base`
- **Type:** Sentence Transformer
- **Usage:** Vectorisation des documents médicaux PDF et des questions utilisateurs pour la recherche sémantique.

## 3. Speech-to-Text (STT)
- **Nom:** `Systran/faster-whisper-medium`
- **Type:** Whisper (Implementation CTranslate2)
- **Usage:** Transcription de la voix (Arabe/Français) vers le texte.

## 4. Text-to-Speech (TTS)
- **Moteur:** Microsoft Edge TTS (Service Cloud)
- **Voix:** 
  - Arabe: `ar-SA-ZariyahNeural`
  - Français: `fr-FR-HenriNeural`