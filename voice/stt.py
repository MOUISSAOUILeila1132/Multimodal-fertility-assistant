import os
import torch
from faster_whisper import WhisperModel #moteur STT basé sur whisper
from pydub import AudioSegment #pour charger l'audio 

class STTEngine:
    def __init__(self, model_size="medium", device="cuda"):
        
        
        print(f" STT : Chargement de Whisper ({model_size}) sur GPU...")
        try:
            #chargement du modele whisper sur GPU en float16
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print(" Whisper GPU chargé.")
        except Exception as e:
            print(f" GPU indisponible, fallback CPU : {e}")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path):
        #une chaine vide renvoyeé si aucun fichier audio n'est fourni
        if not audio_path: return ""

        #fichier temporaire netoyee
        clean_path = "/content/temp_clean.wav"
        try:
            #chargement et normalisation de l'audio
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(clean_path, format="wav")
        except:
            clean_path = audio_path

        try:
            #transcription avec beam search et sans VAD
            #beam search pour generer le text , les 5 meilleurs hypotheses de transcription (beam_size=5)
            segments, _ = self.model.transcribe(clean_path, beam_size=5, vad_filter=False)
            #concatenation du text 
            text = " ".join([s.text for s in segments]).strip()
            return text
        except Exception as e:
            print(f" Erreur STT : {e}")
            return ""
