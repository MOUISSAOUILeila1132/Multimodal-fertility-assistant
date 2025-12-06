import edge_tts #bibliotheque Microsoft Edge TTS pour generer l'audio a partir du text 
import asyncio
import nest_asyncio
import tempfile
import os


nest_asyncio.apply()

#Dictionnaire des voix disponnibles par langue
VOICES = {
    "fr": "fr-FR-HenriNeural", #voix francaise 
    "ar": "ar-SA-ZariyahNeural" #voix arabe 
}
#fonction asynchrone pour generer l'audio
async def generate_audio(text: str, language: str = "ar") -> str | None:
    """
    Génère un fichier audio MP3 à partir d'un texte.

    Args:
        text (str): texte à convertir en audio.
        language (str): "ar" pour arabe, "fr" pour français.

    Returns:
        str | None: chemin vers le fichier audio généré ou None si échec.
    """
    
    if not text or not text.strip():
        print(" Texte vide fourni au TTS.")
        return None

    #choix de la voix a partir de la langue demandee et par defaut arabe 
    voice = VOICES.get(language.lower(), VOICES["ar"])

    #fichier temporaire pour stocker l'audio genereé
    output_path = tempfile.mktemp(suffix=".mp3")

    try:
        #genere de l'audio avec edge_tts et l'enregistre
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            print(" Aucun audio généré. Vérifiez les paramètres de voix.")
            return None
    except Exception as e:
        print(f" Erreur TTS : {e}")
        return None
#fonction sysnchrone pour rappeler la fonction asynchrone 
def speak(text: str, language: str = "ar") -> str | None:
    """
    Fonction synchrone pour générer l'audio.

    Args:
        text (str): texte à convertir.
        language (str): "ar" ou "fr".

    Returns:
        str | None: chemin du fichier MP3 généré
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(generate_audio(text, language))
