import os
import torch
import nest_asyncio
import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from voice.stt import STTEngine
from voice.tts import speak

nest_asyncio.apply()

# 1. Configuration du device et chemin RAG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAG_DIR = "rag/graphrag_index"
print(f" Tanit (Fertility Expert) démarre sur : {DEVICE}")

# 2.Chargement du modèle Qwen2-VL 7B avec quantization 4-bit
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f" Chargement de {MODEL_ID}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=bnb_config, torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 3. Initialisation STT et embeddings
stt = STTEngine(device="cuda")
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")# Modèle d'embeddings multilingue pour transformer le texte en vecteurs sémantiques
# 4. Chargement de l’index RAG
HAS_RAG = False
rag_retriever = None
if os.path.exists(RAG_DIR):
    try:
        rag_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=RAG_DIR))
        rag_retriever = rag_index.as_retriever(similarity_top_k=2) #on prend les 2 documents les plus proches
        HAS_RAG = True
        print(" RAG chargé.")
    except:
        print(" RAG non trouvé ou erreur.")


# 5. Prompt système pour eliminer l'allucination et voir une bonne reponse

BASE_SYSTEM_PROMPT = """Tu es Tanit, une assistante médicale experte en fertilité et gynécologie.

IDENTITÉ ET RÈGLES DE SÉCURITÉ :
1. Tu n'es pas humaine, tu es une IA d'analyse.
2. CRITIQUE : NE JAMAIS INVENTER DE CHIFFRES. Si une valeur est floue, manquante ou illisible, dis explicitement "Valeur non visible" ou "Illisible".
3. Ne fais jamais de diagnostic vital (cancer, urgence vitale). Renvoie vers un médecin.

MODE OPÉRATOIRE :

CAS A : ANALYSE D'IMAGE (Bilan Hormonal, Spermogramme, Écho)
- Lis le document ligne par ligne.
- Repère la colonne "Résultat" et la colonne "Valeur de référence" (Norme).
- Compare le résultat à la norme fournie SUR LE PAPIER (pas tes connaissances générales, car les labos diffèrent).
- Signale uniquement ce qui est ANORMAL (Haut ou Bas).
- Si tout est normal, dis : "Les résultats visibles semblent dans les normes du laboratoire."

CAS B : QUESTION GÉNÉRALE (Sans image)
- Réponds de manière pédagogique, courte et empathique.
- Base-toi sur les connaissances médicales standards en fertilité.
"""
#detecte si le text contient arabe , pour choisir la langue de reponse 
def contains_arabic(text):
    if not text: return False
    return any("\u0600" <= c <= "\u06FF" for c in text)

# 6. fonction chat 
def chat(audio, image, history):
    if history is None: history = []

    logs = ""
    user_text = ""

    # Transcription Audio
    if audio:
        try:
            user_text = stt.transcribe(audio)
            logs += f"[Audio: {user_text}] "
        except Exception as e:
            logs += f"[Erreur Audio: {e}] "

    
    if not user_text and not image:
        return history, None, "Veuillez parler ou envoyer une image."

    # detection de la langue 
    is_arabic_query = contains_arabic(user_text)

    # Injection dynamique de la langue et du ton
    if is_arabic_query:
        lang_instruction = """
        \nIMPORTANT : L'utilisateur parle ARABE.
        1. Tu DOIS répondre en ARABE (clair et empathique).
        2. Traduis les termes médicaux techniques mais explique-les simplement en arabe.
        3. Ne réponds PAS en français.
        """
    else:
        lang_instruction = """
        \nIMPORTANT : Réponds en FRANÇAIS.
        Utilise un ton professionnel mais rassurant.
        """

    final_system_prompt = BASE_SYSTEM_PROMPT + lang_instruction

    #  Récupération contexte RAG
    context_str = ""
    if HAS_RAG and user_text:
        try:
            nodes = rag_retriever.retrieve(user_text)
            context_str = "\n".join([n.get_content() for n in nodes])
            logs += "[RAG utilisé] "
        except:
            pass

    if context_str:
        final_system_prompt += f"\n\nCONTEXTE MÉDICAL VÉRIFIÉ :\n{context_str}"

    #  Construction des messages pour Qwen
    content_user = []
    if image:
        content_user.append({"type": "image", "image": image})
        if not user_text:
            if is_arabic_query:
                prompt_text = "قم بتحليل هذا التقرير الطبي بدقة. اذكر فقط القيم غير الطبيعية. إذا كانت الصورة غير واضحة، قل ذلك."
            else:
                prompt_text = "Analyse ce rapport médical strictement. Cite les valeurs, compare-les aux références de l'image. Dis-moi ce qui est anormal. Si illisible, dis 'Illisible'."
        else:
            prompt_text = user_text
        content_user.append({"type": "text", "text": prompt_text})
    else:
        content_user.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": content_user}
    ]

    # Génération de réponse
    try:
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        generated = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.01,
            do_sample=True,
            top_p=0.8
        )
        response = processor.batch_decode(generated, skip_special_tokens=True)[0].split("assistant")[-1].strip()

    except Exception as e:
        response = "Une erreur technique empêche l'analyse. Veuillez réessayer avec une image plus claire."
        logs += f"[Erreur Gen: {e}]"

    #  Génère l’audio du texte de réponse selon la langue détectée
    tts_lang = "ar" if contains_arabic(response) else "fr"
    audio_out = speak(response, tts_lang)

    display_text = user_text if user_text else "[Analyse Document]"
    history.append([display_text, response])
    return history, audio_out, logs


# Interface Gradio

custom_css = """
.chatbot {
    border-radius: 20px !important;
    box-shadow: 0 4px 20px rgba(233, 30, 99, 0.1) !important;
    border: 1px solid #fce4ec !important;
}
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto;
    background-color: #fafafa;
}
h1 {
    background: linear-gradient(45deg, #e91e63, #ff80ab);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
footer {display: none !important;}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="rose",
        font=gr.themes.GoogleFont("Quicksand")
    ),
    css=custom_css,
    title="Tanit - Assistant Fertilité"
) as demo:

    gr.HTML("""
    <div style="text-align: center; padding: 25px 0;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">🌸 Tanit</h1>
        <p style="color: #666; font-size: 1.1rem; font-weight: 500;">
            Assistant d'Analyse Fertilité & Gynécologie
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
            <span style="background: #fce4ec; color: #e91e63; padding: 4px 12px; border-radius: 15px; font-size: 0.8rem;">Bilingue Arabe/Français</span>
            <span style="background: #e3f2fd; color: #1976d2; padding: 4px 12px; border-radius: 15px; font-size: 0.8rem;">Analyse Vision IA</span>
        </div>
    </div>
    """)

    
    chatbot = gr.Chatbot(
        height=550,
        avatar_images=(
            None, 
            "https://cdn-icons-png.flaticon.com/512/3304/3304567.png"  
        ),
        show_copy_button=True,
        show_share_button=False,
        bubble_full_width=False,
        render_markdown=True,
        elem_classes="chatbot"
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("🎤 Audio & Image"):
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Message Vocal")
                    image_in = gr.Image(type="filepath", label="Document Médical", height=250)

            gr.Examples(
                examples=[
                    ["J'ai 35 ans, mon AMH est à 0.5, est-ce inquiétant ?", None],
                    ["هل نتيجة تحليل FSH 12 تعتبر طبيعية؟", None],
                    ["Analyse ce cycle menstruel s'il te plait.", None]
                ],
                inputs=[audio_in, image_in],
                label="Questions types :"
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ Effacer", variant="secondary", size="lg")
                btn = gr.Button("🌸 Envoyer / إرسال", variant="primary", size="lg")

        with gr.Column(scale=3):
            gr.Markdown("### 🔊 Réponse Vocale")
            audio_out = gr.Audio(label=None, autoplay=True, interactive=False)

            with gr.Accordion("⚙️ Détails Techniques", open=False):
                debug_box = gr.Textbox(label="Logs", interactive=False, lines=6)

    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; color: #999; font-size: 0.8rem; border-top: 1px solid #eee; padding-top: 10px;">
        ⚠️ <strong>Avis de non-responsabilité :</strong> Tanit est une IA d'assistance. Elle ne remplace pas un médecin.
    </div>
    """)

    btn.click(chat, inputs=[audio_in, image_in, chatbot], outputs=[chatbot, audio_out, debug_box])
    clear_btn.click(lambda: ([], None, None, ""), outputs=[chatbot, audio_out, image_in, debug_box])

if __name__ == "__main__":
    demo.launch(share=True)
