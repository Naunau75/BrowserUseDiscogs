import os
import base64
from lightrag import LightRAG
from lightrag.storage import JsonKVStorage, FaissVectorDBStorage
from lightrag.llm import OpenAILLM

# Configuration
os.environ["OPENAI_API_KEY"] = "votre-clé-api"


# 1. Fonction pour encoder l'image en base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 2. Configuration du LLM avec support vision
llm = OpenAILLM(model_name="gpt-4-vision-preview", temperature=0.7, max_tokens=500)

# 3. Configuration du RAG
kv_storage = JsonKVStorage(path="./data/kv_store.json")
vector_storage = FaissVectorDBStorage(path="./data/vector_store.faiss", dimension=1536)

rag = LightRAG(
    llm=llm,
    kv_storage=kv_storage,
    vector_storage=vector_storage,
    chunk_size=500,
    chunk_overlap=50,
)

# 4. Préparation du document de contexte
with open("document1.txt", "r", encoding="utf-8") as f:
    context_doc = f.read()

# Ajout du document au RAG
rag.add_documents([{"content": context_doc, "metadata": {"source": "document1.txt"}}])


# 5. Fonction principale pour analyser l'image avec contexte
def analyze_image_with_context(image_path, question):
    # Encode l'image
    base64_image = encode_image(image_path)

    # Prépare le message pour GPT-4-Vision
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]

    # Obtient la description initiale de l'image
    initial_description = llm.generate(messages=messages)

    # Utilise le RAG pour enrichir la réponse avec le contexte
    final_response = rag.query(
        f"En utilisant ce contexte additionnel, peux-tu enrichir cette description d'image : {initial_description}",
        search_type="hybrid",
        top_k=3,
    )

    return final_response


# 6. Utilisation
if __name__ == "__main__":
    # Exemple d'utilisation
    image_path = "chemin/vers/votre/image.jpg"
    question = "Que vois-tu dans cette image et peux-tu la décrire en détail?"

    response = analyze_image_with_context(image_path, question)
    print("Réponse finale:", response)
