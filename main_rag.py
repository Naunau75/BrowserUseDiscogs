import os
from lightrag import LightRAG
from lightrag.storage import JsonKVStorage, FaissVectorDBStorage
from lightrag.llm import OpenAILLM

# Configuration des clés d'API
os.environ["OPENAI_API_KEY"] = "votre-clé-api"

# 1. Initialisation du LLM
llm = OpenAILLM(model_name="gpt-4", temperature=0.7, max_tokens=500)

# 2. Configuration des stockages
kv_storage = JsonKVStorage(path="./data/kv_store.json")
vector_storage = FaissVectorDBStorage(
    path="./data/vector_store.faiss",
    dimension=1536,  # Dimension pour les embeddings d'OpenAI
)

# 3. Initialisation de LightRAG
rag = LightRAG(
    llm=llm,
    kv_storage=kv_storage,
    vector_storage=vector_storage,
    chunk_size=500,
    chunk_overlap=50,
)

# 4. Ajout de documents
documents = [
    {
        "content": "Voici un exemple de document...",
        "metadata": {"source": "exemple1.txt"},
    },
    {
        "content": "Un autre document d'exemple...",
        "metadata": {"source": "exemple2.txt"},
    },
]

# Indexation des documents
rag.add_documents(documents)

# 5. Interrogation du système
question = "Quelle information pouvez-vous me donner sur..."
response = rag.query(
    question,
    search_type="hybrid",  # Peut être 'local', 'global', 'hybrid', 'naive' ou 'mix'
    top_k=3,  # Nombre de documents à récupérer
)

print(response)
