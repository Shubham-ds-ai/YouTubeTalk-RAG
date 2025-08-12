from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor
from datasets import Dataset, Features, Value, Sequence
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import pandas as pd

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or "YOUR_KEY_HERE"
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or "YOUR_ENDPOINT"
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

DOCS_FAISS_PATH, MEMORY_FAISS_PATH = "faiss_docs_index", "faiss_memory_index"
SHORT_TERM_LIMIT, MEMORY_K, DOCS_K, COMBINED_MAX_DOCS = 6, 4, 4, 8

llm = AzureChatOpenAI(deployment_name=AZURE_CHAT_DEPLOYMENT, api_key=AZURE_API_KEY,
                      azure_endpoint=AZURE_ENDPOINT, api_version=AZURE_API_VERSION, temperature=0)
embeddings = AzureOpenAIEmbeddings(api_key=AZURE_API_KEY, azure_endpoint=AZURE_ENDPOINT,
                                   api_version=AZURE_API_VERSION, azure_deployment=AZURE_EMBED_DEPLOYMENT)
ragas_llm, ragas_emb = LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

prompt = PromptTemplate(
    template="""You are a helpful AI assistant.
If the user's question is about the YouTube video, use ONLY the transcript and memory context to answer.
If the input is casual or unrelated, respond naturally.

Transcript + Memory Context:
{context}

Question: {question}""",
    input_variables=["context", "question"],
)

def build_dataset(rows):
    features = Features({
        "question": Value("string"), "answer": Value("string"),
        "contexts": Sequence(Value("string")), "ground_truth": Value("string"),
    })
    return Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [[str(x) for x in r["contexts"]] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    }, features=features)

video_id = "2DoDflpemBk"
try:
    transcript_data = YouTubeTranscriptApi().list(video_id).find_transcript(['en']).fetch()
    transcript_text = " ".join(chunk.text for chunk in transcript_data)
except TranscriptsDisabled:
    raise SystemExit("No captions available.")
if not transcript_text.strip():
    raise SystemExit("Transcript empty.")

texts = [doc.page_content for doc in RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200).create_documents([transcript_text]) if doc.page_content.strip()]

doc_store = FAISS.load_local(DOCS_FAISS_PATH, embeddings, allow_dangerous_deserialization=True) if os.path.exists(DOCS_FAISS_PATH) else FAISS.from_texts(texts, embeddings)
doc_store.save_local(DOCS_FAISS_PATH)
doc_retriever_mmr = doc_store.as_retriever(search_type="mmr", search_kwargs={"k": DOCS_K, "fetch_k": 20, "lambda_mult": 0.5})

if os.path.exists(MEMORY_FAISS_PATH):
    memory_store = FAISS.load_local(MEMORY_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    memory_store = FAISS.from_texts(["__MEMORY_PLACEHOLDER__"], embeddings, metadatas=[{"seed": True}])
    memory_store.save_local(MEMORY_FAISS_PATH)
memory_retriever = memory_store.as_retriever(search_type="similarity", search_kwargs={"k": MEMORY_K})

emb_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.70)
compression_retriever_mmr_embed = ContextualCompressionRetriever(base_retriever=doc_retriever_mmr, base_compressor=emb_filter)
compression_retriever_mmr_llm = ContextualCompressionRetriever(base_retriever=doc_retriever_mmr, base_compressor=LLMChainExtractor.from_llm(llm))
RETRIEVER_VARIANTS = {"mmr": doc_retriever_mmr, "mmr+emb_filter": compression_retriever_mmr_embed, "mmr+llm_extract": compression_retriever_mmr_llm}

short_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def summarize_short_memory_and_persist(user_id="user_1"):
    conv = short_memory.load_memory_variables({}).get("chat_history", [])
    if not conv: return None
    summary_prompt = "Summarize the conversation:\n" + "\n".join(f"{m.type}: {m.content}" for m in conv)
    summary = llm.invoke(summary_prompt).content.strip()
    if not summary: return None
    ts = datetime.utcnow().isoformat()
    memory_store.add_texts([f"[summary @ {ts}] {summary}"], metadatas=[{"user_id": user_id, "timestamp": ts}])
    memory_store.save_local(MEMORY_FAISS_PATH)
    short_memory.clear()

def combine_and_answer(question, docs_retriever, include_memory=True, min_context_len=200):
    mem_docs = memory_retriever.invoke(question) if include_memory else []
    doc_docs = docs_retriever.invoke(question)
    combined_docs = (mem_docs or []) + (doc_docs or [])
    context_text = "\n\n".join(d.page_content for d in combined_docs[:COMBINED_MAX_DOCS])
    if len(context_text.strip()) < min_context_len:
        context_text = "\n\n".join(texts)
    answer = llm.invoke(prompt.format(context=context_text, question=question)).content
    short_memory.chat_memory.add_user_message(question)
    short_memory.chat_memory.add_ai_message(answer)
    if len(short_memory.load_memory_variables({}).get("chat_history", [])) >= SHORT_TERM_LIMIT:
        summarize_short_memory_and_persist()
    return {"question": question, "answer": answer, "source_documents": combined_docs}

eval_inputs = [
    {"question": "Where in New York did the presenter explore on a Friday night?", "reference": "Times Square on a Friday night."},
    {"question": "Name the bar the group went to.", "reference": "Long Acre."},
]
metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
score_rows, all_rows = [], []
for label, retriever_variant in RETRIEVER_VARIANTS.items():
    rows = []
    for item in eval_inputs:
        out = combine_and_answer(item["question"], retriever_variant)
        rows.append({"question": out["question"], "answer": out["answer"],
                     "contexts": [d.page_content for d in out["source_documents"]],
                     "ground_truth": item["reference"]})
    dataset = build_dataset(rows)
    results = evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb)
    score_rows.append({"variant": label, **{k: float(v) for k, v in results.items()}})
    all_rows.extend({**r, "variant": label} for r in rows)

pd.DataFrame(score_rows).to_csv("ragas_scores_by_variant_with_memory.csv", index=False)
pd.DataFrame(all_rows).to_csv("ragas_eval_rows_by_variant_with_memory.csv", index=False)

print("\nInteractive chat (type 'exit' to quit).")
while True:
    q = input("\nYou: ")
    if q.strip().lower() in ("exit", "quit"): break
    resp = combine_and_answer(q, RETRIEVER_VARIANTS["mmr"])
    print("\nAssistant:", resp["answer"])
