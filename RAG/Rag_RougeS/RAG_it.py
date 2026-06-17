import os
import csv
from pathlib import Path

import torch
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# =========================
# CONFIG MODEL
# =========================
BASE_MODEL = "meta-llama/Llama-3.1-8B"
CHECKPOINT = r"checkpoint_path"
TOKENIZER_PATH = r"tokenizer_path"

PDF_DIR = r"rag_path"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 2
MAX_CONTEXT_CHARS = 5000
OUTPUT_CSV = "csv_path"



# =========================
# MODEL
# =========================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, CHECKPOINT)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================
# PDF READING
# =========================
def extract_text_from_pdf(pdf_path: str):
    docs = []
    reader = PdfReader(pdf_path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append({
                "source": Path(pdf_path).name,
                "page": i + 1,
                "text": text.strip()
            })

    return docs


def load_all_pdfs(pdf_dir: str):
    all_docs = []
    pdf_paths = list(Path(pdf_dir).glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF in: {pdf_dir}")

    for pdf_path in pdf_paths:
        all_docs.extend(extract_text_from_pdf(str(pdf_path)))

    return all_docs


# =========================
# CHUNKING
# =========================
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    chunks = []
    start = 0
    text = text.strip()

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def build_chunks(documents):
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": i,
                "text": chunk
            })

    return all_chunks


# =========================
# EMBEDDINGS
# =========================
def build_vector_store(chunks):
    embedder = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    return embedder, embeddings


def retrieve_chunks(query, embedder, embeddings, chunks, top_k=2):
    query_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")[0]

    scores = np.dot(embeddings, query_emb)

    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        item = chunks[idx].copy()
        item["score"] = float(scores[idx])
        retrieved.append(item)

    return retrieved


# =========================
# PROMPT RAG
# =========================
def build_rag_prompt(question, retrieved_chunks):
    context_parts = []
    total_chars = 0

    for c in retrieved_chunks:
        piece = f"[Fonte: {c['source']} - pagina {c['page']}]\n{c['text']}\n"
        if total_chars + len(piece) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(piece)
        total_chars += len(piece)

    context = "\n".join(context_parts)

    prompt = f"""Sei un assistente scientifico.

REGOLE RIGIDE:
- Usa SOLO le informazioni presenti nel contesto fornito.
- NON usare conoscenze pregresse.
- NON aggiungere informazioni che non sono esplicitamente presenti.
- Rispondi SEMPRE in italiano.
- Se manca qualche passaggio o informazione, scrivi: Non disponibile nel contesto fornito.

CONTESTO:
{context}

DOMANDA:
{question}

RISPOSTA:
"""

    return prompt, context


# =========================
# ROUGE-S (SKIP-BIGRAM)
# =========================
def get_skip_bigrams(tokens):
    pairs = []
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((tokens[i], tokens[j]))
    return pairs


def rouge_s_score(reference: str, candidate: str):
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    ref_skip = get_skip_bigrams(ref_tokens)
    cand_skip = get_skip_bigrams(cand_tokens)

    if len(ref_skip) == 0 or len(cand_skip) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "overlap_skip_bigrams": 0,
            "reference_skip_bigrams": len(ref_skip),
            "candidate_skip_bigrams": len(cand_skip),
        }

    ref_counts = {}
    for bg in ref_skip:
        ref_counts[bg] = ref_counts.get(bg, 0) + 1

    cand_counts = {}
    for bg in cand_skip:
        cand_counts[bg] = cand_counts.get(bg, 0) + 1

    overlap = 0
    for bg in cand_counts:
        if bg in ref_counts:
            overlap += min(cand_counts[bg], ref_counts[bg])

    precision = overlap / len(cand_skip) if cand_skip else 0.0
    recall = overlap / len(ref_skip) if ref_skip else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap_skip_bigrams": overlap,
        "reference_skip_bigrams": len(ref_skip),
        "candidate_skip_bigrams": len(cand_skip),
    }


# =========================
# GENERATION + ROUGE-S
# =========================
def generate_rag_answer(question, embedder, embeddings, chunks, top_k=TOP_K):
    retrieved = retrieve_chunks(question, embedder, embeddings, chunks, top_k=top_k)
    prompt, context_it = build_rag_prompt(question, retrieved)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if decoded.startswith(prompt):
        answer = decoded[len(prompt):].strip()
    else:
        answer = decoded.strip()

    rouge_s_scores = rouge_s_score(context_it, answer)

    return answer, retrieved, rouge_s_scores, context_it


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("PDF Loading...")
    documents = load_all_pdfs(PDF_DIR)

    print("chunk creation...")
    chunks = build_chunks(documents)

    print("embeddings creation...")
    embedder, embeddings = build_vector_store(chunks)

    questions = [
        # Esperimento_1_Scienze_della_vita
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento sulla legge di Lavoisier e verificare il principio di conservazione della massa?",
        "Qual è l'obiettivo dell'esperimento sulla legge di Lavoisier riguardo alla conservazione della massa nella reazione tra zinco e acido cloridrico?",
        "Ho svolto l'esperimento lasciando il matraccio aperto durante la reazione tra cristalli di zinco e acido cloridrico: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando un matraccio, zinco e acido cloridrico per verificare che la massa totale rimanga costante secondo la legge di Lavoisier?",

        # Esperimento_Elettricita
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento con il multimetro digitale e misurare correttamente resistenza, tensione e corrente?",
        "Qual è l'obiettivo dell'esperimento con il multimetro digitale riguardo alla misura di resistenza, tensione (differenza di potenziale) e intensità di corrente?",
        "Ho svolto l'esperimento misurando la resistenza di un componente mentre era ancora alimentato: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando un multimetro digitale, i puntali e una batteria per misurare con precisione tensione, corrente e resistenza?",

        # Esperimento_biologia_modificato
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento di osservazione delle cellule vegetali e confrontare al microscopio la struttura delle cellule di cipolla e orchidea?",
        "Qual è l'obiettivo dell'esperimento sulle cellule vegetali riguardo al confronto tra cellule di cipolla e cellule di Orchidea Phalaenopsis osservate al microscopio?",
        "Ho svolto l'esperimento posizionando il coprioggetto senza fare attenzione alle bolle d'aria durante la preparazione del campione di epidermide di cipolla: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando un microscopio, vetrini, epidermide di cipolla e blu di metilene per osservare e confrontare accuratamente le cellule vegetali?",

        # Esperimento_elettromagnetismo_modificato
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento sui magneti e osservare l'interazione tra i poli magnetici?",
        "Qual è l'obiettivo dell'esperimento sui magneti riguardo all'interazione tra due magneti e i loro poli?",
        "Ho svolto l'esperimento avvicinando i magneti senza cambiarne l'orientamento per testare poli diversi: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando due magneti per osservare l'attrazione e la repulsione tra poli uguali e opposti?",

        # Esperimento_fototropismo_modificato
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento sul fototropismo e osservare come le piante crescono in risposta alla direzione della luce?",
        "Qual è l'obiettivo dell'esperimento sul fototropismo riguardo alla crescita delle piante in diverse condizioni di luce e alla distinzione tra fototropismo positivo e negativo?",
        "Ho svolto l'esperimento esponendo tutte le piante a una luce uniforme invece di dirigere la luce da un solo lato nella configurazione sperimentale: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando semi, terreno, scatole e una sorgente luminosa controllata per osservare e misurare accuratamente il fototropismo nelle piante?",

        # Esperimento_meccanica_modificato
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento con il calibro e misurare con precisione le dimensioni di un oggetto solido?",
        "Qual è l'obiettivo dell'esperimento con il calibro riguardo alla misura di larghezza, lunghezza e profondità di un oggetto usando un calibro a corsoio?",
        "Ho svolto l'esperimento leggendo solo la scala principale del calibro senza considerare il nonio: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando un calibro a corsoio e un oggetto solido per misurare le dimensioni con precisione, leggendo sia la scala principale sia il nonio?",

        # Estrazione_dna_vegetale_sintetico
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento di estrazione del DNA vegetale e ottenere DNA visibile dalle cellule vegetali?",
        "Qual è l'obiettivo dell'esperimento di estrazione del DNA riguardo all'isolamento e alla visualizzazione del DNA da cellule vegetali usando materiali comuni di laboratorio?",
        "Ho svolto l'esperimento usando alcol a temperatura ambiente invece di alcol freddo durante la fase di precipitazione del DNA: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando materiale vegetale, detergente, sale e alcol freddo per estrarre e osservare accuratamente il DNA?",

        # Manuale_energia_alternativa_modificato
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento sul motore a combustione interna e osservare come avviene l'accensione del carburante all'interno di un cilindro?",
        "Qual è l'obiettivo dell'esperimento sul motore a combustione interna riguardo al funzionamento della combustione del carburante e dell'accensione all'interno di un sistema confinato?",
        "Ho svolto l'esperimento aumentando la quantità di alcol spruzzata nel cilindro senza considerare l'ossigeno disponibile: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando un cilindro, alcol e una sorgente di scintilla per osservare e comprendere accuratamente il processo di combustione?",

        # Manuale_la_termodinamica_modificato
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento di distillazione e separare l'alcol da una miscela acqua-alcol?",
        "Qual è l'obiettivo dell'esperimento di distillazione riguardo alla separazione di sostanze in base ai loro diversi punti di ebollizione?",
        "Ho svolto l'esperimento senza raffreddare il tubo a U durante il processo di distillazione: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando un apparato di distillazione, una sorgente di calore e un sistema di raffreddamento per separare e raccogliere accuratamente l'alcol da una miscela?",

        # Saggio-fiamma_libro_zanichelli
        "Puoi estrarre un elenco dei passaggi necessari per svolgere l'esperimento del saggio alla fiamma e identificare diversi ioni metallici in base al colore della fiamma?",
        "Qual è l'obiettivo dell'esperimento del saggio alla fiamma riguardo all'identificazione dei metalli attraverso la loro caratteristica emissione di luce?",
        "Ho svolto l'esperimento senza pulire l'ansa di filo tra la prova di diversi sali metallici: è corretto? Se no, cosa dovrei correggere per eseguire bene l'esperimento?",
        "Come posso completare correttamente l'esperimento usando una sorgente di fiamma, sali metallici e un'ansa di filo per osservare e distinguere accuratamente i colori caratteristici dei diversi elementi?",
    ]

    results = []

    for q in questions:
        print("\n" + "=" * 120)
        print("DOMANDA:")
        print(q)

        answer, retrieved, rouge_scores, context_it = generate_rag_answer(q, embedder, embeddings, chunks)

        print("\nRISPOSTA:")
        print(answer)

        print("\nCONTESTO:")
        print(context_it)

        print("\nROUGE-S:")
        for k, v in rouge_scores.items():
            print(f"{k}: {v}")

        print("\nCHUNK RECUPERATI:")
        for r in retrieved:
            print(f"- {r['source']} | pagina {r['page']} | score={r['score']:.4f}")

        retrieved_summary = [
            f"{r['source']} | p{r['page']} | score={r['score']:.4f}"
            for r in retrieved
        ]

        results.append({
            "question": q,
            "answer": answer,
            "rouge_s_precision": rouge_scores["precision"],
            "rouge_s_recall": rouge_scores["recall"],
            "rouge_s_f1": rouge_scores["f1"],
            "rouge_s_overlap_skip_bigrams": rouge_scores["overlap_skip_bigrams"],
            "rouge_s_reference_skip_bigrams": rouge_scores["reference_skip_bigrams"],
            "rouge_s_candidate_skip_bigrams": rouge_scores["candidate_skip_bigrams"],
            "retrieved_chunks": " || ".join(retrieved_summary)
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "answer",
                "rouge_s_precision",
                "rouge_s_recall",
                "rouge_s_f1",
                "rouge_s_overlap_skip_bigrams",
                "rouge_s_reference_skip_bigrams",
                "rouge_s_candidate_skip_bigrams",
                "retrieved_chunks",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_CSV}")

