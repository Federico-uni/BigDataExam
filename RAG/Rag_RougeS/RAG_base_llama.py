import os
import csv
from pathlib import Path

import torch
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from deep_translator import GoogleTranslator
from huggingface_hub import login

# =========================
# CONFIG MODEL
# =========================

HF_TOKEN="HF_TOKEN"

login(token=HF_TOKEN)

BASE_MODEL = "meta-llama/Llama-3.1-8B"
OUT_DIR = "./ft_llama31_8b_lora"

os.makedirs(OUT_DIR, exist_ok=True)


PDF_DIR = r"C:/Users/Tesisti/Desktop/EN_Dataset_RAG"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 2
MAX_CONTEXT_CHARS = 5000
OUTPUT_CSV = "C:/Users/Tesisti/Desktop/OUTPUT_EN_Dataset/rag_results_rouge_sDoSampleFALSE_ModelloBase.csv"

# =========================
# TRADUZIONE
# =========================
translator = GoogleTranslator(source="auto", target="en")


def translate_text(text: str, max_chunk_chars: int = 4500):
    text = text.strip()
    if not text:
        return text

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        piece = text[start:end]
        try:
            translated = translator.translate(piece)
        except Exception:
            translated = piece
        pieces.append(translated)
        start = end

    return "\n".join(pieces)


# =========================
# MODEL
# =========================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
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

    prompt = f"""You are a scientific assistant.

STRICT RULES:
- Use ONLY the information from the provided context.
- DO NOT use prior knowledge.
- DO NOT add information that is not explicitly stated.
- If any step is missing, say: Not available in the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
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

    context_en = translate_text(context_it)
    rouge_s_scores = rouge_s_score(context_en, answer)

    return answer, retrieved, rouge_s_scores, context_en


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
        "Can you extract a list of the steps required to carry out the Lavoisier's law experiment and verify the principle of conservation of mass?",
        "What is the objective of the Lavoisier's law experiment regarding the conservation of mass in the reaction between zinc and hydrochloric acid?",
        "I carried out the experiment leaving the flask open during the reaction between zinc crystals and hydrochloric acid: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using a flask, zinc, and hydrochloric acid to verify that the total mass remains constant according to Lavoisier's law?",

        # Esperimento_Elettricita
        "Can you extract a list of the steps required to carry out the digital multimeter experiment and correctly measure resistance, voltage, and current?",
        "What is the objective of the digital multimeter experiment regarding the measurement of resistance, voltage (potential difference), and current intensity?",
        "I carried out the experiment by measuring the resistance of a component while it was still powered: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using a digital multimeter, probes, and a battery to measure voltage, current, and resistance accurately?",

        # Esperimento_biologia_modificato
        "Can you extract a list of the steps required to carry out the plant cell observation experiment and compare the structure of onion and orchid cells under a microscope?",
        "What is the objective of the plant cell experiment regarding the comparison between onion cells and Orchid Phalaenopsis cells observed under a microscope?",
        "I carried out the experiment placing the cover slip without paying attention to air bubbles during the preparation of the onion epidermis sample: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using a microscope, slides, onion epidermis, and methylene blue to observe and compare plant cells accurately?",

        # Esperimento_elettromagnetismo_modificato
        "Can you extract a list of the steps required to carry out the magnet experiment and observe the interaction between magnetic poles?",
        "What is the objective of the magnet experiment regarding the interaction between two magnets and their poles?",
        "I carried out the experiment by bringing the magnets close together without changing their orientation to test different poles: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using two magnets to observe the attraction and repulsion between opposite and like poles?",

        # Esperimento_fototropismo_modificato
        "Can you extract a list of the steps required to carry out the phototropism experiment and observe how plants grow in response to light direction?",
        "What is the objective of the phototropism experiment regarding the growth of plants under different light conditions and the distinction between positive and negative phototropism?",
        "I carried out the experiment by exposing all plants to uniform light instead of directing light from one side in the experimental setup: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using seeds, soil, boxes, and a controlled light source to observe and measure phototropism in plants accurately?",

        # Esperimento_meccanica_modificato
        "Can you extract a list of the steps required to carry out the caliper experiment and measure the dimensions of a solid object accurately?",
        "What is the objective of the caliper experiment regarding the measurement of width, length, and depth of an object using a vernier caliper?",
        "I carried out the experiment by reading only the main scale of the caliper without considering the vernier scale: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using a vernier caliper and a solid object to measure dimensions with precision, including reading both the main scale and the vernier scale?",

        # Estrazione_dna_vegetale_sintetico
        "Can you extract a list of the steps required to carry out the plant DNA extraction experiment and obtain visible DNA from plant cells?",
        "What is the objective of the DNA extraction experiment regarding the isolation and visualization of DNA from plant cells using common laboratory materials?",
        "I carried out the experiment by using alcohol at room temperature instead of cold alcohol during the DNA precipitation step: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using plant material, detergent, salt, and cold alcohol to extract and observe DNA accurately?",

        # Manuale_energia_alternativa_modificato
        "Can you extract a list of the steps required to carry out the internal combustion engine experiment and observe how fuel ignition occurs inside a cylinder?",
        "What is the objective of the internal combustion engine experiment regarding the functioning of fuel combustion and ignition inside a confined system?",
        "I carried out the experiment by increasing the amount of alcohol sprayed inside the cylinder without considering the available oxygen: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using a cylinder, alcohol, and a spark source to observe and understand the combustion process accurately?",

        # Manuale_la_termodinamica_modificato
        "Can you extract a list of the steps required to carry out the distillation experiment and separate alcohol from a water-alcohol mixture?",
        "What is the objective of the distillation experiment regarding the separation of substances based on their different boiling points?",
        "I carried out the experiment without cooling the U-shaped tube during the distillation process: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using a distillation setup, heat source, and cooling system to separate and collect alcohol from a mixture accurately?",

        # Saggio-fiamma_libro_zanichelli
        "Can you extract a list of the steps required to carry out the flame test experiment and identify different metal ions based on flame color?",
        "What is the objective of the flame test experiment regarding the identification of metals through their characteristic emission of light?",
        "I carried out the experiment without cleaning the wire loop between testing different metal salts: is this correct? If not, what should I correct to perform the experiment properly?",
        "How can I correctly complete the experiment using a flame source, metal salts, and a wire loop to accurately observe and distinguish the characteristic colors of different elements?",
    ]

    results = []

    for q in questions:
        print("\n" + "=" * 120)
        print("QUESTION:")
        print(q)

        answer, retrieved, rouge_scores, context_en = generate_rag_answer(q, embedder, embeddings, chunks)

        print("\nANSWER:")
        print(answer)

        print("\nTRANSLATED CONTEXT (EN):")
        print(context_en)

        print("\nROUGE-S:")
        for k, v in rouge_scores.items():
            print(f"{k}: {v}")

        print("\nCHUNK KEPT:")
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

