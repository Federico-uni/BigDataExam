# Fine-Tuning for Scientific Education

## Project Description

This project focuses on the preparation of a scientific dataset and the fine-tuning of a large
language model for educational purposes and Rag evaluation.

------------------------------------------------------------------------

## Project Structure

    Fine Tuning/
        - Final_dataset/                        Dataset for fine-tuning
        - FineTuning.py                         Fine-tuning script
    
    RAG/
        - Datasets/
            - IT_Dataset_Rag/                   Italian RAG dataset  
            - EN_Dataset_RAG/                   English RAG dataset
        - Outputs/
            - OUTPUT_EN_Dataset_Rouge/          English Dataset outputs for RougeS
            - OUTPUT_IT_Dataset_Rouge/          Italian Dataset outputs for RougeS
            - OUTPUT_IT_Dataset_bertscore/      Italian Dataset outputs for bertscore
            - OUTPUT_IT_Dataset_bertscore/      Italian Dataset outputs for bertscore
        - RAG_RougeS/                                  
            - RAG.py                            English fine-tuned model RAG  
            - RAG_base_llama.py                 English base model RAG  
            - RAG_it.py                         Italian fine-tuned model RAG
            - RAG_it_base_llama.py              Italian base model RAG
        - RAG_BertScore/                                  
            - RAG_bertscore.py                  English fine-tuned model RAG  
            - RAG_base_llama_bertscore.py       English base model RAG  
            - RAG_it_bertscore.py               Italian fine-tuned model RAG
            - RAG_it_base_llama_bertscore.py    Italian base model RAG
        - GPT Evaluations                       GPT evaluations for every experiments  

    Documentation/
        - Plots/                                metrics plots
            - RougeS_plots
            - BertScore_plots
            - Llm_asJudge_plots                  
        - Doc.pdf                               Report

------------------------------------------------------------------------

## System Workflow

1.  Dataset preparatioin
2.  Llama 3.1 Fine Tuning
3.  Document loading and chunking
4.  Embedding generation
5.  Retrieval using cosine similarity
6.  Response generation with LLaMA
7.  Evaluation using ROUGE
8.  Evaluation LLM as a Judge (ChatGPT)
9.  Evaluation with BertScore


