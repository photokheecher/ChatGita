# Bhagavad Gita QA Retrieval-Augmented Generation (RAG) System

## Overview
This repository contains code for building a Retrieval-Augmented Generation (RAG) model using the Bhagavad Gita in Hindi. The model leverages LangChain, Hugging Face, and ChromaDB to retrieve relevant documents from the Bhagavad Gita text and generate structured answers in Hindi based on the input questions. 

The answers include a shloka (verse), its meaning, explanation, and a practical example, all strictly adhering to the context provided by the retriever.

## Features
- **Document Loading and Splitting:** Loads the Bhagavad Gita text and splits it into manageable chunks for efficient retrieval.
- **Embedding and Vector Database:** Uses `sentence-transformers` for embeddings and stores them in a Chroma vector database for fast retrieval.
- **Hugging Face Integration:** Leverages Hugging Face's Mixtral-8x7B-Instruct model for natural language generation.
- **QA Chain:** A RetrievalQA pipeline is created with the ability to query the Bhagavad Gita text and generate answers based on retrieved references.
- **Custom Prompt Template:** The model strictly answers questions in Hindi, using the Bhagavad Gita shlokas for context, and follows a specific format for answers.

## Setup

### Prerequisites
To run this code, ensure the following libraries are installed:
- `langchain`
- `langchain-core`
- `langchain-huggingface`
- `openai`
- `chromadb`
- `bitsandbytes`
- `huggingface_hub`

These can be installed via the following command:
```bash
pip install -U bitsandbytes langchain langchain-core langchain-huggingface langchain-community openai chromadb
```

### Getting Started
1. **Document Loading:**
   The Bhagavad Gita text is loaded from a specified path (`file_path`). You will need to place the text file in the correct directory or modify the path.

2. **Text Splitting:**
   The text is split into chunks of 256 characters with an overlap of 50 characters. This is done using the `RecursiveCharacterTextSplitter`.

3. **Vector Database Setup:**
   The text chunks are embedded using a multilingual model (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) and stored in a Chroma vector database for efficient retrieval.

4. **Model Integration:**
   The `Mixtral-8x7B-Instruct-v0.1` model from Hugging Face is used to generate answers. You can customize this by changing the model ID or switching to a different model.

5. **RetrievalQA Setup:**
   A custom retrieval prompt is designed to ensure that answers are based only on the Bhagavad Gita shlokas. The answers are structured with a shloka, its meaning, explanation, and practical example.

6. **Querying the Model:**
   Once everything is set up, you can ask questions about the Bhagavad Gita. The model will generate responses based on the retrieved references and the custom prompt.

### Example Query
```python
question = "कृष्ण कौन थे? उसकी सीख क्या थी?"
result = qa_chain.invoke({"query": question})

print("\n🔍 **उत्तर:**")
print(result["result"].split("---")[-1])

print("\n📚 **संदर्भ स्रोत:**")
for doc in result["source_documents"]:
    print(doc.page_content)
```

### Answer Format
The generated answer is structured as follows:

- 📚 **श्लोक:** The verse from the Bhagavad Gita related to the query.
- 📝 **अर्थ:** The meaning of the shloka in Hindi.
- 🔎 **व्याख्या:** The detailed explanation of the shloka.
- 💡 **उदाहरण:** A practical example to explain the shloka in real life.



## Acknowledgments
- [LangChain](https://github.com/hwchase17/langchain)
- [Hugging Face](https://huggingface.co/)
- [ChromaDB](https://www.trychroma.com/)
- [Mixtral-8x7B-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [Building a Retrieval-Augmented Generation (RAG) Model for Hindi and Sanskrit Question Answering with the Bhagavad Gita](https://medium.com/@satyamyadav61/building-a-retrieval-augmented-generation-rag-model-for-hindi-and-sanskrit-question-answering-0e5ac4f3a168)
