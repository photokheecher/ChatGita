# Import necessary libraries
import numpy as np  # For linear algebra operations
import pandas as pd  # For data manipulation and analysis
import os  # For file system operations
from langchain.document_loaders import TextLoader  # For loading text documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting documents into smaller chunks
from huggingface_hub import notebook_login, login  # For logging into HuggingFace API
from langchain.vectorstores import Chroma  # For storing and retrieving vectors
from langchain.retrievers.self_query.base import SelfQueryRetriever  # For query retrieval
from langchain.chains.query_constructor.base import AttributeInfo  # For query attributes
from langchain.llms import HuggingFaceHub  # For HuggingFace model integration
from langchain_huggingface import HuggingFaceEmbeddings  # For embedding generation
from langchain.chains import RetrievalQA  # For setting up the QA chain
from langchain.prompts import PromptTemplate  # For setting up prompt templates
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # For handling transformers models

# Step 1: Load the Bhagavad Gita text
file_path = "/input/bhagvadnew.txt"
loader = TextLoader(file_path)  # Load the text file
pages = loader.load()  # Load the content of the Bhagavad Gita
print(f"Total pages loaded: {len(pages)}")

# Step 2: Split the text into manageable chunks
chunk_size = 256  # Define the chunk size for splitting
chunk_overlap = 50  # Define the overlap between chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
splits = text_splitter.split_documents(pages)  # Split the documents
print(f"Total splits created: {len(splits)}")

# Step 3: Login to Hugging Face
token_1 = ""  # Add your HuggingFace token here
login(token=token_1)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token_1  # Set the API token in the environment

# Step 4: Set up embeddings using Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Step 5: Set up Chroma for storing vectors
persist_directory = '/vectors'  # Define the directory to store the vectors
vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)

# Check the number of documents in the vector database
print(f"Number of documents in vector database: {vectordb._collection.count()}")

# Step 6: Set up the QA prompt template for retrieval-based generation
template = """
<s>[INST]  
आप एक सम्मानित और सुसंगत सहायक हैं, जिसका उद्देश्य **श्रीमद्भगवद्गीता** के श्लोकों के आधार पर प्रश्नों का उत्तर देना है।  
आपका उत्तर निम्नलिखित **दिशानिर्देशों का कठोरता से पालन** करेगा:  

---

### ✅ **1. उत्तर केवल हिंदी में दें:**  
   - उत्तर **हमेशा और अनिवार्य रूप से हिंदी में ही दें**।  
   - यदि प्रश्न अंग्रेज़ी या किसी अन्य भाषा में हो, तो भी उत्तर केवल हिंदी में दें।  
   - अंग्रेज़ी या अन्य भाषा में उत्तर देना सख्त वर्जित है।  

---

### ✅ **2. संदर्भ-आधारित उत्तर दें:**  
   - उत्तर केवल उस संदर्भ (**context**) से दें, जो **retriever द्वारा प्राप्त हुआ है।**  
   - **संदर्भ में मौजूद श्लोकों के अनुसार ही उत्तर दें।**  
   - संदर्भ से बाहर की जानकारी बिल्कुल न दें, भले ही वह सही हो।  
   - यदि संदर्भ में उत्तर देने के लिए उपयुक्त श्लोक नहीं हो, तो स्पष्ट रूप से कहें:  
     👉 *"इस प्रश्न का उत्तर संदर्भ में उपलब्ध श्लोकों में नहीं है।"*  
   - **कभी भी अनुमान या संदर्भ से बाहर की जानकारी न जोड़ें।**  

---

### ✅ **3. श्लोक, अर्थ और व्याख्या के साथ उत्तर दें:**  
   - हर उत्तर में कम से कम **एक श्लोक** का उल्लेख करें।  
   - श्लोक के साथ निम्नलिखित तीन घटक अनिवार्य रूप से प्रस्तुत करें:  
     1. 📚 **श्लोक:** श्लोक को **सटीक संस्कृत** में उद्धृत करें।  
     2. 📝 **अर्थ:** श्लोक का सरल, स्पष्ट और शुद्ध हिंदी में अर्थ बताएं।  
     3. 🔎 **व्याख्या:** श्लोक का **गहरा और व्यावहारिक अर्थ** स्पष्ट करें।  
     4. 💡 **उदाहरण:** उत्तर को अधिक व्यावहारिक और स्पष्ट बनाने के लिए  
         - रोजमर्रा के जीवन से **प्रासंगिक उदाहरण** दें।  
         - उदाहरण संक्षिप्त लेकिन सटीक हो।  

---

### ✅ **4. उत्तर की गुणवत्ता:**  
   - उत्तर **सुस्पष्ट, तर्कसंगत और क्रमबद्ध** हो।  
   - **अल्पविराम, पूर्णविराम और व्याकरण का सही प्रयोग करें।**  
   - उत्तर **सुसंगत और संतुलित** हो, अनावश्यक विस्तार या दोहराव न करें।  

---

### ✅ **उत्तर का प्रारूप:**  
📚 **श्लोक:**  
{context}  

📝 **अर्थ:**  
- श्लोक का सरल और स्पष्ट अर्थ दें।  

🔎 **व्याख्या:**  
- श्लोक का वास्तविक जीवन में क्या अर्थ है, इसे विस्तार से समझाएं।  

💡 **उदाहरण:**  
- उत्तर को स्पष्ट करने के लिए **व्यावहारिक जीवन से उदाहरण** दें।  
- उदाहरण संक्षिप्त और प्रभावी हो।  

---

### **प्रश्न:**  
{question}  
[/INST]</s>
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Set up the QA chain with the defined template

# Step 7: Set up the Hugging Face model (Mixtral-8x7B)
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Model ID (not URL)
    model_kwargs={
        "max_length": 4096,                   # Max input length
        "max_new_tokens": 1024,               # Max output tokens
        "temperature": 0.1,                   # Sampling temperature
        "do_sample": True                     # Enable sampling
    }
)

# Step 8: Set up the retriever for similarity search
retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # Retrieve fewer chunks for better performance

# Step 9: Integrate with RetrievalQA for question answering
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": QA_CHAIN_PROMPT
    }  # Use custom Hindi prompt
)

# Step 10: Ask a sample question
question = "कृष्ण कौन थे? उसकी सीख क्या थी?"
result = qa_chain.invoke({"query": question})

# Step 11: Display the answer and source documents
print("\n🔍 **उत्तर:**")
print(result["result"].split("---")[-1])  # Display the answer

print("\n📚 **संदर्भ स्रोत:**")
for doc in result["source_documents"]:  # Display the source documents
    print(doc.page_content)
