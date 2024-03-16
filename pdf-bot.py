#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import pypdf

import os
os.environ["OPENAI_API_KEY"] = ""


# In[11]:


file_path = "C:\\Users\\James Lee\\Desktop\\HM-Group-Annual-and-Sustainability-Report-2022.pdf"
pdf_reader = pypdf.PdfReader(file_path)


# In[20]:


pdfs = pdf_reader.pages[6:8]
text = ''
for i in range(len(pdfs)):
    text = pdfs[i].extract_text()
    text += text
text = text.replace("\n", "")
text


# In[21]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 32,
    length_function = len,
)
texts = text_splitter.split_text(text)
texts


# In[23]:


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


# In[25]:


vectorstore = Chroma.from_texts(texts, embeddings)


# In[ ]:


qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())
chat_history = []
while True:
    query = input('\nQ: ') 
    if not query:
        break
    result = qa({"question": query + ' (用繁體中文回答)', "chat_history": chat_history})
    print('A:', result['answer'])
    chat_history.append((query, result['answer']))


# In[ ]:




