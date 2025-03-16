import os
import gradio as gr
from langchain_groq import ChatGroq  # Groq for LLM
from langchain_huggingface import HuggingFaceEmbeddings  # Updated BERT embeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

# 1. Set API Keys 
os.environ["GROQ_API_KEY"] = "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"  

# 2. Load and Split a Text Document
file_path = "sample.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please ensure it exists in the same directory.")

loader = TextLoader(file_path)
documents = loader.load()

# Splitting text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. Use BERT Embeddings from Hugging Face
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create FAISS Vector Store with BERT Embeddings
vector_store = FAISS.from_documents(docs, embedding_function)
retriever = vector_store.as_retriever()

# 5. Set Up Groq Chat Model and RetrievalQA Chain
llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0.5)  # Using Groq Mixtral Model
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 6. Chat function that updates conversation history
def respond(message, chat_history):
    try:
        answer = qa_chain.run(message)
    except Exception as e:
        answer = f"⚠️ Error: {e}"
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history

# 7. Create a Gradio Blocks interface with a chatbot and input textbox
with gr.Blocks(css=".gradio-container {background-color: #f7f7f7;}") as demo:
    gr.Markdown("# LangChain Chatbot with Groq + BERT Embeddings")
    gr.Markdown("### Ask any question about the loaded document:")
    
    chatbot = gr.Chatbot(label="Chatbot Conversation", type="messages")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message here...", lines=2)
    
    txt.submit(respond, [txt, state], [txt, chatbot], queue=True)

if __name__ == "__main__":
    demo.launch()
