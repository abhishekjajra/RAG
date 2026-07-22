import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


class RAGPipeline:

    def __init__(self):
        load_dotenv()
    
        self.connection = (
            f"postgresql+psycopg://"
            f"{os.getenv('POSTGRES_USER')}:"
            f"{os.getenv('POSTGRES_PASSWORD')}@"
            f"{os.getenv('POSTGRES_HOST')}:"
            f"{os.getenv('POSTGRES_PORT')}/"
            f"{os.getenv('POSTGRES_DB')}"
        )
        
    
        os.environ["Groq_API_Key"] = os.getenv("Groq_API_Key")
        self.engine = create_engine(self.connection)

    # load data
    def load_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        self.documents = loader.load() 
        print(f"Loaded documents count: {len(self.documents)}")

    # split data
    def Split_doc(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunk = splitter.split_documents(self.documents)
        print(f"Split chunks count: {len(self.chunk)}")

    # embedding model
    def create_embedding_model(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # to prevent duplicate chunks 
    def collection_exists(self):  
        query = """
            SELECT COUNT(*)
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c
            ON e.collection_id = c.uuid
            WHERE c.name = :collection;
            """

        with self.engine.connect() as conn:
            result = conn.execute(
                text(query),
                {"collection": "constitution"}
            ).scalar()

        return result > 0

    # vector
    def vector_store(self):
        self.vector = PGVector(
            embeddings=self.embedding_model,
            collection_name="constitution",
            connection=self.connection
        )
        if self.collection_exists():
            print("Vector store already exists")
        else:
            print("Creating Vector Store")
            self.vector.add_documents(self.chunk)
    
    # Retriever
    """def create_retriever(self):
        self.retriever = self.vector.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )"""
    def create_retriever(self):
        self.retriever = self.vector.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
        )
    # LLM Model
    def LLm_model(self):
        self.model = ChatGroq(
            model="llama-3.1-8b-instant"
        )

    def create_prompt(self):
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an authoritative expert in the Constitution of India, possessing deep knowledge of constitutional law, landmark Supreme Court judgements, and legislative amendments.

            Analyze the user's question strictly using the provided context from the official document.

            Rules for your response:
            1. **Factual and Grounded:** Base your response *only* on the provided context. If the answer cannot be found in the context, state: "I cannot find the answer to this question in the provided constitutional text." Do not guess.
            2. **Legal Precision:** Cite specific Articles, Parts, Schedules, or Clauses mentioned in the text whenever relevant.
            3. **Formatting:** Structure long answers using clear bullet points or numbered lists. Keep sentences punchy, objective, and neutral.
            4. **Tone:** Maintain a professional, academic, and non-partisan legal tone.
            

            Context:
            {context}

            Question:
            {input}

            Answer:
            """
        )

    # rag chain 
    def create_ragchain(self):
        self.doc_chain = create_stuff_documents_chain(
            self.model, self.prompt
        )
        self.rag_chain = create_retrieval_chain(
            self.retriever,self.doc_chain 
        )

    # Complete RAG Pipeline
    def build_pipeline(self, pdf_path):
        self.load_pdf(pdf_path)
        self.Split_doc()  
        self.create_embedding_model()
        self.vector_store()
        self.create_retriever()
        self.LLm_model()
        self.create_prompt()
        self.create_ragchain()
        return self

    # chat box
    def ask(self, question):
        response = self.rag_chain.invoke({"input": question}) 
        return response["answer"]

    def chat(self):
        print("="*50)
        print("Constitution of India RAG Chatbot")
        print("="*50)

        while True:
            question = input("\nQuestion (type 'exit' to quit): ")  
            if question.lower() == "exit":
                print("CHAT END!")
                break

            answer = self.ask(question)
    

            print("\nAnswer:")
            print(answer)
         

def main():
    rag = RAGPipeline()
    rag.build_pipeline("ICI.pdf")
    rag.chat()


if __name__ == "__main__":
    main()
