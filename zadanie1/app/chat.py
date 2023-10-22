from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from app.loader import MyDocLoader


class CHATGPT:
    def __init__(self):
        llm = LlamaCpp(
            model_path="app/llama-2-7b-chat.Q4_K_M.gguf",
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            verbose=False,
        )

        template = """Используя следующий контекст ответь на вопрос. Отвечай на русском языке.
        Если ты не знаешь ответ, скажи что не знаешь, не пытайся придумать ответ.
        Контекст: {context}
        Вопрос: {question}
        Выведи только полезный ответ и ничего более.
        Ответ:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question'])
        loader = DirectoryLoader("app/docs/", glob="*", loader_cls=MyDocLoader)
        texts = loader.load()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder="app/sentences",
        )
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={'k': 6})

        self.qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})

    def __call__(self, question):
        return self.qa_llm({'query': question})
