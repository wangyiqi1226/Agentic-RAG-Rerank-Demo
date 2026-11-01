import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import dotenv

dotenv.load_dotenv()
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
# 初始大模型llm
chat_model = ChatOpenAI(
    model="deepseek-flash",
    temperature=0
)
from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 初始向量模型
embeddings_model = HuggingFaceEmbeddings(
    model="C:/Users/24005/.cache/modelscope/models/BAAI--bge-m3/snapshots/master",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
# 重排模型
rerank_model = HuggingFaceCrossEncoder(
    model_name="C:/Users/24005/.cache/modelscope/models/models/BAAI--bge-reranker-v2-m3/snapshots/master"
)
reranker = CrossEncoderReranker(model=rerank_model, top_n=3)
# 文档加载
text_file = DirectoryLoader(
    path="./data",
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    loader_kwargs={"encoding": "utf-8"}
)
text_loader = text_file.load()
csv_file = DirectoryLoader(
    path="./data",
    glob="**/*.csv",
    loader_cls=CSVLoader,
    show_progress=True,
    loader_kwargs={"encoding": "utf-8"}
)
csv_loader = csv_file.load()
all_docs = text_loader + csv_loader
# 文档分割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50
)
docs_splitter = splitter.split_documents(all_docs)
# 构建向量库
db = Chroma.from_documents(
    documents=docs_splitter,
    embedding=embeddings_model,
    persist_directory="./chroma_db_agentic"
)
base_retriever = db.as_retriever(search_kwargs={"k": 4})
compression = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=reranker
)
# 查询链路
prompt_template = ChatPromptTemplate.from_template(
    """
    你是查询优化器，把用户原始问题改写成更适合知识库检索的检索问句，
    输出只返回改写后的问句，不要多余解释。
    原始问题：{question}
    """
)
rewrite_chain = prompt_template | chat_model | StrOutputParser()
# 模型回答
answer_prompt = ChatPromptTemplate.from_template(
    """
    你是知识库问答助手，严格依据下面提供的检索片段回答。
    如果信息不足，直接说明【知识库无相关内容】，禁止编造。
    回答结尾列出用到的片段来源。

    【检索上下文】
    {context}

    【用户问题】
    {question}
    """
)


# 内容来源
def format_with_source(docs):
    lines = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知资源")
        lines.append(f"---片段{i} | 文件：{source}---\n{doc.page_content}")
    return "\n".join(lines)


# 完整链
agentic_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "prompt_template": rewrite_chain
        }
        |RunnablePassthrough()
        |
        {
            "context":RunnableLambda(lambda x:x["prompt_template"]) | compression | format_with_source,
            "question":RunnableLambda(lambda x:x["question"])
        }
        | answer_prompt
        | chat_model
        | StrOutputParser()
)
if __name__ == "__main__":
    print("===== 进阶Agentic-RAG（查询改写+BGE重排+溯源），输入exit退出 =====")
    while True:
        q = input("\n请提问:")
        if q.strip().lower() == "exit":
            break
        resp = agentic_rag_chain.invoke({"question": q})
        print("\n回答:\n", resp)