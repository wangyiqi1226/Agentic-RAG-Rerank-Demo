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
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
#初始向量库
embeddings_model=HuggingFaceEmbeddings(
    model="C:/Users/24005/.cache/modelscope/models/BAAI--bge-m3/snapshots/master",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)
#重排模型
rerank_model=HuggingFaceCrossEncoder(
    model_name="C:/Users/24005/.cache/modelscope/models/models/BAAI--bge-reranker-v2-m3/snapshots/master"
)
reranker=CrossEncoderReranker(model=rerank_model,top_n=3)
#文档加载
text_file=DirectoryLoader(
    path="./data",
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    loader_kwargs={"encoding":"utf-8"}
)
text_loader=text_file.load()
csv_file=DirectoryLoader(
    path="./data",
    glob="**/*.csv",
    loader_cls=CSVLoader,
    show_progress=True,
    loader_kwargs={"encoding":"utf-8"}
)
csv_loader=csv_file.load()
all_docs=text_loader+csv_loader
#文档分块
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=300
)
docs_splitter=splitter.split_documents(all_docs)
#构建向量库
db=Chroma.from_documents(
    documents=docs_splitter,
    embedding=embeddings_model,
    persist_directory="./chroma_db_agentic"
)
base_retriever=db.as_retriever(search_kwargs={"k":10})
compress=ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=reranker
)
#记忆会话
store={}
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]
#元数据过滤的检索
def filtered_retriever(input_dict:str):
    query=input_dict["question"]
    filter_condition=input_dict.get("filter")
    if filter_condition is not None:
        rew_docs=db.similarity_search(query,k=10,filter=filter_cindition)
    else:
        rew_docs=db.similarity_search(query,k=10)
    return reranker.compress_documents(rew_docs,query)
#查询链路
prompt_template=ChatPromptTemplate.from_template(
    """
    你是查询优化器，把用户原始问题改写成更适合知识库检索的检索问句，
    输出只返回改写后的问句，不要多余解释。
    原始问题：{question}
    """
)
rewrite_chain=prompt_template | chat_model | StrOutputParser()
anwer_prompt=ChatPromptTemplate.from_messages([
    ("system", """
    你是知识库问答助手，结合检索上下文和历史对话回答用户问题。
    回答时优先承接对话里的指代，读懂用户说的“它、他、这个”对应的对象。
    如果上下文没有对应信息，直接回复【知识库无相关内容】，禁止编造。
    回答末尾标注文档来源。
    """),
    MessagesPlaceholder("history"),
    ("human", "检索上下文：\n{context}\n用户当前问题：{question}")
])
#内容来源
def format_with_source(docs):
    if not docs:
        return ["无检索到相关文档"]
    lines=[]
    for i,doc in enumerate(docs):
        source=doc.metadata.get("source","未知来源")
        lines.append(f"---片段{i} | 文件：{source}---\n{doc.page_content}")
    return "\n".join(lines)
#完整链
agentic_rag_chain=(
    RunnablePassthrough.assign(
        prompt_template=rewrite_chain
    )
    |
    {
        "question":RunnableLambda(lambda x:x["question"]),
        "context":RunnableLambda(lambda x:x["prompt_template"]),
        "history":RunnableLambda(lambda x:x["history"]),
        "filter":RunnableLambda(lambda x:x.get("filter")),
        "retriever_docs":RunnableLambda(lambda x:filtered_retriever(x))
    }
    |
    RunnablePassthrough.assign(
        context=lambda x:format_with_source(x["retriever_docs"])
    )
    |anwer_prompt
    |chat_model
    |StrOutputParser()
)
#包装记忆链
history_chain=RunnableWithMessageHistory(
    agentic_rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
if __name__=="__main__":
    print("===== 进阶Agentic-RAG（查询改写+BGE重排+溯源+空检索兜底+会话记忆），输入exit退出 =====")
    print("指令示例：")
    print("  #filter source=./data/test.txt  只检索该文件")
    print("  #filter clear 取消过滤")
    session_id="demo_001"
    current_cond = None
    while True:
        q=input("\n请提问:")
        if q.strip().lower()=="exit":
            break
        if q.strip().lower()=="clear":
            store[session_id]=ChatMessageHistory()
            print("会话清除成功！")
            continue
        if q.startswith("#filter"):
            parts=q.split(" ",1)
            if len(parts)<2:
                print("格式错误！示例：#filter source=./data/test.txt")
                continue
            cond=parts[1]
            if cond=="clear_filter":
                current_cond=None
                print("已取消元数据过滤")
            else:
                k,v=cond.split("=")
                current_cond={k:v}
                print(f"过滤条件已设置:{current_cond}")
            continue
        resp=history_chain.invoke({"question":q,"filter":current_cond},config={"configurable":{"session_id":session_id}})
        for chunk in resp:
            print(chunk,end="",flush=True)