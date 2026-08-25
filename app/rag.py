"""
RAG 코어 로직

이전 06_rag_agent.py와 다른 점:
- InMemoryVectorStore(메모리 휘발성) 대신 Chroma(디스크에 영구 저장) 사용
  -> 한번 인덱싱하면 프로그램 껐다 켜도 다시 안 만들어도 됨
- 여러 개의 .md 파일을 폴더째로 로딩 (실무에서 문서가 파일 여러 개인 상황 흉내)
- "모르면 모른다고 답해라" 라는 프롬프트로 hallucination 방지 (요구사항 문서의 성공 기준)
"""

import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "policies")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_vectorstore() -> Chroma:
    """data/policies 폴더의 .md 파일들을 읽어 Chroma에 인덱싱한다."""
    loader = DirectoryLoader(
        DATA_DIR, glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(raw_docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def load_vectorstore() -> Chroma:
    """이미 인덱싱된 DB가 있으면 그냥 불러온다."""
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return build_vectorstore()


vectorstore = load_vectorstore()


@tool
def search_hr_policy(query: str) -> str:
    """사내 HR 정책 문서(연차, 재택근무, 경비)에서 관련 내용을 검색한다."""
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "관련 정책 문서를 찾을 수 없음"
    return "\n---\n".join(r.page_content for r in results)


SYSTEM_PROMPT = (
    "너는 사내 HR 챗봇이다. 반드시 search_hr_policy 도구로 찾은 내용에만 근거해서 답변해라. "
    "도구 검색 결과에 관련 내용이 없으면 "
    "'문서에서 확인되지 않습니다. HR 담당자에게 문의해주세요.' 라고 답해라. "
    "절대로 지어내서 답변하지 마라."
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
agent = create_react_agent(llm, tools=[search_hr_policy], prompt=SYSTEM_PROMPT)


def ask(question: str) -> str:
    result = agent.invoke({"messages": [("user", question)]})
    content = result["messages"][-1].content
    # Gemini는 content를 문자열이 아니라 [{"type": "text", "text": "..."}] 형태로 줄 때가 있음
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


if __name__ == "__main__":
    print(ask("재택근무 신규 입사자도 바로 가능해?"))
    print(ask("퇴사 절차 알려줘"))  # 문서에 없는 질문 -> "모른다" 답변 확인용
