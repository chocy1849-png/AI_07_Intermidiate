import os
import warnings
from typing import Optional
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")

# 1. 환경 및 API 키 설정 (본인의 키로 변경하세요)
os.environ["OPENAI_API_KEY"] = ""
# 1. DB 및 모델 세팅
DB_PATH = "./my_rfp_vectordb"
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'}
)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

# 2. LLM이 출력할 '필터 조건'의 구조 정의 (Pydantic)
class SearchPlan(BaseModel):
    search_query: str = Field(description="공고번호를 제외한, 본문에서 검색할 핵심 요약 질문")
    notice_id: Optional[str] = Field(description="질문에 포함된 10자리 이상의 숫자 공고번호. 없으면 null", default=None)

# LLM에 구조화된 출력 강제 적용 (이 기능이 SelfQueryRetriever를 완벽히 대체합니다)
structured_llm = llm.with_structured_output(SearchPlan)

# 3. 에이전틱 스마트 검색 함수
def agentic_retriever(user_query):
    # LLM이 질문을 분석해서 검색 계획을 세움
    plan = structured_llm.invoke(user_query)
    print(f"\n[시스템] LLM 분석 결과: 검색어='{plan.search_query}', 공고번호='{plan.notice_id}'")
    
    if plan.notice_id:
        print("[시스템] 🎯 메타데이터 필터 검색 실행")
        docs = vectorstore.similarity_search(
            query=plan.search_query,
            k=5,
            filter={"id": {"$contains": plan.notice_id}}
        )
    else:
        print("[시스템] 🔍 일반 의미 검색 실행")
        docs = vectorstore.similarity_search(
            query=plan.search_query,
            k=3
        )
    return docs

# 4. 프롬프트 및 문서 포맷팅
prompt_template = """입찰메이트 수석 컨설턴트로서 아래 문맥을 바탕으로 답해.
문서에 없는 내용은 "해당 정보는 확인할 수 없습니다"라고 해.

[문맥]
{context}

[질문]
{question}

답변:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join([f"공고번호(ID): {doc.metadata.get('id', 'N/A')}\n내용: {doc.page_content}" for doc in docs])

# 5. 실행 테스트
if __name__ == "__main__":
    query = "한영대학교 사업의 입찰 참여 마감일은 언제인가요?"
    print(f"👤 질의: {query}")
    
    # 검색 (LLM이 알아서 필터링)
    retrieved_docs = agentic_retriever(query)
    formatted_context = format_docs(retrieved_docs)
    
    # 답변 생성
    final_chain = prompt | llm | StrOutputParser()
    response = final_chain.invoke({
        "context": formatted_context,
        "question": query
    })
    
    print("\n🤖 답변:")
    print(response)