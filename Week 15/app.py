import os
import sys
import argparse
import pandas as pd
import ast
import re
import pysqlite3

from huggingface_hub import snapshot_download

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.chains.retrieval import create_retrieval_chain

from langchain_ollama import ChatOllama

# --- SQL Agent 관련 라이브러리 추가 ---
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType

# 오픈AI API 키 설정
sys.modules["sqlite3"] = pysqlite3 # ChromaDB 호환성
open_api_key = "your_key"
gemini_api_key = "your_key"

llm_model_dict = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", openai_api_key=open_api_key),
    "llama3:latest": ChatOllama(model="llama3:latest"),
    "gemini-flash-latest": ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=gemini_api_key),
    "gemini-2.5-flash": ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)
}

# --- RAG 및 SQL 에이전트 초기화 함수 ---
@st.cache_resource
def initialize_components(chroma_db_path, sqlite_db_path, embedding_model, llm_model):

    # --- RAG 체인 설정 (기존 코드) ---
    if not os.path.exists(chroma_db_path):
        snapshot_download(
            repo_id="SoccerData/namuwiki_db",
            repo_type="dataset",
            local_dir=chroma_db_path,
            local_dir_use_symlinks=False
        )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
    vector_db = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 10})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks regarding football.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If the answer is not present in the context, utilize your general knowledge to provide a helpful response.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    if llm_model in llm_model_dict.keys():
        llm = llm_model_dict[llm_model]
    else:
        print(f"Failed LLM Loading: {llm_model}")

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- SQL 에이전트 설정 ---
    SQL_AGENT_PROMPT = """
        # Role
        SQLite Expert. Convert questions to SQL queries for `player_stats`.

        # Schema
        Table: `player_stats`
        Columns:
        - game_number (int): Match ID
        - competition_name (text): e.g, '하나은행 K리그1 2024'
        - season_name (int): e.g., 2024
        - team_name (text): Korean name (e.g., '포항')
        - home_team (text): 'Home' or 'Away'
        - player_name (text): Korean name
        - jersey_number (int)
        - position (text): 'GK', 'DF', 'MF', 'FW', '대기'
        - tsg_rating (real): 0.0 to 10.0

        # Critical Rules
        1. Zero Rating: `tsg_rating` 0.0 means "No Rating". Must exclude 0.0 when calculating averages (Add `WHERE tsg_rating > 0`).
        2. Text Matching: Use exact Korean strings for `team_name` and `player_name`.

        # Example1
        User: "2024 포항 선수들 평균 평점"
        SQL: SELECT AVG(tsg_rating) FROM player_stats WHERE season_name = 2024 AND team_name = '포항' AND tsg_rating > 0;

        # Example2
        User: "2024 울산 vs 광주 경기 모든 선수 평점"
        SQL: SELECT player_name, tsg_rating FROM player_stats
        WHERE season_name = 2024 AND team_name IN ('울산', '광주') AND tsg_rating > 0;
    """
    
    db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")
    def run_query_save_results(db, query):
        res = db.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return res

    # 고유한 선수 이름 및 팀 이름 추출
    player_names = run_query_save_results(db, "SELECT DISTINCT player_name FROM player_stats;")
    team_names = run_query_save_results(db, "SELECT DISTINCT team_name FROM player_stats;")

    sql_vector_db = FAISS.from_texts(set(player_names + team_names), embeddings)
    sql_retriever = sql_vector_db.as_retriever()
    sql_retriever_tool = create_retriever_tool(
        sql_retriever,
        name="name_search",
        description="Searches for proper nouns present in the database, such as player names and team names.",
        )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    custom_suffix = """
        If the user requests filtering by a proper noun, you MUST first use the `name_search` tool to verify the exact spelling.
        Otherwise, examine the database tables to identify available information.
        Then, query the schema of the most relevant table.
    """

    sql_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=[sql_retriever_tool],
        prefix=SQL_AGENT_PROMPT,
        suffix=custom_suffix,
    )

    # --- 3. 라우터(Router) 에이전트 설정 ---
    # func: 복잡한 출력을 처리하기 위해 각 도구 함수는 최종 문자열 출력만 반환하도록 래핑합니다.
    # 사용자의 질문 유형에 따라 RAG 또는 SQL 에이전트를 선택적으로 호출하는 도구를 정의합니다.
    tools = [
        Tool(
            name="FootballKnowledgeSearch",
            func=lambda query: rag_chain.invoke({"input": query}).get("answer", "No answer found in the knowledge base."), 
            description=(
                "Useful for answering general questions about football history, rules, news, and player biographies. "
                "Use this for qualitative inquiries or unstructured text data. "
                "Example: 'Who is Messi?', 'Explain the offside rule'."
            ),
        ),
        Tool(
            name="FootballDataSearch",
            # run_sql_agent 함수 대신, lambda 함수로 직접 실행 및 결과 파싱을 정의합니다.
            func=lambda query: sql_agent_executor.invoke({"input": query}).get("output", "An error occurred during data analysis."),
            description=(
                "Useful for querying structured data regarding match statistics, player ratings, and team performance. "
                "Use this for quantitative questions involving specific numbers, scores, aggregations, or records. "
                "Example: 'What was the score of the 2024 Ulsan vs. Gwangju match?', 'Average rating of Pohang players'."
            ),
        ),
    ]

    # 라우터 에이전트 프롬프트
    ROUTER_PROMPT_TEMPLATE = """
        You are a friendly and knowledgeable AI expert on everything regarding football.
        Engage in natural conversation, maintaining context from previous turns.

        You have access to the following tools:
        1. `FootballKnowledgeSearch`: Use this for general inquiries about football history, rules, player biographies, or team backgrounds.
        2. `FootballDataSearch`: Use this for requests involving specific match statistics, scores, player ratings, or quantitative records.

        Decision Logic & Guidelines:
        - Chitchat & General Logic: For simple greetings (e.g., "Hi") or questions unrelated to football tools (e.g., "2+2"), answer directly without invoking any tools.
        - Complex Queries: If a user's question requires both qualitative background AND quantitative statistics, you MUST use both tools sequentially. Gather all necessary information and synthesize a comprehensive answer.
        - Language Constraint: Regardless of the internal reasoning language, your final response to the user MUST BE in Korean.

        User Question: {input}
    """

    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_PROMPT_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            # ✨ 에이전트의 중간 생각 과정을 위한 필수 공간
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # main_agent: 대화 전체 맥락을 관리하는 라우터 에이전트
    # history과 input을 받아 새로운 query를 생성하고, 적절한 도구를 호출하여 답변을 생성
    # 즉, tools에 들어가는 query는 사용자의 원래 질문이 아니라, history와 input을 반영한 새로운 질문임
    main_agent = create_tool_calling_agent(llm, tools, router_prompt)
    main_agent_executor = AgentExecutor(agent=main_agent, tools=tools, verbose=True)
    
    return main_agent_executor

def main():
    """
        메인 실행 함수: Argument 파싱 및 Streamlit UI 실행

        streamlit run app.py --server.port 8504 -- --chroma_db_path ./data/namuwiki_db \
        --sqlite_db_path player_stats_db \
        --embedding-model "jhgan/ko-sroberta-multitask" \
        --llm-model "gpt-4o-mini"
        
        streamlit run app.py --server.port 8503 -- --chroma_db_path ./data/namuwiki_db \
        --sqlite_db_path player_stats_db \
        --embedding-model "jhgan/ko-sroberta-multitask" \
        --llm-model "gemini-flash-latest"

        streamlit run app.py --server.port 8503 -- --chroma_db_path ./data/namuwiki_db \
        --sqlite_db_path player_stats_db \
        --embedding-model "jhgan/ko-sroberta-multitask" \
        --llm-model "gemini-2.5-flash"
        
        streamlit run app.py --server.port 8502 -- --chroma_db_path ./data/namuwiki_db \
        --sqlite_db_path player_stats_db \
        --embedding-model "jhgan/ko-sroberta-multitask" \
        --llm-model "llama3:latest"
    """

    # --- ArgumentParser 설정 ---
    parser = argparse.ArgumentParser(description="⚽ 축구 지식 & 데이터 챗봇")
    parser.add_argument("--chroma_db_path", type=str, default="./data/namuwiki_db", help="ChromaDB 벡터 저장소 경로")
    parser.add_argument("--sqlite_db_path", type=str, default="player_stats_db", help="SQLite 데이터베이스 파일 경로")
    parser.add_argument("--embedding-model", type=str, default="jhgan/ko-sroberta-multitask", help="HuggingFace 임베딩 모델 이름")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="OpenAI LLM 모델 이름")
    args = parser.parse_args()

    # --- Streamlit UI 설정 ---
    st.title("⚽ 축구 지식 & 데이터 챗봇")
    st.caption("나무위키 기반 지식 검색과 경기 데이터 분석이 모두 가능합니다.")

    with st.spinner("챗봇을 초기화하는 중입니다. DB 및 모델 로딩..."):
        main_agent_executor = initialize_components(
            chroma_db_path=args.chroma_db_path,
            sqlite_db_path=args.sqlite_db_path,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model
        )

    chat_history = StreamlitChatMessageHistory(key="chat_messages")
    conversational_agent = RunnableWithMessageHistory(
        main_agent_executor,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )

    if not chat_history.messages:
        chat_history.add_ai_message("축구에 대한 지식 질문이나 특정 경기 데이터에 대해 무엇이든 물어보세요!")

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt_message := st.chat_input("2023년 K-리그 경기에서 가장 높은 평점을 받은 선수는?"):
        st.chat_message("human").write(prompt_message)
        with st.chat_message("ai"):
            with st.spinner("생각 중..."):
                config = {"configurable": {"session_id": "any"}}
                response = conversational_agent.invoke({"input": prompt_message}, config)
                answer = response.get('output', '죄송합니다, 답변을 생성하는 데 문제가 발생했습니다.')
                if isinstance(answer, dict) and 'answer' in answer:
                    st.write(answer['answer'])
                else:
                    st.write(answer)

if __name__ == "__main__":
    main()
