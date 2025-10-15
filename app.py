import streamlit as st
import google.generativeai as genai
import os
import psycopg
import shutil
import urllib.parse

# LangChain関連のライブラリ
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

# ドキュメント読み込み用
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader

# --- 定数 ---
DEFAULT_SYSTEM_PROMPT = """
あなたは、あらゆる分野のエキスパートであり、初心者の方にも分かりやすく説明することを心がけるAIアシスタントです。
質問された内容に簡潔に回答してください。
提供された文脈情報があればそれを参考にし、なければあなたの一般的な知識を活用して回答してください。
質問に関連する補足情報やアドバイスがあれば、簡潔に提案する形で含めても構いません。
"""
COLLECTION_NAME = "chatbot_knowledge_base"

# --- データベース接続 & 初期設定 ---
try:
    raw_connection_string = st.secrets["DATABASE_URL"]
    parsed_url = urllib.parse.urlparse(raw_connection_string)

    scheme = parsed_url.scheme
    username = parsed_url.username
    password = parsed_url.password
    hostname = parsed_url.hostname
    port = parsed_url.port
    path = parsed_url.path

    encoded_password = urllib.parse.quote_plus(password) if password else ""

    if username and encoded_password:
        netloc = f"{username}:{encoded_password}@{hostname}"
    elif username:
        netloc = f"{username}@{hostname}"
    else:
        netloc = hostname

    if port:
        netloc = f"{netloc}:{port}"

    CONNECTION_STRING = urllib.parse.urlunparse((scheme, netloc, path, '', '', ''))
    # Store parsed components for direct psycopg.connect() calls
    PARSED_DB_COMPONENTS = {
        "host": hostname,
        "port": port,
        "dbname": path[1:] if path else "postgres",
        "user": username,
        "password": urllib.parse.unquote_plus(encoded_password) # Unquote for direct arg
    }

except (FileNotFoundError, KeyError):
    st.error("データベース接続情報 (DATABASE_URL) がSecretsに設定されていません。")
    st.stop()
except Exception as e:
    st.error(f"データベース接続情報の解析中にエラーが発生しました: {e}")
    st.stop()

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Gemini APIキー (GEMINI_API_KEY) がSecretsに設定されていません。")
    st.stop()

genai.configure(api_key=api_key)

# --- Embeddingモデルのキャッシュ ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

embeddings = load_embeddings()

# PGVectorストアのインスタンス化
vector_store = PGVector(
    connection=CONNECTION_STRING,
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    create_extension=False,
    engine_args={"connect_args": {"connect_timeout": 10}},
)

# --- データベース操作関数 (システムプロンプト用) ---
def get_system_prompt_from_db():
    try:
        with psycopg.connect(**PARSED_DB_COMPONENTS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM app_settings WHERE key = 'system_prompt'")
                result = cur.fetchone()
                if result:
                    return result[0]
    except Exception as e:
        st.error(f"データベースからのプロンプト読み込み中にエラー: {e}")
    return DEFAULT_SYSTEM_PROMPT

def save_system_prompt_to_db(prompt_text):
    try:
        with psycopg.connect(**PARSED_DB_COMPONENTS) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO app_settings (key, value) VALUES ('system_prompt', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (prompt_text,)
                )
        return True
    except Exception as e:
        st.error(f"データベースへのプロンプト保存中にエラー: {e}")
        return False

# --- データベース操作関数 (知識ベース用) ---
def get_knowledge_base_summary():
    total_docs = 0
    sources = []
    try:
        with psycopg.connect(**PARSED_DB_COMPONENTS) as conn:
            with conn.cursor() as cur:
                # コレクションIDを取得
                cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (COLLECTION_NAME,))
                collection_uuid = cur.fetchone()

                if collection_uuid:
                    collection_uuid = collection_uuid[0]
                    # ドキュメント数をカウント
                    cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s", (collection_uuid,))
                    total_docs = cur.fetchone()[0]

                    # ユニークなソースファイルを取得
                    cur.execute("SELECT DISTINCT cmetadata->>'source' FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'source' IS NOT NULL", (collection_uuid,))
                    sources = [row[0] for row in cur.fetchall()]
    except Exception as e:
        st.error(f"知識ベースの概要取得中にエラー: {e}")
    return total_docs, sources

def delete_documents_by_source(source_name):
    try:
        with psycopg.connect(**PARSED_DB_COMPONENTS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (COLLECTION_NAME,))
                collection_uuid = cur.fetchone()

                if collection_uuid:
                    collection_uuid = collection_uuid[0]
                    cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'source' = %s", (collection_uuid, source_name))
                    conn.commit()
                    st.success(f"ファイル '{source_name}' に関連する知識データを削除しました。")
                    return True
    except Exception as e:
        st.error(f"ファイル '{source_name}' の削除中にエラーが発生しました: {e}")
    return False

# --- RAG関連の関数 ---
def get_documents(uploaded_files):
    docs = []
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        
        loader = None
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(file_path, mode="elements")
            
            if loader:
                docs.extend(loader.load())

        except Exception as e:
            st.error(f"ファイル処理中にエラーが発生しました ({uploaded_file.name}): {e})")
    
    shutil.rmtree(temp_dir)
    return docs

from duckduckgo_search import DDGS

# --- Streamlitアプリのメイン部分 ---
st.title("SmartAssistant")
st.caption("サイドバーから知識ファイルをアップロードできます。`#`で始まるメッセージでAIの役割を変更できます。")

# --- セッションステートの初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = get_system_prompt_from_db()

# --- サイドバー ---
with st.sidebar:
    st.header("知識ベース設定")

    knowledge_files = st.file_uploader(
        "知識ファイル（メモ帳/Word/Excelなど）をアップロード",
        accept_multiple_files=True,
        type=["txt", "docx", "xlsx", "xls"]
    )
    if st.button("ファイルを処理して知識ベースに追加"):
        if knowledge_files:
            with st.spinner("ファイルを処理し、データベースに追加中..."):
                documents = get_documents(knowledge_files)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                vector_store.add_documents(chunks)
                st.success("知識ベースへの追加が完了しました！")
                st.rerun() # 知識ベースの概要を更新
        else:
            st.warning("ファイルを選択してください。")

    st.subheader("現在の知識ベース")
    total_docs, sources = get_knowledge_base_summary()
    st.write(f"登録ドキュメント数: {total_docs}")
    if sources:
        st.write("登録ファイル:")
        selected_source = st.selectbox("削除するファイルを選択", ["--選択してください--"] + sources)
        if selected_source != "--選択してください--":
            if st.button(f"'{selected_source}' を削除"):
                if delete_documents_by_source(selected_source):
                    st.rerun() # 知識ベースの概要を更新
    else:
        st.info("知識ベースにファイルは登録されていません。")

    st.header("現在のAIへの指示")
    st.info(st.session_state.system_prompt)

# --- メインチャット画面 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("メッセージを入力してください..."):
    if prompt.startswith("#"):
        new_prompt = prompt[1:].strip()
        if new_prompt:
            # 新しいデフォルトプロンプトを提案
            SUGGESTED_SYSTEM_PROMPT = """
あなたは、あらゆる分野のエキスパートであり、初心者の方にも分かりやすく説明することを心がけるAIアシスタントです。
提供された「知識ベースの情報」と「Web検索結果」の両方を参考に、情報の正確性を検証し、質問に対して最適な回答を生成してください。
もし情報源によって内容が矛盾する場合は、その点を指摘し、最も信頼性が高いと考えられる情報を提示してください。
回答は簡潔にまとめ、質問に関連する補足情報やアドバイスがあれば、提案する形で含めても構いません。
"""
            if new_prompt == "デフォルトに戻す":
                 new_prompt = SUGGESTED_SYSTEM_PROMPT

            if save_system_prompt_to_db(new_prompt):
                st.session_state.system_prompt = new_prompt
                st.success("AIへの指示を更新し、データベースに保存しました！")
                st.rerun()
        else:
            st.warning("#に続けて指示内容を入力してください。'#デフォルトに戻す'で推奨プロンプトをセットします。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AIが考えています...（Web検索中）"):
                # 1. 知識ベースから情報を取得
                rag_context = ""
                try:
                    docs = vector_store.similarity_search(prompt, k=3)
                    if docs:
                        rag_context = "\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    st.warning(f"知識ベースの検索中にエラーが発生しました: {e}")

                # 2. Web検索を実行
                web_context = ""
                st.write("デバッグ: Web検索処理を開始します。")
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(prompt, max_results=5))
                        if results:
                            web_context = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
                            st.write("デバッグ: Web検索が成功し、結果を取得しました。")
                        else:
                            st.write("デバッグ: Web検索は成功しましたが、結果は0件でした。")
                except Exception as e:
                    st.error(f"デバッグ: Web検索中に例外が発生しました。エラータイプ: {type(e).__name__}, エラーメッセージ: {e}")
                    web_context = f"Web検索中にエラーが発生しました: {e}" # コンテキストにもエラー情報を渡す

                # --- デバッグ用: 検索結果の表示 ---
                with st.expander("【デバッグ情報】AIに渡された参考情報"):
                    st.subheader("知識ベースの情報")
                    st.text(rag_context if rag_context else "関連情報なし")
                    st.subheader("Web検索結果")
                    st.text(web_context if web_context else "関連情報なし")
                # ------------------------------------ 

                # 3. AIに渡す最終的なプロンプトを構築
                final_prompt = f"""{st.session_state.system_prompt}

---
### 参考情報
#### 知識ベースの情報
{rag_context if rag_context else "関連情報なし"}

#### Web検索結果
{web_context if web_context else "関連情報なし"}
---

### 質問
{prompt}

### 回答
"""
                try:
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    response = model.generate_content(final_prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"AIからの応答取得中にエラーが発生しました: {e}")