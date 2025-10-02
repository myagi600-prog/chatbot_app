import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import io

# LangChain関連のライブラリをインポート
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Googleの代わりにHuggingFaceのEmbeddingモデルを利用
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ドキュメント読み込み用のライブラリ
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader

# --- APIキーの設定 ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# --- RAG関連の関数 ---
@st.cache_resource # 計算結果をキャッシュして高速化
def get_documents_text(uploaded_files):
    # （この関数は変更なし）
    text = ""
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        
        loader = None
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(uploaded_file.name)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(uploaded_file.name)
            elif file_extension == ".txt":
                loader = TextLoader(uploaded_file.name)
            elif file_extension in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(uploaded_file.name, mode="elements")
            
            if loader:
                documents = loader.load()
                text += "".join(doc.page_content for doc in documents) + "\n"

        except Exception as e:
            st.error(f"ファイル処理中にエラーが発生しました ({uploaded_file.name}): {e})")
        finally:
            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)
    return text

@st.cache_resource # 計算結果をキャッシュして高速化
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# st.cache_resourceを使って、重いモデルの読み込みやベクトルストアの計算をキャッシュ
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        # ベクトルストアが存在しない場合は新しく作成
        st.session_state.vector_store = DocArrayInMemorySearch.from_texts(text_chunks, embedding=embeddings)
    else:
        # 既存のベクトルストアにテキストを追加
        st.session_state.vector_store.add_texts(text_chunks)

def get_conversational_chain():
    prompt_template = """
    あなたは、あらゆる分野のエキスパートであり、提供された文脈情報に基づいて質問に回答するAIアシスタントです。
    初心者の方にも分かりやすく説明することを心がけてください。
    質問された内容のみに簡潔に回答してください。
    文脈情報にない内容や、質問と直接関係のない話題は出さないでください。
    ただし、質問に関連する補足情報やアドバイスがあれば、簡潔に提案する形で含めても構いません。

    文脈:
 {context}

    質問: 
{question}

    回答:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlitアプリのメイン部分 ---
st.title("SmartAssistant")
st.caption("サイドバーから知識ファイルをアップロードできます")

# --- サイドバー ---
with st.sidebar:
    st.header("知識ベース設定")
    
    knowledge_files = st.file_uploader(
        "知識ファイル（メモ帳/Word/Excelなど）をアップロード", 
        accept_multiple_files=True,
        type=["txt", "xlsx", "xls", "docx"]
    )
    if st.button("ファイルを処理して知識ベースを構築"):
        if knowledge_files:
            with st.spinner("ファイルを処理中...（初回はモデルのダウンロードに時間がかかります）"):
                raw_text = get_documents_text(knowledge_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("知識ベースの準備ができました！")
        else:
            st.warning("ファイルをアップロードしてください。")

    if st.button("知識ベースをクリア"):
        st.session_state.vector_store = None
        st.cache_resource.clear()
        st.success("知識ベースをクリアしました。")

# --- メインチャット画面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("メッセージを入力してください..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AIが考えています..."):
            if "vector_store" in st.session_state and st.session_state.vector_store is not None:
                vector_store = st.session_state.vector_store
                
                # HuggingFaceEmbeddingsをその場でインスタンス化
                embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
                
                # 類似度検索を実行し、関連ドキュメントとスコアを取得
                # k=3は取得するドキュメントの数
                docs_with_score = vector_store.similarity_search_with_score(prompt, k=3)
                
                # 類似度スコアの閾値設定 (スコアが小さいほど類似度が高い)
                # 閾値は実際の挙動を見ながら調整が必要
                relevance_threshold = 0.3 # 例: 距離が0.3未満を関連ありとする
                relevant_docs = [doc for doc, score in docs_with_score if score < relevance_threshold]

                if relevant_docs:
                    # RAGを実行する場合
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                    prompt_template = f"""
あなたは、あらゆる分野のエキスパートであり、初心者の方にも分かりやすく説明することを心がけるAIアシスタントです。
質問された内容に簡潔に回答してください。
提供された文脈情報に加えて、あなたの一般的な知識も活用して回答してください。
ただし、質問の主旨が文脈情報にある場合は、そちらを優先し、文脈情報に基づいて回答してください。
質問に関連する補足情報やアドバイスがあれば、簡潔に提案する形で含めても構いません。

文脈:\n {context}\n
質問: \n{prompt}\n
回答:
"""
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    response = model.generate_content(prompt_template)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                else:
                    # RAGをスキップし、汎用的な回答を生成する場合
                    st.info("知識ベースに関連する情報が見つかりませんでした。一般的な知識に基づいて回答します。")
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    response = model.generate_content(prompt) # 汎用的なプロンプトで質問
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
            else:
                st.warning("知識ベースが構築されていません。ファイルをアップロードして「ファイルを処理して知識ベースを構築」ボタンを押してください。")
                try:
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")