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
@st.cache_resource
def get_vector_store(_text_chunks): # アンダースコアはキャッシュのための慣習
    # HuggingFaceのオープンソースモデルを利用
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = DocArrayInMemorySearch.from_texts(_text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    以下の文脈情報に基づいて、質問にできるだけ詳しく回答してください。
    文脈:\n {context}?\n
    質問: \n{question}\n
    回答:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlitアプリのメイン部分 ---
st.title("AIチャットボット (RAG機能付き)")
st.caption("サイドバーから知識ファイルをアップロードできます")

# --- サイドバー ---
with st.sidebar:
    st.header("知識ベース設定")
    
    knowledge_files = st.file_uploader(
        "知識ファイル（メモ帳/Excel/画像など）をアップロード", 
        accept_multiple_files=True,
        type=["txt", "xlsx", "xls", "png", "jpg", "jpeg"]
    )
    if st.button("ファイルを処理して知識ベースを構築"):
        if knowledge_files:
            with st.spinner("ファイルを処理中...（初回はモデルのダウンロードに時間がかかります）"):
                raw_text = get_documents_text(knowledge_files)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
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
            if "vector_store" in st.session_state:
                try:
                    vector_store = st.session_state.vector_store
                    docs = vector_store.similarity_search(prompt)
                    
                    # LangChainを介さず、手動でプロンプトを構築して直接APIを呼び出す
                    context = "\n".join([doc.page_content for doc in docs])
                    prompt_template = f"""
以下の文脈情報に基づいて、質問にできるだけ詳しく回答してください。
文脈:\n {context}\n
質問: \n{prompt}\n
回答:
"""
                    
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    response = model.generate_content(prompt_template)
                    
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
            else:
                try:
                    model = genai.GenerativeModel('models/gemini-pro-latest')
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")