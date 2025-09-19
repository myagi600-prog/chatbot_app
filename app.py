import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

# --- APIキーの設定 ---
try:
    # StreamlitのSecretsからAPIキーを取得
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    # ローカル環境用のフォールバック
    api_key = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# --- モデルの初期設定 ---
model = genai.GenerativeModel('gemini-1.5-flash')

st.title("AIチャットボット (Gemini)")
st.caption("画像アップロード機能付き。画像の内容について質問できます。")

# --- チャット履歴の初期化 ---
# 画像を含む会話は履歴に残さず、テキストのみの会話履歴を管理
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# --- UIの表示 ---
# ファイルアップローダー
uploaded_file = st.file_uploader("画像をアップロードしてください...", type=["jpg", "jpeg", "png"])
image_to_display = None
if uploaded_file is not None:
    image_to_display = Image.open(uploaded_file)
    st.image(image_to_display, caption="アップロードされた画像", width=300)

# テキストのみのチャット履歴を表示
for message in st.session_state.chat.history:
    role = "You" if message.role == "user" else "AI"
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

# --- ユーザーからの入力とAIの応答 ---
if prompt := st.chat_input("メッセージを入力してください..."):
    # 画像がある場合：画像とテキストを送信（履歴には残らない）
    if uploaded_file is not None:
        with st.chat_message("You"):
            st.image(image_to_display, width=150)
            st.markdown(prompt)
        
        with st.chat_message("AI"):
            with st.spinner("AIが考えています..."):
                try:
                    response = model.generate_content([prompt, image_to_display])
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
    
    # 画像がない場合：テキストのみを送信（履歴に残る）
    else:
        with st.chat_message("You"):
            st.markdown(prompt)

        with st.chat_message("AI"):
            with st.spinner("AIが考えています..."):
                try:
                    response = st.session_state.chat.send_message(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
