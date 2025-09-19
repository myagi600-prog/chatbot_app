import streamlit as st
import google.generativeai as genai
import os

# StreamlitのSecretsからAPIキーを取得
# ローカル環境でテストする際は、環境変数に "GEMINI_API_KEY" を設定するか、
# 以下の行を api_key = "ご自身のAPIキー" に書き換えてください。
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    # Secrets.tomlがない、またはキーが設定されていない場合のフォールバック
    api_key = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# --- 初期設定 ---
# モデルの選択
model = genai.GenerativeModel('gemini-1.5-flash')

st.title("AIチャットボット (Powered by Gemini)")
st.caption("簡単な質問応答ができるAIチャットボットです。")

# セッションステートでチャット履歴を初期化
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# --- チャット履歴の表示 ---
# st.session_state.chat.history には user と model のやり取りが交互に格納される
for message in st.session_state.chat.history:
    role = "You" if message.role == "user" else "AI"
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

# --- ユーザーからの入力 ---
if prompt := st.chat_input("メッセージを入力してください..."):
    # ユーザーのメッセージを履歴に追加して表示
    with st.chat_message("You"):
        st.markdown(prompt)
    
    # AIからの応答を生成して表示
    with st.chat_message("AI"):
        with st.spinner("AIが考えています..."):
            try:
                response = st.session_state.chat.send_message(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.info("APIキーが正しく設定されているか、StreamlitのSecretsを確認してください。")