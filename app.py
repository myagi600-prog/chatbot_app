import streamlit as st

# タイトルを設定
st.title("AI Chatbot")

# セッション状態でチャット履歴を初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 以前のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーの入力を待つ
if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーメッセージをチャット履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの応答（現在はオウム返し）を履歴に追加して表示
    response = f"Echo: {prompt}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
