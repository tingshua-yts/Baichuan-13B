import streamlit as st
import logging
logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.DEBUG)
# if 'key' not in st.session_state:
#     st.session_state['key'] = 'value'
# v = st.session_state['key']
# logging.info(v)
# st.write(st.session_state['key'])
# # # Updates
# st.session_state.key = 'value2'     # Attribute API
# st.session_state['key'] = 'value2'  # Dictionary like API

def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是百川大模型，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
        logging.info("message in session_state")
    else:
        logging.info("message not in session_state")
        st.session_state["messages"] = 'a'
        messages = st.session_state["messages"]
        logging.info(f"message: {messages}")

    return st.session_state["messages"]
def main():
    # 使用huggingface接口，构建model和tokenizer
    #model, tokenizer = init_model()
    messages = init_chat_history()