import json
import torch
import streamlit as st
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.INFO)


st.set_page_config(page_title="Baichuan-13B-Chat")
st.title("Baichuan-13B-Chat")


@st.cache_resource
def init_model():
    model_name = "baichuan-inc/Baichuan-13B-Chat"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    # 添加一个chat message container，其实就是message最前面的小图标
    with st.chat_message("assistant", avatar='🤖'):
        # with中为当前这个chat 图标所对应的message
        st.markdown("您好，我是百川大模型，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            # 根据avatar类型的不同个，创建对应的chat message
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
        logging.info("message in session_state")
    else:
        logging.info("message not in session_state")
        st.session_state["messages"] = []
        messages = st.session_state["messages"]
        logging.info(f"message: {messages}")

    return st.session_state["messages"]


def main():
    # 使用huggingface接口，构建model和tokenizer
    model, tokenizer = init_model()
    messages = init_chat_history()

    # chat_input用于创建一个inputwidget
    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        # 展示用户刚刚的输入
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        # 在message中添加用户的输入信息
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)

        # 输出assistant对应的信息
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                # 输出model chat的返回结果
                placeholder.markdown(response)
                # if torch.backends.mps.is_available():
                #     torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
