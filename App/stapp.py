import streamlit as st
from backend import load_and_split_documents, initialize_vectorstore, respond_to_query, load_and_split_websites

# Streamlit Frontend
st.title('💬 Write your questions')
st.sidebar.title("Chat History")
app = st.session_state

# Initialize session state if not present
if "messages" not in app:
    app["messages"] = [{"role": "assistant", "content": "I'm ready to retrieve information"}]

if 'history' not in app:
    app['history'] = []

if 'full_response' not in app:
    app['full_response'] = '' 

# Choose between PDF and web URL
option = st.radio("Select input source:", ("Upload PDF", "Enter Web URL"))

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file is not None:
        documents = load_and_split_documents(uploaded_file)
        vectorstore = initialize_vectorstore(documents)
        retriever = vectorstore.as_retriever()
elif option == "Enter Web URL":
    url = st.text_input("Enter Web URL")
    
    if url:
        documents = load_and_split_websites(url)
        vectorstore = initialize_vectorstore(documents)
        retriever = vectorstore.as_retriever()

# Display existing messages
for msg in app["messages"]:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="😎").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar="👾").write(msg["content"])

if txt := st.chat_input():
    app["messages"].append({"role": "user", "content": txt})
    st.chat_message("user", avatar="😎").write(txt)

    # Show spinner while waiting for the response
    with st.spinner('Generating response...'):
        response = respond_to_query(app["messages"], retriever)
    
    # Update session state and display the response
    app["messages"].append({"role": "assistant", "content": response})
    app["full_response"] = response

    st.chat_message("assistant", avatar="👾").write(response)
    
    app['history'].append("😎: " + txt)
    app['history'].append("👾: " + response)
    st.sidebar.markdown("<br />".join(app['history']) + "<br /><br />", unsafe_allow_html=True)
else:
    st.write("Select an input source and provide the necessary input to get started.")
