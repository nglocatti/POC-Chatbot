import azure.functions as func
import logging
import json
import time
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
app = func.FunctionApp(http_auth_level=func.AuthLevel.ADMIN)
llm = ChatOpenAI(
    openai_api_key="",
    model="gpt-4o-2024-08-06"
)
llm_history = ChatOpenAI(
    openai_api_key="",
    model="gpt-4o-mini-2024-07-18"
)
@app.route(route="chatsupabase")
def chatsupabase(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Invalid JSON payload.",
            status_code=400
        )
    messages = req_body.get('messages')
    if not messages or not isinstance(messages, list):
        return func.HttpResponse(
            "Missing or invalid 'messages' parameter.",
            status_code=400
        )
    # Extract the last message as the question
    last_message = messages[-1]
    question = last_message.get('message')
    if not question:
        return func.HttpResponse(
            "The last message does not contain a valid 'message' field.",
            status_code=400
        )
    # Process the messages to create the chat history
    chat_history = []
    for msg in messages[:-1]:
        if msg['sender_id'] == "assistant":
            chat_history.append(AIMessage(content=msg['message']))
        else:
            chat_history.append(HumanMessage(content=msg['message']))
    embeddings = OpenAIEmbeddings(
        openai_api_key="",
        model="text-embedding-3-small"
    )
    supabase_client = create_client("https://nfpzaafmcdeutkkqoxct.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5mcHphYWZtY2RldXRra3FveGN0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNjc2NzIwOSwiZXhwIjoyMDQyMzQzMjA5fQ.CFWB-W0TH08qTr4au2oMkMrVOqgBG0PLujBM_qpOKaM")
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )
    retriever = vector_store.as_retriever()
    contextualize_q_system_prompt = (
        "Dados un historial de chat y la última pregunta del usuario "
        "que podría hacer referencia al contexto en el historial de chat, "
        "formula una pregunta independiente que pueda ser entendida "
        "sin el historial de chat. NO respondas la pregunta, "
        "simplemente reformúlala si es necesario y, de lo contrario, devuélvela tal como está."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm_history, retriever, contextualize_q_prompt
    )
    system_prompt = (
        "Eres un asistente útil respondiendo preguntas relacionadas a una división de la empresa de software e inteligencia artificial llamada CREAI. "
        "Usa las siguientes piezas de contexto para responder "
        "la pregunta. Si no sabes la respuesta a la pregunta, di que "
        "no sabes. Usa máximo 3 oraciones y manten la "
        "respuesta consistente."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    # Create the new message with the AI response
    new_message = {
        "message": ai_msg["answer"],
        "timestamp": int(time.time()),
        "sender_id": "assistant",
        "type": "text"
    }
    return func.HttpResponse(
        json.dumps(new_message),
        status_code=200,
        mimetype="application/json"
    )