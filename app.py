from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import openai
import gradio as gr
from langchain.document_loaders import PyPDFLoader
import os

history = []

def run(query, file, temperature, Chain, chunk_size, chunk_overlap, clear_history):
    global history  # make sure we're modifying the global history variable

    if clear_history:
        history = []
        return "", ""  # return empty strings if we're just clearing history

    os.environ["OPENAI_API_KEY"] = "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"
   
    loader = PyPDFLoader(file.name)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
 
    qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo"), chain_type=Chain, retriever=docsearch.as_retriever(), return_source_documents=True)
   
    result = qa({"query": query})
    answer = result["result"] + "\n\n_________s o u r c e ______\n\n_" + ('\n\n_--------------------------------------------------------------------\n\n_'.join(doc.page_content for doc in result["source_documents"]))

    history.append(("Question: " + query, "Answer: " + answer))

    history_str = "\n".join(f"{q}\n{a}" for q, a in history)

    return answer, history_str

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(lines=8, placeholder="Enter your question here. It is best to start with 'Based on the text provided'...you can also specify the length / purpose of the answer", label="Your question"),
        gr.File(label="Upload PDF file"), 
        gr.Slider(0, 1, value=0, label="Temperature", info="Choose between zero for accurate and precise answers and one for random/creative answers"),
        gr.Radio(["stuff", "map_reduce", "refine", "map_rerank"], label="Chain Type", info="Please select 'stuff' for general use", value="stuff"),
        gr.Slider(5, 1000, value=1000, label="Chunk Size", info="Standard setting 1000"),
        gr.Slider(0, 200, value=0, label="Chunk Overlap", info="Standard setting 0"),
        gr.inputs.Checkbox(label="Clear History")
    ],
    outputs=[
        gr.outputs.Textbox(label="Answer"),
        gr.outputs.Textbox(label="Chat History")
    ],
    title="Question Answering",
    description="Use this application to question documents. Please upload a PDF document.",
)

demo.launch(debug=True)