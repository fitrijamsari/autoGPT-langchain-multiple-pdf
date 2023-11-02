# INTRODUCTION

- Purpose: To allow chat conversation with bot regarding the content after uploading multiple PDFs files.
- General Description: The project will enable autoGPT chat upon uploading multiple PDFs files
- Technology Stack: OpenAI, Langchain, Streamlit, FAISS

## INSTALLATION

1. Setup a new conda environment

```
conda create --name ENV_NAME python
```

2. Install the environment dependencies

```
pip install streamlit langchain openai PyPDF2 huggingface-hub faiss
```

or

```
pip install -r requirements.txt
```

3. Create .env file and insert your own private key

```
OPENAI_API_KEY=""
HUGGIGFACEHUB_API_TOKEN=""
```

4. Run the code on streamlit

```
streamlit run app.py
```
