from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import torch
from tqdm import tqdm
from itertools import islice
from uuid import uuid4
from datasets import load_dataset
from dotenv import dotenv_values
from huggingface_hub import login
import argparse
import time

def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dataset_name', type=str, default="Zappu/legal-docs-vn",
                        help='Dataset name')
    parser.add_argument('-m', '--model_embed', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        help='Model name')
    parser.add_argument('-p', '--path_vtstore', type=str, default='VectorStore/Train_Legal_v3', help='Path to save vtstore')
    return parser.parse_args()     

def main():
    start_time = time.time()
    arg = parse()
    venv = dotenv_values('.env')
    login(token=venv["HF_TOKEN"], add_to_git_credential=True)

    dataset_name = arg.dataset_name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    cache_dir = r"../.cache"

    embeddings = HuggingFaceEmbeddings(
        model_name=arg.model_embed,
        cache_folder=cache_dir,
        model_kwargs=model_kwargs,
        show_progress=True,
    )

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    dataset = load_dataset(dataset_name, "corpus", split="train")

    chunk_size = 1000
    # create_vector_store_from_dataset(dataset, collection_name, vector_store, chunk_size)
    for i in range(0, len(dataset), chunk_size):
        # Lặp qua tất cả các split (train, test, validation)
        documents = []
        chunk = islice(dataset, i, i + chunk_size)
        for row in tqdm(chunk, desc=f"Processing split {chunk}"):
            text = row['context']
            if isinstance(text, list):
                    text = " ".join(text)  # Nối list thành string nếu cần
            # Tiền xử lý text nếu cần
            if '"' in text:
                text = text.replace('"', '')
            if r'\n' in text:
                text = text.replace(r'\n', '')
            if r'/' in text:
                text = text.replace(r'/', '')

            doc = Document(
                page_content=text,
                metadata={"cid": row['cid']},
            )
            documents.append(doc)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        print(f"Chunk: {i // chunk_size}/{len(dataset) // chunk_size + 1}")
        vector_store.add_documents(documents=documents, ids=uuids)

    faiss_path = arg.path_vtstore
    vector_store.save(faiss_path)
    print(f"Saved vector store to {faiss_path}")
    print("Time of processed create vectorstore faiss: " + "{:.4f} second".format(time.time() - start_time))

if __name__ == '__main__':
    main()