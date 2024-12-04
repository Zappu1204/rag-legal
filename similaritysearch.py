from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import ray
from langchain_core.documents import Document
import time
import math
import torch
from tqdm import tqdm
from itertools import islice
from uuid import uuid4
from datasets import load_dataset
from dotenv import dotenv_values
from huggingface_hub import login

ray.init()

venv = dotenv_values('.env')
login(token=venv["HF_TOKEN"], add_to_git_credential=True)

MODEL_EMBED = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
dataset_name = "Zappu/legal-docs-vn"#"Zappu/Legal-vn"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_kwargs = {'device': device}
cache_dir = r"../.cache"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_EMBED,
    cache_folder=cache_dir,
    model_kwargs=model_kwargs,
    show_progress=True,
)

dataset = load_dataset(dataset_name, "train", split="train")

@ray.remote(num_cpus=1, num_gpus=0)
def predict_by_chunk(i, chunk, vtstore_ref):
    start_time = time.time()
    predict_path = f"Predicts/predict_{i:03d}.txt"
    with open(predict_path, 'w') as p:
        for row in tqdm(chunk, desc=f"Processing split {chunk} {i}"):
            content = row['question']
            qid = row['qid']
            real_cid = row['cid']

            results = vtstore_ref.similarity_search(
                content,
                k = 8,
            )
            cids = [doc.metadata['cid'] for doc in results]  # Assuming 'qid' is what you meant by cid
            more_cid = f"{real_cid}"
            for id in cids:
                more_cid += f" {id}"
            response = f"{qid} {more_cid}"

            p.write(response + '\n')
    return f"Time of processed {i} chunk: " + "{:.4f} second".format(time.time() - start_time)

faiss_path = "VectorStore/Train_Legal_v2"
vtstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
vtstore_ref = ray.put(vtstore)

chunk_size = 1000
cnt = 0
chunk_results = []
max_task = 8
total_chunk = math.ceil(len(dataset) / chunk_size)
for i in range(0, total_chunk):
    cnt += 1
    chunk = dataset.shard(num_shards=total_chunk, index=i)
    while cnt - 1 >= max_task:
        ready_ids, _ = ray.wait(chunk_results, num_returns=1)
        ready_id = ready_ids[0]
        chunk_results.remove(ready_id)
        cnt -= 1
    chunk_results.append(predict_by_chunk.remote(i, chunk, vtstore_ref))
processed_chunks = ray.get(chunk_results)
with open("Predicts/processed_chunks.txt", 'a') as f:
    for chunk_result in processed_chunks:
        f.write(chunk_result + '\n')
ray.shutdown()