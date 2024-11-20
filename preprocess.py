import os, time
import ray
import argparse
import pandas as pd
from underthesea import sent_tokenize, text_normalize, word_tokenize, pos_tag, chunk, ner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
# import zipfile

ray.init()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'inference'], default='train', help='Mode of the script')
    parser.add_argument('--train_file', type=str, default='train.csv', help='Name of the training file')
    parser.add_argument('--test_file', type=str, default='public_test.csv', help='Name of the test file')
    parser.add_argument('--corpus_file', type=str, default='corpus.csv', help='Name of the corpus file')
    parser.add_argument('--data_path', type=str, default='Data', help='Path to the output directory (preprocessed data)')
    return parser.parse_args()

def process_data(text, file_stopwords):
    stopword = []
    with open(file_stopwords, 'r') as f:
        for line in f:
            stopword.append(line.strip())
    text = text_normalize(text)
    sentences = sent_tokenize(text)
    tokens = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            if word not in stopword:
                tokens.append(word)
    return ' '.join(tokens)

def list_to_string(list):
    try:
        return ' '.join(list)
    except:
        return ''
    
@ray.remote(num_cpus=1, num_gpus=0)
def read_csv_mutil(i, chunk, flag):
    # try:
    # print(f"Reading chunk {i}")
    for index, row in chunk.iterrows():
        match flag:
            case "corpus":
                context = process_data(row['text'], 'nltk/vn-stopwords.txt')
                chunk.at[index, 'text'] = context
            case "train":
                question = process_data(row['question'], 'nltk/vn-stopwords.txt')
                chunk.at[index, 'question'] = question
                context = process_data(list_to_string(row['context']), 'nltk/vn-stopwords.txt')
                chunk.at[index, 'context'] = context
            case "pubtest":
                question = process_data(row['question'], 'nltk/vn-stopwords.txt')
                chunk.at[index, 'question'] = question
    chunk.to_csv(f'Temp/{flag}_{i}.csv', index=False)
    return chunk
    # except Exception as e:
    #     print(e)
    #     return None

def main():
    arg = parse_args()
    corpus_file = os.path.join(arg.data_path, arg.corpus_file)
    train_file = os.path.join(arg.data_path, arg.train_file)
    test_file = os.path.join(arg.data_path, arg.test_file)
    # mode = arg.mode

    len_train = 119456
    len_corpus = 261597
    len_pubtest = 10000
    print("Đang đọc dữ liệu...")
    chunk_size = 10000
    # batch_size = 1000
    max_task = 8
    corpusdf = pd.DataFrame()
    traindf = pd.DataFrame()
    public_testdf = pd.DataFrame()
    cases = [['corpus', len_corpus, corpus_file], 
             ['train', len_train, train_file], 
             ['pubtest', len_pubtest, test_file]]
    # Đọc file csv 
    for case, len_row, path in cases:
        cnt = 0
        num_chunks = len_row // chunk_size + 1
        chunk_results = []
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
            cnt += 1
            print(f"Reading chunk {i} of {case} file")
            while cnt - 1 >= max_task:
                ready_ids, _ = ray.wait(chunk_results, num_returns=1)
                ready_id = ready_ids[0]
                chunk_results.remove(ready_id)
                cnt -= 1
            chunk_results.append(read_csv_mutil.remote(i, chunk, case))
        processed_chunks = ray.get(chunk_results)
        df = pd.concat(processed_chunks)
        match case:
            case "corpus":
                corpusdf = df
            case "train":
                traindf = df
            case "pubtest":
                public_testdf = df

        print(f"Done reading {case} file")
        time.sleep(2)

    # Bước 3: Xây dựng ma trận TF-IDF cho corpus
    print("Xây dựng ma trận TF-IDF cho corpus...")
    vectorizer = TfidfVectorizer(
        max_features=50000,  # Giới hạn số từ
        ngram_range=(1,2),    # Sử dụng unigram và bigram
        # stop_words='english'  # Có thể thay bằng danh sách stopwords tiếng Việt nếu có
    )
    corpus_tfidf = vectorizer.fit_transform(corpusdf['text'])
    print(f"Ma trận TF-IDF của corpus có kích thước: {corpus_tfidf.shape}")

    # Bước 4: Xây dựng ma trận TF-IDF cho các truy vấn trong train và public_test
    print("Xây dựng ma trận TF-IDF cho các truy vấn...")

    # Kết hợp các truy vấn từ train và public_test để đảm bảo vectorizer có đủ từ
    all_queries = pd.concat([traindf['question'], public_testdf['question']]).unique()
    vectorizer_queries = TfidfVectorizer(
        vocabulary=vectorizer.vocabulary_,
        ngram_range=(1,2)
    )
    queries_tfidf = vectorizer_queries.fit_transform(all_queries)
    print(f"Ma trận TF-IDF của các truy vấn có kích thước: {queries_tfidf.shape}")

    # Tạo từ điển qid -> vector
    query_ids = pd.concat([traindf[['qid', 'question']], public_testdf[['qid', 'question']]])
    query_ids = query_ids.drop_duplicates(subset='qid').set_index('qid')
    query_vectors = vectorizer.transform(query_ids['question'])

    # Bước 5: Tính toán độ tương đồng và dự đoán top k
    print("Tính toán độ tương đồng và dự đoán top k...")

    def get_top_k(cos_sim, k=10):
        top_k_indices = cos_sim.argsort()[-k:][::-1]
        return top_k_indices

    k = 10  # Số văn bản liên quan tối đa

    predict_results = []

    for qid, q_vec in tqdm(zip(query_ids.index, query_vectors), total=len(query_ids)):
        # Tính cosine similarity giữa truy vấn và tất cả các văn bản trong corpus
        cos_sim = cosine_similarity(q_vec, corpus_tfidf).flatten()
        top_k_idx = get_top_k(cos_sim, k)
        top_k_cids = corpusdf.iloc[top_k_idx]['cid'].values
        # Tạo chuỗi kết quả: qid followed by cids
        result = ' '.join([str(qid)] + [str(cid) for cid in top_k_cids])
        predict_results.append(result)

    # Bước 6: Lưu kết quả vào predict.txt và nén thành predict.zip
    print("Lưu kết quả vào predict.txt...")

    predict_path = os.path.join(arg.data_path, 'predict.txt')
    with open(predict_path, 'w', encoding='utf-8') as f:
        for line in predict_results:
            f.write(line + '\n')

    # print("Nén predict.txt thành predict.zip...")
    # with zipfile.ZipFile('predict.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    #     zipf.write('predict.txt')

    # print("Hoàn thành! Kết quả đã được lưu vào predict.zip")

if __name__ == "__main__":
    main()