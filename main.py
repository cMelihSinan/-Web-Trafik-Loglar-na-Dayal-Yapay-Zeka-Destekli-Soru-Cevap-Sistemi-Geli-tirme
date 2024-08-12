import os
import re
import numpy as np
import faiss
import openai
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths
access_log_path = "/Users/melihsinancubukcuoglu/Desktop/access_log kopyası.txt"
error_log_path = "/Users/melihsinancubukcuoglu/Desktop/error_log kopyası.txt"

# OpenAI API key
openai.api_key = "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"

def parse_access_log_line(line):
    """
    Parse an access log line into a dictionary.
    """
    pattern = r'(?P<ip>[\d\.:]+) - - \[(?P<time>[^\]]+)\] "(?P<request>[^"]+)" (?P<status>\d{3}) (?P<size>\d*)'
    match = re.match(pattern, line)
    if match:
        data = match.groupdict()
        if data['request'] and data['request'] != '-':
            return {
                'ip': data['ip'],
                'time': data['time'],
                'request': data['request'],
                'status': data['status'],
                'size': data['size']
            }
    print(f"Line does not match access log format: {line.strip()}")
    return None

def parse_log_file(file_path, log_type):
    """
    Parse a log file into a list of log entries.
    """
    entries = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        print(f"Read {len(lines)} lines from {file_path}")

        for line in lines:
            print(f"Processing line: {line.strip()}")
            if log_type == 'access':
                entry = parse_access_log_line(line)
                if entry:
                    entries.append(
                        f"IP: {entry['ip']}, Time: {entry['time']}, Request: {entry['request']}, Status: {entry['status']}")
            elif log_type == 'error' and line.strip():
                entries.append(line.strip())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    if not entries:
        print(f"No valid log entries found in {file_path}")
    return entries

def create_vectors(log_entries, vectorizer=None):
    """
    Create vectors from log entries using a vectorizer.
    """
    if not log_entries:
        raise ValueError("Log entries are empty. Check the log file parsing.")
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        log_vectors = vectorizer.fit_transform(log_entries)
    else:
        log_vectors = vectorizer.transform(log_entries)
    print(f"Created vectors with shape: {log_vectors.shape}")
    return log_vectors, vectorizer

def add_vectors_to_index(index, vectors):
    """
    Add vectors to a FAISS index.
    """
    if vectors.shape[0] == 0:
        print("No vectors to add to the index.")
        return
    vectors_np = vectors.toarray().astype(np.float32)
    print(f"Adding vectors with shape: {vectors_np.shape}")
    if vectors_np.shape[1] != index.d:
        raise ValueError("Dimension mismatch between vectors and index.")
    index.add(vectors_np)


def ask_question(question, index, vectorizer, logs):
    """
    Ask a question and get an answer from the OpenAI API.
    """
    question_vector = vectorizer.transform([question])
    question_np = question_vector.toarray().astype(np.float32)
    D, I = index.search(question_np, k=5)

    retrieved_logs = [logs[i] for i in I[0] if i < len(logs)]
    if not retrieved_logs:
        print("No relevant logs found.")
        return "No relevant logs found."
    context = " ".join(retrieved_logs)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" or other models
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {question}\nContext: {context}\nAnswer:"}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        raise


def main():
    print("Entering main function...")
    access_logs = parse_log_file(access_log_path, 'access')
    error_logs = parse_log_file(error_log_path, 'error')

    # Create vectors with the same vectorizer
    access_vectors, vectorizer = create_vectors(access_logs)
    error_vectors, _ = create_vectors(error_logs, vectorizer)  # Use the same vectorizer

    # Check the dimensions of the vectors
    print(f"Access vectors shape: {access_vectors.shape}")
    print(f"Error vectors shape: {error_vectors.shape}")

    # Ensure the vectors have the same number of features
    if access_vectors.shape[1] != error_vectors.shape[1]:
        raise ValueError("Vectors have different number of features")

    # Initialize FAISS index
    dimension = access_vectors.shape[1]
    print(f"FAISS index dimension: {dimension}")
    index = faiss.IndexFlatL2(dimension)

    # Add vectors to index
    add_vectors_to_index(index, access_vectors)
    add_vectors_to_index(index, error_vectors)

    print("Vectors successfully added to the index.")

    # Ask a question
    question = "Hangi sayfalar 404 hatası döndürdü?"
    try:
        answer = ask_question(question, index, vectorizer, access_logs + error_logs)
        print(answer)
    except Exception as e:
        print(f"Error asking question: {e}")

if __name__ == "__main__":
    main()
