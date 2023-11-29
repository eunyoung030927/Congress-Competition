# import streamlit as st
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np

# # 데이터 로드
# @st.cache
# def load_data():
#     return pd.read_csv('your_local_data.csv')

# # 코사인 유사도 계산
# def calculate_similarity(data, user_input):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(data.append(user_input))
#     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#     return cosine_sim

# # 유사도가 높은 상위 5개 데이터 추출
# def get_top_matches(data, cosine_sim, top_n=5):
#     indices = np.argsort(cosine_sim[0])[::-1][:top_n]
#     return data.iloc[indices]

# # 서브페이지 메인 함수
# def main():
#     st.title("Data Similarity Checker")

#     # 사용자 입력
#     user_input = st.text_area("Enter your text here:")
#     if user_input:
#         data = load_data()
#         cosine_sim = calculate_similarity(data['text_column_name'], pd.Series([user_input]))
#         top_matches = get_top_matches(data, cosine_sim)

#         st.write("Top 5 similar entries from the database:")
#         st.table(top_matches)

# # 스트림릿 앱 실행
# if __name__ == '__main__':
#     main()

# # similarity_checker.py
# import streamlit as st
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# import numpy as np

# # NumPy를 사용한 코사인 유사도 계산 함수
# def calculate_similarity(embeddings, input_embedding):
#     norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#     norm_input_embedding = input_embedding / np.linalg.norm(input_embedding)
#     cosine_similarities = np.dot(norm_embeddings, norm_input_embedding)
#     return cosine_similarities

# # 문장 임베딩 함수 정의
# def get_sentence_embedding(sentence, tokenizer, model):
#     encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     return mean_pooling(model_output, encoded_input['attention_mask'])

# # Mean pooling 함수 정의 (NumPy로 변경)
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0].detach().numpy()
#     input_mask_expanded = np.expand_dims(attention_mask.numpy(), -1)
#     sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
#     sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
#     return sum_embeddings / sum_mask


# # 메인 함수
# def main():
#     st.title("Data Similarity Checker")

#     # 토크나이저와 모델 로드
#     tokenizer = AutoTokenizer.from_pretrained('/Users/marketdesigners/Documents/GitHub/Congress_Competition/streamlit-agent/all-MiniLM-L6-v2')
#     model = AutoModel.from_pretrained('/Users/marketdesigners/Documents/GitHub/Congress_Competition/streamlit-agent/all-MiniLM-L6-v2')

#     # 데이터베이스 로드 (CSV 파일에서 'embedding' 컬럼 로드)
#     df = pd.read_csv('/Users/marketdesigners/Documents/GitHub/Congress_Competition/DB/law_list/embeddings_data.csv')
#     db_embeddings = np.stack(df['Embedding'].apply(lambda x: np.array(eval(x))))

#     # 사용자 입력
#     user_input = st.text_area("Enter your text here:")
#     if user_input:
#         # 사용자 입력에 대한 임베딩 계산
#         input_embedding = get_sentence_embedding(user_input, tokenizer, model)

#         # 코사인 유사도 계산
#         cosine_sim = calculate_similarity(db_embeddings, input_embedding)

#         # 상위 5개 유사도 인덱스 추출
#         top_indices = cosine_sim.topk(5).indices

#         # 결과 출력
#         for index in top_indices:
#             st.write(f"문장 {index + 1}: {sentences1.iloc[index]}")

# if __name__ == '__main__':
#     main()

import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def calculate_similarity(embeddings, input_embedding):
    # embeddings를 토치 텐서로 변환
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    cosine_similarities = F.cosine_similarity(embeddings_tensor, input_embedding)
    return cosine_similarities

# 문장 임베딩 함수 정의
def get_sentence_embedding(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

# Mean pooling 함수 정의
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def main():
    st.title("Data Similarity Checker")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Load the database (Here you should load your CSV file accordingly)
    df = pd.read_csv('/Users/marketdesigners/Documents/GitHub/Congress_Competition/DB/law_list/embeddings_data.csv')
    db_embeddings = torch.stack(tuple(df['Embedding'].apply(lambda emb: torch.tensor(eval(emb), dtype=torch.float))))

    # User input
    user_input = st.text_area("Enter your text here:")
    if user_input:
        # Calculate embedding for user input
        input_embedding = get_sentence_embedding(user_input, tokenizer, model).unsqueeze(0)  # Add batch dimension

        # Calculate cosine similarity
        cosine_sim = calculate_similarity(db_embeddings, input_embedding)

        # Get top 5 similar indices
        top_indices = cosine_sim.topk(5).indices.numpy()  # Convert to numpy array to use as index

        # Display results
        for index in top_indices:
            # title = df.iloc[index]['제목']
            title = df.iloc[index]['제목'].replace('\\n', '\n')  # Replace '\\n' with actual newlines
            content = df.iloc[index]['내용'].replace('\\n', '\n')  # Replace '\\n' with actual newlines
            # st.write(f"### {title}")
            st.write(title + content)
            st.write("------")
            # st.write(content)
            # st.write("------")

if __name__ == '__main__':
    main()