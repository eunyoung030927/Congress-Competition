import openai
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def calculate_similarity(embeddings, input_embedding):
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    cosine_similarities = F.cosine_similarity(embeddings_tensor, input_embedding)
    return cosine_similarities

def get_sentence_embedding(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_openai_response(prompt):
    openai.api_key = 'API-KEY'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1500
    )
    return response.choices[0].text.strip()

def main():
    st.title("AI-based Legislation Drafting and Similarity Checker")

    user_problem = st.text_area("Describe your problem or situation for the legislation:")
    
    if 'generated_context' not in st.session_state:
        if st.button('Generate Legislation Context'):
            if user_problem:
                prompt = f"너는 한국어로 input값을 받아서 국회 법안을 발의 하는 ai야. 100자 이내로 {user_problem}을 국회에 입법하게 할 법안으로 만들어줘."
                context = get_openai_response(prompt)
                
                st.session_state['generated_context'] = context
                
                st.text_area("Generated context for legislation:", context, height=150)
            else:
                st.warning("Please enter a problem description first.")
    else:
        st.text_area("Generated context for legislation:", st.session_state['generated_context'], height=150)

    if 'generated_context' in st.session_state and st.button('Find Similar Laws'):
        
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        df = pd.read_csv('embeddings_data.csv')
        db_embeddings = torch.stack(tuple(df['Embedding'].apply(lambda emb: torch.tensor(eval(emb), dtype=torch.float))))

        context_embedding = get_sentence_embedding(st.session_state['generated_context'], tokenizer, model).unsqueeze(0)

        cosine_sim = calculate_similarity(db_embeddings, context_embedding)

        top_indices = cosine_sim.topk(5).indices.numpy()

        for index in top_indices:
            # title = df.iloc[index]['제목']
            title = df.iloc[index][['제목','내용']].replace('\\n', '\n')  
            # content = df.iloc[index]['내용'].replace('\\n', '\n')  
            # st.write(f"### {title}")
            st.write(title)
            st.write("------")
            # st.write(content)
            # st.write("------")

if __name__ == '__main__':
    main()
