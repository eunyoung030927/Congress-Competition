import os
import openai 
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()  

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

def get_legislator_persona(user_problem, similar_laws):
    prompt = f"사용자가 제출한 문제와 식별된 상위 5개의 유사한 법률을 기반으로 이 문제를 챔피언할 국회의원의 페르소나를 만듭니다. 그들의 정책 방향, 공약 및 유권자에 대한 어필 전략을 포함하세요. 페르소나가 시민의 필요와 제공된 입법적 맥락과 조화를 이루도록 합니다.\n\n사용자 문제: {user_problem}\n식별된 유사한 법률: {similar_laws}\n\n입법 환경에서 이러한 문제를 효과적으로 옹호할 수 있는 필요한 특성을 가진 페르소나를 생성하세요. 이름은 생성하면 안됨."
    persona = get_openai_response(prompt)
    return persona

def get_openai_response(prompt):
    openai.api_key = os.getenv('OPENAI_API_KEY') 
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()


# def main():
#     st.title("AI-based Legislation Drafting and Similarity Checker")

#     user_problem = st.text_area("Describe your problem or situation for the legislation:")
    
#     if 'generated_context' not in st.session_state:
#         if st.button('Generate Legislation Context', key='generate_context'):
#             if user_problem:
#                 prompt = f"너는 한국어로 {user_problem}값을 받아서 국회 법안을 발의 하는 ai야. 너가 뭘했다라고 먼저 얘기하지 말고 보고서 형식으로만 출력해. A4 한장 분량으로 {user_problem}을 국회에 입법하게 할 법안으로 만들어줘. 들어가야하는 내용은 법안의 제목, 목적, 내용, 발의의 이유 및 원인, 해결방안, 앞으로의 전략 등이야."
#                 context = get_openai_response(prompt)
                
#                 st.session_state['generated_context'] = context
                
#                 st.text_area("Generated context for legislation:", context, height=150)
#             else:
#                 st.warning("Please enter a problem description first.")
#     else:
#         st.text_area("Generated context for legislation:", st.session_state['generated_context'], height=150)

#     if 'generated_context' in st.session_state and st.button('Find Similar Laws'):
        
#         tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
#         model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

#         df = pd.read_csv('web/embeddings_data.csv')
#         db_embeddings = torch.stack(tuple(df['Embedding'].apply(lambda emb: torch.tensor(eval(emb), dtype=torch.float))))

#         context_embedding = get_sentence_embedding(st.session_state['generated_context'], tokenizer, model).unsqueeze(0)

#         cosine_sim = calculate_similarity(db_embeddings, context_embedding)

#         top_indices = cosine_sim.topk(5).indices.numpy()

#     #     similar_laws_info = []
#     #     for index in top_indices:
#     #         law_info = df.iloc[index][['제목','내용']].to_dict()
#     #         similar_laws_info.append(f"{law_info['제목']} - {law_info['내용']}")
#     #         st.write(law_info)
#     #         st.write("------")

#     #     st.session_state['similar_laws_info'] = '\n'.join(similar_laws_info)

#     # if 'generated_context' in st.session_state and 'similar_laws_info' in st.session_state:
#     #     if st.button('Create Legislator Persona'):
#     #         legislator_persona = get_legislator_persona(user_problem, st.session_state['similar_laws_info'])
#     #         st.subheader("국회의원 페르소나")
#     #         st.text_area("페르소나 세부 사항:", legislator_persona, height=300)

#         # similar_laws_info = []
#         # for index in top_indices:
#         #     law_info = df.iloc[index][['제목', '내용']].to_dict()
#         #     similar_laws_display = f"{law_info['제목']} - {law_info['내용']}"
#         #     similar_laws_info.append(similar_laws_display)
#         #     st.text(similar_laws_display) 
#         #     st.write("------")

#         # st.session_state['similar_laws_info'] = similar_laws_info

#     #     similar_laws_info = []
#     #     for index in top_indices:
#     #         law_info = df.iloc[index]  # 각 법안의 데이터 추출

#     #         # 각 법안의 제목과 내용을 순차적으로 처리
#     #         for law_id, title in law_info['제목'].items():
#     #             content = law_info['내용'][law_id]

#     #             # 줄바꿈 처리
#     #             title = title.replace('\\n', '\n')
#     #             content = content.replace('\\n', '\n')

#     #             # Markdown 형식으로 표시
#     #             similar_laws_display = f"**제목:** \n{title}\n**내용:** \n{content}\n---\n"
#     #             similar_laws_info.append(similar_laws_display)
#     #             st.markdown(similar_laws_display, unsafe_allow_html=True)

#     #     st.session_state['similar_laws_info'] = similar_laws_info

#     # # Similar Laws 정보가 유지되도록 출력
#     # if 'similar_laws_info' in st.session_state:
#     #     st.subheader("Similar Laws")
#     #     for law in st.session_state['similar_laws_info']:
#     #         st.markdown(law, unsafe_allow_html=True)
#     # if 'generated_context' in st.session_state:
#     #     if st.button('Find Similar Laws', key='find_similar_laws'):
#     #         # DataFrame 초기화
#     #         similar_laws_df = pd.DataFrame(columns=['제목', '내용'])
            
#     #         for index in top_indices:
#     #             # DataFrame에서 직접 데이터 추출
#     #             law_data = df.iloc[index]
#     #             title = law_data['제목'].replace('\\n', '\n')
#     #             content = law_data['내용'].replace('\\n', '\n')

#     #             # DataFrame에 추가
#     #             similar_laws_df = similar_laws_df.append({'제목': title, '내용': content}, ignore_index=True)

#     #         # DataFrame을 session state에 저장
#     #         st.session_state['similar_laws_df'] = similar_laws_df

#     #     # session state에 저장된 DataFrame이 있으면, DataFrame 형식으로 출력
#     #     if 'similar_laws_df' in st.session_state:
#     #         st.dataframe(st.session_state['similar_laws_df'])

#     # if 'generated_context' in st.session_state:

#     #     if 'similar_laws_info' in st.session_state:
#     #         st.subheader("Similar Laws")
#     #         for law in st.session_state['similar_laws_info']:
#     #             st.text(law)
#     #             st.write("------")

#         for index in top_indices:
#             # title = df.iloc[index]['제목']
#             title = df.iloc[index][['제목','내용']].replace('\\n', '\n')  
#             # content = df.iloc[index]['내용'].replace('\\n', '\n')  
#             # st.write(f"### {title}")
#             st.write(title)
#             st.write("------")
#             # st.write(content)
#             # st.write("------")

#         if st.button('Create Legislator Persona', key='create_persona'):
#             legislator_persona = get_legislator_persona(user_problem, '\n'.join(st.session_state['similar_laws_info']))
#             st.subheader("국회의원 페르소나")
#             st.text_area("페르소나 세부 사항:", legislator_persona, height=300)


# if __name__ == '__main__':
#     main()
#         # for index in top_indices:
#         #     # title = df.iloc[index]['제목']
#         #     title = df.iloc[index][['제목','내용']].replace('\\n', '\n')  
#         #     # content = df.iloc[index]['내용'].replace('\\n', '\n')  
#         #     # st.write(f"### {title}")
#         #     st.write(title)
#         #     st.write("------")
#         #     # st.write(content)
#         #     # st.write("------")

# # if __name__ == '__main__':
# #     main()

def main():
    st.title("AI-based Legislation Drafting and Similarity Checker")

    user_problem = st.text_area("Describe your problem or situation for the legislation:")
    
    if 'generated_context' not in st.session_state:
        if st.button('Generate Legislation Context', key='generate_context'):
            if user_problem:
                prompt = f"너는 한국어로 {user_problem}값을 받아서 국회 법안을 발의 하는 ai야. 너가 뭘했다라고 먼저 얘기하지 말고 보고서 형식으로만 출력해. A4 한장 분량으로 {user_problem}을 국회에 입법하게 할 법안으로 만들어줘. 들어가야하는 내용은 법안의 제목, 목적, 내용, 발의의 이유 및 원인, 해결방안, 앞으로의 전략 등이야."
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

        df = pd.read_csv('web/embeddings_data.csv')
        db_embeddings = torch.stack(tuple(df['Embedding'].apply(lambda emb: torch.tensor(eval(emb), dtype=torch.float))))

        context_embedding = get_sentence_embedding(st.session_state['generated_context'], tokenizer, model).unsqueeze(0)

        cosine_sim = calculate_similarity(db_embeddings, context_embedding)

        top_indices = cosine_sim.topk(5).indices.numpy()

        similar_laws_info = []
        for index in top_indices:
            law_info = df.iloc[index][['제목', '내용']].to_dict()
            similar_laws_display = f"{law_info['제목']} - {law_info['내용']}"
            similar_laws_info.append(similar_laws_display)
            st.markdown(similar_laws_display, unsafe_allow_html=True)

        st.session_state['similar_laws_info'] = similar_laws_info

    if 'similar_laws_info' in st.session_state and st.button('Create Legislator Persona', key='create_persona'):
        legislator_persona = get_legislator_persona(user_problem, '\n'.join(st.session_state['similar_laws_info']))
        st.subheader("국회의원 페르소나")
        st.markdown(legislator_persona, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
