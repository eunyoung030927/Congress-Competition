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

def get_openai_response(prompt, api_key):
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",  # gpt-3.5-turbo # gpt-4-1106-preview
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

def get_legislator_persona(user_problem, similar_laws, api_key):
    prompt = f"""
    문제 해결 과정:
    1. 사용자가 제출한 문제를 분석하여, 이 문제의 핵심 요소와 관련된 법적 및 사회적 맥락을 식별합니다.
    2. 이 문제에 대응하기 위해 필요한 입법적 조치의 유형을 고려합니다. 이는 새로운 법률의 제정, 기존 법률의 수정, 또는 특정 정책의 도입을 포함할 수 있습니다.
    3. 관련 부처와 이해관계자들이 이 문제에 대해 어떤 입장을 가지고 있는지 조사합니다. 이는 입법 과정에서의 협력이나 반대를 예상하고 대응 전략을 수립하는 데 도움이 됩니다.
    4. 비슷한 문제를 해결하기 위한 다른 국가들의 접근 방식을 조사하여, 그들의 경험으로부터 배울 수 있는 점을 찾습니다.
    5. 이 모든 정보를 바탕으로, {user_problem}을 해결하기 위한 법안의 초안을 작성합니다. 이 때 법안은 문제의 원인을 명확히 하고, 효과적인 해결책을 제시하며, 예상되는 결과와 그 실행 계획을 포함해야 합니다.

    사용자 문제: {user_problem}
    기반으로 할 작업:
    - 문제의 핵심 요소와 관련된 법적 및 사회적 맥락 식별
    - 필요한 입법적 조치의 유형 고려
    - 관련 부처와 이해관계자들의 입장 조사
    - 다른 국가들의 접근 방식 조사
    - 법안 초안 작성

    이 정보를 바탕으로 {user_problem} 문제를 해결하기 위한 법안 초안을 작성해주세요. 이 법안은 문제의 원인을 명확히 하고, 효과적인 해결책을 제시하며, 예상되는 결과와 그 실행 계획을 포함해야 합니다.
    """
    persona = get_openai_response(prompt, api_key)
    return persona

def analyze_legislation(user_problem, generated_context, similar_laws_info, api_key):
    """
    생성된 법안과 유사한 법률들을 바탕으로 법안의 효용성, 실현 가능성, 그리고 비슷한 법안의 존재 여부를 평가합니다.
    """
    prompt = f"""
    한국어 사고 과정:
    사용자 문제: "{user_problem}"에 대해 제안된 법률안이 있습니다. 이 법률안은 다음과 같습니다: "{generated_context}". 또한, 이와 유사한 법률들에 대한 정보가 여기 있습니다: {similar_laws_info}. 
    
    첫 번째 단계로, 제안된 법률안이 해결하고자 하는 주요 문제와 그 해결책을 분석해 봅시다. 그 다음, 유사한 법률들과 비교하여 이 법률안의 차별점과 새로운 접근 방식이 무엇인지 살펴보겠습니다. 마지막으로, 이 법률안의 실현 가능성과 사회적, 법적 효용성을 평가하여, 이 법률안이 실제로 효과적인 해결책이 될 수 있을지, 그리고 이미 존재하는 비슷한 법안들과 어떻게 다른지 결론을 내리겠습니다.
    
    이 법률안의 효용성과 실현 가능성에 대해 어떻게 생각하십니까? 그리고 이 법률안이 새롭게 제시하는 해결책은 무엇이며, 이미 존재하는 비슷한 법률과는 어떻게 다릅니까?
    """
    analysis = get_openai_response(prompt, api_key)
    return analysis


def main():
    st.title("AI-based Legislation Drafting and Similarity Checker")
    api_key = st.text_input("Enter your OpenAI GPT API key:", type="password")

    user_problem = st.text_area("Describe your problem or situation for the legislation:")
    
    if 'generated_context' not in st.session_state:
        if st.button('Generate Legislation Context', key='generate_context'):
            if user_problem:
                prompt = f"""첫째, 사용자가 제공한 문제 '{user_problem}'에 대해 생각해봅시다. 이 문제가 사회에 미치는 영향은 무엇일까요? 문제의 근본 원인은 무엇이고, 이를 해결하기 위해 어떤 접근 방식이 필요할까요? 둘째, 문제를 해결하기 위한 법안을 구상할 때, 법안의 제목, 목적, 내용을 어떻게 설정할 수 있을지 고민해봅시다. 법안의 목적은 문제의 해결에 초점을 맞추고, 내용은 구체적인 실행 계획과 방법을 포함해야 합니다. 셋째, 발의 이유 및 원인을 명확히 하고, 해결 방안과 앞으로의 전략을 제시해야 합니다. 이 과정에서 우리는 문제에 대한 깊은 이해와 함께 법안이 가져올 긍정적인 변화를 예상할 수 있어야 합니다.

                너는 이 과정을 거쳐 '{user_problem}' 문제를 해결할 수 있는 국회 법안을 발의하는 AI야. 보고서 형식으로 결과를 출력하되, 모든 단계를 거치며 논리적으로 사고해야 합니다. A4 한 장 분량으로 법안을 구성하며, 다음 내용을 포함해야 합니다:

                - 보고서 제목
                - 보고서 작성일자
                - 관련 부처
                - 제안 법안 명칭
                - 법안 목적
                - 법안 내용 (구체적인 조항 및 실행 계획)
                - 발의의 이유 및 원인
                - 해결 방안
                - 앞으로의 전략

                이 프로세스를 통해, '{user_problem}'에 대한 효과적이고 실현 가능한 법안을 만들어봅시다.
                """

                context = get_openai_response(prompt, api_key)
                st.session_state['generated_context'] = context
                st.markdown("#### Generated context for legislation:")
                st.markdown(context, unsafe_allow_html=True)
            else:
                st.warning("Please enter a problem description first.")
    else:
        st.markdown("#### Generated context for legislation:")
        st.markdown(st.session_state['generated_context'], unsafe_allow_html=True)

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
            similar_laws_info.append(f"{law_info['제목']} - {law_info['내용']}")

        st.session_state['similar_laws_info'] = similar_laws_info
        st.session_state['similar_laws_display'] = []

        for index in top_indices:
            law_info = df.iloc[index][['제목', '내용']]
            law_info['제목'] = law_info['제목'].replace('\\n', '\n')
            law_info['내용'] = law_info['내용'].replace('\\n', '\n')
            st.session_state['similar_laws_display'].append(law_info)

    if 'similar_laws_display' in st.session_state:
        for law_info in st.session_state['similar_laws_display']:
            st.write(law_info)
            st.write("------")

    if 'similar_laws_info' in st.session_state:
        if st.button('Create Legislator Persona', key='create_persona'):
            legislator_persona = get_legislator_persona(user_problem, '\n'.join(st.session_state['similar_laws_info']), api_key)
            st.subheader("국회의원 페르소나")
            st.markdown(legislator_persona, unsafe_allow_html=True)

    if 'generated_context' in st.session_state and 'similar_laws_info' in st.session_state:
        if st.button('법안 분석', key='analyze_legislation'):
            analysis_result = analyze_legislation(
                user_problem=user_problem,
                generated_context=st.session_state['generated_context'],
                similar_laws_info='\n'.join(st.session_state['similar_laws_info']),
                api_key=api_key
            )
            st.subheader("법안 분석 결과")
            st.markdown(analysis_result, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
