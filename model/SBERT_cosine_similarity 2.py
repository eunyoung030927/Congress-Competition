from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd

# 파일을 읽어오는 부분
sentences1 = pd.read_csv("/Users/lhe339/Documents/GitHub/Congress_Competition/DB/law_list/law_list.csv")
sentences1 = sentences1["내용"]
sentences2 = """나라에 성폭력 피해자들의 보호를 위한 법률안으로

'성폭력 보호 및 지원 방안 제정법'을 마련하고, 이를 국회가 채택하고 국민에게 새로운 보호장치를 제공하고자 합니다.

이 법률에 의해 보호자가 배정되고 당해 사건에 민원 및 피해자를 보호하기 위해 법률 상 여러 조치도 가능하도록 하여야 합니다. 또한 경찰에 대한 강력한 책임을 맡게 하고, 경찰이 성폭력 피해자들과 상대방들을 적극적으로 보호하고 도움을 줄 수 있도록 규정하는 것 등이 포함되어야 합니다."""

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# sentences1의 각 문장에 대해 토큰화 및 임베딩
encoded_input1 = tokenizer(list(sentences1), padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output1 = model(**encoded_input1)

# sentences2에 대해 토큰화 및 임베딩
encoded_input2 = tokenizer(sentences2, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output2 = model(**encoded_input2)

# Mean pooling 함수
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 임베딩 계산
sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])

# 정규화
sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)
sentence_embeddings2 = F.normalize(sentence_embeddings2, p=2, dim=1).unsqueeze(0)

# 각 문장에 대한 코사인 유사도 계산
cosine_similarities = F.cosine_similarity(sentence_embeddings1, sentence_embeddings2, dim=1)

# 상위 5개 문장 선택
top5_indices = cosine_similarities.topk(5).indices

# 결과 출력
for index in top5_indices:
    print(f"문장 {index + 1}: {sentences1.iloc[index]}")