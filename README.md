# K Congress Competition 2023
- notion [link](https://emphasized-leech-c56.notion.site/K-Congress-Data-Competition-3762258f913047dd9bfb563a6c3b4cd3?pvs=4)

# Deployment Using Streamlit
- KooK AI [link](https://kookai.streamlit.app/)

> ![스크린샷 2023-12-05 오후 4 24 24](https://github.com/sparkerhoney/Congress-Competition/assets/108461006/6ba7d0c4-35a4-4f69-8593-8299902a39e7)

## Contents Progress
```bash
Congress-Competition/
├── DB
│   ├── README.md
│   ├── Read_OpenAPI.ipynb
│   ├── XML_parser.ipynb
│   └── law_list
│       └── law_list.csv
├── LICENSE
├── Prompts
│   └── README.md
├── README.md
├── model
│   ├── Embedding.py
│   ├── README.md
│   ├── SBERT.py
│   ├── SBERT_cosine_similarity.py
│   └── cosine_similarity.ipynb
├── requirements.txt
├── run.sh
├── streamlit-agent
└── web
    ├── README.md
    ├── __pycache__
    │   ├── basic_streaming.cpython-311.pyc
    │   ├── chat_with_documents.cpython-311.pyc
    │   └── top5_recommendation.cpython-311.pyc
    ├── app.py
    ├── basic_streaming.py
    ├── chat_with_documents.py
    ├── embeddings_data.csv
    └── top5_recommendation.py
```

# How to use(in local)
## Web Implementation
```bash
$ chmod +x run.sh
$ sh run.sh
```

### MLOPS & Entire Work Flow
![스크린샷 2023-11-22 오후 4 42 03](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/570f60bf-df72-4c19-9411-1e6b0ff471a6)

---

## Key System for our Project
### 1. Reference Material Recommendation System
![image](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/48a4afbd-eaee-4729-b602-3c8d7c7bf5a8)

### 2. Personalized Bill Drafting System
![image](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/031e91fe-00a3-4bf3-8d61-1bcce8beef89)

### 3. Public Petition Standardization System
![스크린샷 2023-11-22 오후 3 22 22](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/84381997-29b3-451b-a2fd-8d810527d13f)

### 4.Legislator Matching System
![image](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/9450bf12-2c81-4708-93eb-4fe321465fc5)

---

#### Introduction
이 보고서는 국회의 텍스트 데이터를 활용하여 사용자에게 맞춤형 정보를 제공하는 웹사이트 개발에 관한 프로젝트의 세부 계획과 절차를 자세히 설명합니다.<br>
본 프로젝트는 Python 언어를 기반으로 하여, 효율적인 데이터 처리와 사용자 친화적인 경험을 제공하기 위해 FastAPI 및 Streamlit을 사용합니다.<br>
FastAPI를 통해 강력하고 민첩한 백엔드 시스템을 구축하고, Streamlit을 사용하여 직관적이고 상호작용이 용이한 프론트엔드 인터페이스를 개발할 계획입니다.<br>
이 프로젝트의 목표는 국회 데이터를 최대한 활용하여 사용자에게 실질적인 가치를 제공하고, 정보 접근성을 향상시키는 것입니다. 이를 통해 우리는 정치적 투명성과 시민 참여를 증진시키는 데 기여하고자 합니다.<br>

<br>
<br>
---

> `Project Owner : Honey`
