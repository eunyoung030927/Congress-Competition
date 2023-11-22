# KCongress_Competition_2023
- notion [link](https://emphasized-leech-c56.notion.site/K-Congress-Data-Competition-3762258f913047dd9bfb563a6c3b4cd3?pvs=4)
  
## MLOPS & Entire Work Flow
![image](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/df287947-1775-4d13-9e08-64baae131892)

## Key System for our Project
### 1. Reference Material Recommendation System
![image](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/48a4afbd-eaee-4729-b602-3c8d7c7bf5a8)

### 2. Personalized Bill Drafting System
![image](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/031e91fe-00a3-4bf3-8d61-1bcce8beef89)

### 3. Public Petition Standardization System
![image](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/f017b8c7-9e1e-4263-a00b-f69400069199)

### 4.Legislator Matching System
![image](https://github.com/sparkerhoney/Congress_Competition/assets/108461006/9450bf12-2c81-4708-93eb-4fe321465fc5)

#### Introduction
이 보고서는 국회의 텍스트 데이터를 활용하여 사용자에게 맞춤형 정보를 제공하는 웹사이트 개발에 관한 프로젝트의 세부 계획과 절차를 자세히 설명합니다.<br>
본 프로젝트는 Python 언어를 기반으로 하여, 효율적인 데이터 처리와 사용자 친화적인 경험을 제공하기 위해 FastAPI 및 Streamlit을 사용합니다.<br>
FastAPI를 통해 강력하고 민첩한 백엔드 시스템을 구축하고, Streamlit을 사용하여 직관적이고 상호작용이 용이한 프론트엔드 인터페이스를 개발할 계획입니다.<br>
이 프로젝트의 목표는 국회 데이터를 최대한 활용하여 사용자에게 실질적인 가치를 제공하고, 정보 접근성을 향상시키는 것입니다. 이를 통해 우리는 정치적 투명성과 시민 참여를 증진시키는 데 기여하고자 합니다.<br>

#### 1. 데이터 수집 및 전처리
- 수집 데이터: 본 프로젝트에서는 국회의 회의록, 법안 텍스트, 의원 발언 기록과 같은 다양한 형태의 텍스트 데이터를 수집합니다.<br> 이러한 데이터는 국회 웹사이트, 공식 문서, 그리고 관련 정부 데이터베이스에서 접근 가능합니다.<br> 수집하는 데이터는 법률안의 내용, 토론 내용, 의결 과정, 의원들의 다양한 의견 등을 포함하여 광범위합니다.<br> 이 데이터는 국회의 의사결정 과정과 법률 제정의 투명성을 높이는 데 중요한 역할을 합니다.<br>
- 전처리 방법: 데이터 전처리 과정에서는 먼저 오류 수정 및 불필요한 정보 제거를 통해 데이터의 정제를 진행합니다.<br> 이후, 형태소 분석을 통해 한국어 텍스트의 문법적 특성을 분석하고, 토큰화 과정에서는 텍스트를 의미 있는 작은 단위로 분리합니다.<br> 데이터의 정규화를 통해 일관된 형식을 유지하며, 이 과정은 데이터의 품질을 보장하고 후속 분석의 정확성을 향상시킵니다.<br> 또한, 이러한 전처리는 자연어 처리 모델의 학습 및 추론 효율성을 증대시키는 데 중요한 역할을 합니다.<br>

#### 2. LanChain을 이용한 데이터 처리
- **LanChain 사용 목적**: LanChain을 사용하는 주된 목적은 데이터의 무결성을 보장하고 안전한 데이터 관리를 구현하는 것입니다.<br> 이는 특히 민감하고 중요한 국회 데이터를 다루는데 있어 필수적인 요소입니다.<br> LanChain 기술은 데이터의 변경이나 조작 없이 원본 상태를 유지하는 데 중요한 역할을 하며, 이는 데이터 신뢰성을 크게 향상시킵니다.<br> 또한, LanChain은 투명한 데이터 처리 과정을 제공함으로써 사용자와 개발자 모두에게 데이터의 출처와 처리 과정에 대한 신뢰를 제공합니다.<br>

- **구현 방법**: LanChain의 블록체인 기술을 활용하여, 데이터의 저장, 검증 및 접근을 철저하게 관리합니다.<br> 각 데이터 블록은 이전 블록에 대한 참조와 함께 체인에 연결되어, 데이터의 순서와 무결성이 유지됩니다.<br> 이러한 접근 방식은 데이터의 변경이나 조작을 거의 불가능하게 만들어, 보안성을 크게 향상시킵니다.<br> 또한, LanChain 네트워크의 모든 참가자는 데이터의 변화를 확인할 수 있으며, 이는 데이터 처리 과정의 투명성을 보장합니다.<br> 이러한 방식은 데이터의 신뢰도를 높이고, 사용자에게 정확하고 신뢰할 수 있는 정보를 제공하는 데 중요한 역할을 합니다.<br>

#### 3. 데이터베이스 선택 및 사용
- **데이터베이스 선택**: 본 프로젝트에서는 Supabase를 사용하기로 결정했습니다.<br> Supabase는 PostgreSQL을 기반으로 하는 관계형 데이터베이스 서비스로, 데이터의 구조적 관리와 복잡한 쿼리 처리에 강점을 가지고 있습니다.<br> 이는 데이터의 관계를 명확하게 정의하고 관리할 필요가 있는 본 프로젝트에 적합합니다.<br> 또한, Supabase는 관계형 데이터베이스의 강력한 기능과 함께 확장성과 안정성을 제공하여, 프로젝트의 규모 확장에 따른 데이터 관리 요구를 충족시킬 수 있습니다.<br>

- **사용 이유**: Supabase는 실시간 데이터베이스 기능을 제공하여, 데이터 변경 사항을 즉각적으로 반영하고 사용자에게 실시간으로 정보를 제공할 수 있습니다.<br> 이는 국회 데이터와 같이 시시각각 변화하는 정보를 다루는데 매우 중요합니다.<br> 또한, 사용자 인증 및 간편한 API 제공 기능을 통해 개발자가 보다 쉽고 빠르게 애플리케이션을 구축할 수 있습니다.<br> 이러한 기능은 대규모 데이터를 효율적으로 처리하고, 사용자에게 안전하고 편리한 접근성을 제공하는 데 필수적입니다.<br>

#### MongoDB (NoSQL 데이터베이스) 사용 이유 및 장단점
- **사용 이유**: MongoDB는 NoSQL 데이터베이스로, 높은 확장성과 유연한 데이터 모델을 제공합니다.<br> 이는 구조가 불규칙하거나 변경 가능성이 높은 데이터를 다룰 때 유용합니다.<br> MongoDB는 빠른 읽기/쓰기 속도를 제공하며, 대용량의 데이터를 효율적으로 처리할 수 있습니다.<br>

- **장점**: MongoDB의 가장 큰 장점은 스키마가 없는 유연한 데이터 모델입니다.<br> 이는 데이터 구조가 빈번하게 변경될 때 유용하며, 개발자가 데이터베이스 스키마에 구애받지 않고 빠르게 개발할 수 있게 합니다.<br> 또한, 수평적 확장성이 뛰어나 대규모 분산 데이터 처리에 적합합니다.<br>

- **단점**: 반면, MongoDB는 관계형 데이터베이스에 비해 복잡한 쿼리 처리 능력이 제한적입니다.<br> 데이터 간의 관계를 표현하기 위해서는 추가적인 노력이 필요하며, 이는 복잡한 관계를 가진 데이터를 다룰 때 불편함을 초래할 수 있습니다.<br> 또한, 데이터 무결성과 일관성 유지에 관한 관리가 추가적으로 필요합니다.<br>

#### 4. FastAPI 사용 및 백엔드 개발
- **FastAPI 소개**: FastAPI는 Python 기반의 고성능 웹 프레임워크로, 특히 비동기 처리 기능을 지원하여 높은 동시성 처리 능력을 제공합니다.<br> 이는 특히 대규모 사용자가 접근하는 웹 애플리케이션에 적합합니다.<br> 또한, FastAPI는 빠른 개발 주기와 간편한 유지보수를 가능하게 하여 개발자의 효율성을 높입니다. <br>FastAPI의 현대적인 설계와 풍부한 기능들은 백엔드 개발을 보다 쉽고 빠르게 할 수 있도록 돕습니다.<br>

- **백엔드 구현**: FastAPI를 활용하여 데이터 요청 처리, API 라우팅, 데이터베이스 연동 등의 핵심 백엔드 기능을 구현합니다.<br> 이 프레임워크는 데이터 유효성 검사, 보안, 비동기 처리 등 다양한 백엔드 요구 사항을 효과적으로 처리할 수 있도록 지원합니다.<br> 또한, FastAPI의 자동 문서화 기능은 개발자와 사용자 모두에게 API의 구조와 사용 방법을 명확하게 제공하여, API의 이해도와 접근성을 향상시킵니다.<br> 이러한 기능은 프로젝트의 API 관리 및 확장에 있어 큰 이점을 제공합니다.<br>

#### 5. 자연어 처리 모델 선택
- **모델 선택**: 본 프로젝트에서는 BERT, RoBERTa, GPT-4 중에서 최적의 모델을 선택합니다.<br> 선택 과정에서는 각 모델의 특성, 특히 임베딩의 정확도와 처리 속도를 중점적으로 고려합니다.<br> 프로젝트의 목적과 데이터의 특성에 가장 적합한 모델을 선택함으로써, 효율적이고 정확한 자연어 처리 결과를 얻을 수 있습니다.<br> 모델 선택은 데이터 처리의 효율성, 결과의 정확성, 그리고 시스템의 반응 속도에 큰 영향을 미칩니다.<br>

- **모델 비교**:
  - **BERT**: BERT는 다양한 자연어 처리 작업에 널리 사용되는 모델로, 문맥을 기반으로 한 텍스트의 의미 파악에 매우 효과적입니다.<br> 이 모델은 특히 국회 데이터와 같이 복잡하고 다양한 맥락을 가진 텍스트를 처리하는 데 강점을 가지고 있습니다.<br> BERT는 높은 범용성을 가지며, 다양한 언어 처리 작업에서 높은 정확도를 제공합니다.<br>
  
  - **RoBERTa**: RoBERTa는 BERT를 기반으로 하여 더 큰 데이터셋과 더 긴 학습 시간을 통해 성능을 개선한 모델입니다.<br> 이 모델은 특히 더 정밀한 텍스트 분석과 더 깊은 의미 이해에 강점을 가지고 있으며, 복잡한 텍스트 데이터에 더 정확한 결과를 제공합니다.<br>
  
  - **GPT-4**: GPT-4는 OpenAI의 최신 대규모 언어 모델로, 매우 높은 이해도와 창의적인 텍스트 생성 능력을 갖추고 있습니다.<br> 이 모델은 특히 사용자 질의에 대한 응답 생성, 텍스트 요약, 콘텐츠 생성 등의 작업에서 매우 효과적입니다.<br> GPT-4의 API 사용은 인터넷을 통해 쉽게 접근 가능하며, 복잡한 인프라 구축 없이도 강력한 자연어 처리 기능을 활용할 수 있습니다.<br>

#### 6. 코사인 유사도 계산의 중요성
- **목적**: 코사인 유사도 계산은 임베딩된 텍스트 간의 유사도를 측정하여, 텍스트 데이터 내에서 의미적으로 유사한 항목을 식별하는 데 사용됩니다.<br> 이 기법은 벡터 공간 내에서 두 텍스트 간의 각도를 측정함으로써, 그 유사도를 정량적으로 평가합니다.<br> 이는 특히 대규모 텍스트 데이터에서 특정 주제나 내용과 관련된 문서를 신속하게 찾아내는 데 매우 유용합니다.<br> 코사인 유사도는 자연어 처리에서 중요한 도구로, 텍스트 분류, 검색 엔진 최적화, 추천 시스템 등 다양한 응용 분야에서 활용됩니다.<br>

- **응용**: 코사인 유사도는 사용자가 입력한 검색 쿼리와 유사한 문서를 식별하고 추천하는 데 활용됩니다.<br> 예를 들어, 사용자가 국회의 특정 법안에 대해 검색할 때, 이 기법을 사용하여 관련 법안, 의원 발언, 토론 내용 등을 자동으로 추천할 수 있습니다.<br> 또한, 의회 데이터베이스 내에서 특정 주제나 키워드에 대해 서로 연관성이 높은 문서들을 연결하여, 사용자에게 보다 깊이 있는 정보를 제공하는 데 기여합니다.<br> 이러한 방식은 사용자의 검색 경험을 개선하고, 필요한 정보에 빠르게 접근할 수 있도록 도와줍니다.<br> 코사인 유사도를 활용한 이러한 기능은 사용자가 보다 효과적으로 데이터를 탐색하고, 관련 정보를 발견할 수 있게 만듭니다.<br>

#### 7. Streamlit을 이용한 프론트엔드 개발
- **Streamlit 소개**: Streamlit은 Python 개발자들이 쉽게 데이터 시각화와 인터랙티브한 웹 애플리케이션을 구축할 수 있도록 설계된 라이브러리입니다.<br> 이 도구는 복잡한 프론트엔드 코드 없이도 사용자 친화적인 웹 인터페이스를 신속하게 개발할 수 있게 해줍니다.<br> Streamlit을 사용함으로써 개발자는 Python 코드만으로 데이터 시각화, 컨트롤 위젯, 인터랙티브 기능 등을 쉽게 구현할 수 있습니다.<br> 이는 특히 데이터 중심의 웹 애플리케이션에서 강력한 시각화를 제공하는 데 유용합니다.<br>
- **구현 방법**: Streamlit을 사용하여 사용자 친화적인 인터페이스를 구축합니다.<br> 이는 데이터를 시각화하는 도구와 효율적인 사용자 상호작용을 위한 인터페이스 요소를 포함합니다.<br> 프로젝트에서는 국회 데이터를 분석한 결과를 시각적으로 표현하고, 사용자가 직관적으로 정보를 탐색하고 이해할 수 있도록 돕습니다. <br>Streamlit의 간편한 인터페이스 구축 기능은 사용자가 필요한 정보에 쉽게 접근하고, 데이터에 기반한 통찰을 얻을 수 있도록 만듭니다.<br>


#### 8. 사용자 경험 최적화 방안
- **방법**: 사용자 경험을 최적화하기 위해 첫 단계로 사용자 피드백을 적극적으로 수집합니다.<br> 이는 설문조사, 사용자 행동 분석, 직접적인 사용자 피드백 등 다양한 방법을 통해 이루어집니다.<br> 수집된 피드백은 웹사이트 인터페이스의 개선점을 파악하는 데 중요한 역할을 합니다.<br> 또한, 사용자 인터페이스(UI) 테스트를 주기적으로 실시하여 사용성 문제를 식별하고 개선합니다.<br> 성능 최적화는 웹사이트의 로딩 시간, 반응 속도 등을 개선하여 사용자 경험을 향상시키는 데 중요합니다.<br> 이러한 접근 방식은 웹사이트의 사용 편의성과 만족도를 높이는 데 기여합니다.<br>


#### 9. 추천 시스템 구현
- **모델 선택**: 추천 시스템 구현에는 Collaborative Filtering과 Content-Based Filtering이 고려됩니다.<br> Collaborative Filtering은 사용자의 과거 행동과 선호도를 기반으로 추천을 제공하는 반면, Content-Based Filtering은 항목의 내용과 사용자의 선호도를 분석하여 추천합니다.<br> 각 방법의 특성을 고려하여 사용자에게 가장 적합한 추천을 제공하는 모델을 선택합니다.<br>
- **적용 분야**: 추천 시스템은 사용자가 관심을 가질 만한 법안, 회의록, 의원 발언 등을 식별하고 추천하는 데 사용됩니다.<br> 이 시스템은 사용자의 검색 행동과 상호작용을 분석하여, 관련성 높은 콘텐츠를 자동으로 제안합니다.<br> 이를 통해 사용자는 자신의 관심사와 밀접하게 연관된 정보를 보다 쉽게 찾을 수 있습니다.<br> 추천 시스템은 사용자가 웹사이트에서 보다 효과적이고 효율적으로 정보를 탐색하도록 돕는 중요한 역할을 합니다.<br>


#### 결론
본 프로젝트는 국회 데이터를 활용하여 사용자에게 가치 있는 정보를 제공하는 것을 목표로 합니다.<br> 이를 위해 고성능 백엔드, 직관적인 프론트엔드 및 지속적인 시스템 개선을 통해 사용자 경험을 극대화할 계획입니다.<br> 사용자의 요구와 피드백을 적극적으로 수용하고, 최신 기술을 활용하여 효과적인 웹사이트를 구축할 것입니다.<br> 이러한 접근 방식은 국회 데이터를 통한 정치적 투명성 향상 및 시민 참여 증진에 기여할 것입니다.<br>


