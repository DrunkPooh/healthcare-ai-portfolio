# 신정호 · Healthcare AI

데이터 분석에서 시작해 ML 해커톤을 거쳐, 운영을 염두에 둔 LLM 백엔드까지 만들어 봤습니다.
헬스케어·의료 도메인에서 다루는 범위를 한 단계씩 넓혀왔고, 분석의 엄밀함과 백엔드 쪽 실행을 둘 다 해본 게 강점입니다.

[![GitHub](https://img.shields.io/badge/GitHub-DrunkPooh-181717?logo=github&logoColor=white)](https://github.com/DrunkPooh)
[![Email](https://img.shields.io/badge/Email-top03010@naver.com-0F5050)](mailto:top03010@naver.com)

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL_16-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2C5F2D)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

---

## 한눈에 보기

| 시점 | 프로젝트 · 유형 | 핵심 결과 | 역할 |
|---|---|---|---|
| 2026.05~06 | **MediPT** · 풀스택 LLM | 동시성 99× · 캐싱으로 외부 API 장애 격리 · PR 139 | 팀장 (BE-B) |
| 2026.04 | **난임 임신 성공 예측** · 분류 | Public LB 0.74218 (1위와 0.0003차) | 팀장 |
| 2026.02 | **칼로리 소모 예측** · 회귀 | RMSE 0.857 → 0.204 (−76%) | 팀원 |
| 2026.01 | **흡연 여부 분석** · EDA·통계 | 가설 3건 통계 검정 전부 p&lt;0.001 | 팀원 (데이터 분석) |

전 프로젝트가 헬스케어·의료 데이터를 다루고, 시간순으로 난이도가 올라갑니다.

---

## 1. MediPT — 복약 안내·생활습관 가이드 자동생성
`2026.05~06` · 팀 Uponati 4인 · **Backend B 팀장**

처방전을 사진으로 올리면 약품을 자동으로 읽고, 나이·체중·알레르기·질환 같은 프로필을 합쳐 복약 안내와 생활습관 가이드를 만들어주는 서비스. 저는 백엔드 B를 맡아 **약품 연동·안전 응답·챗봇·성능**을 담당했습니다.

| 동시성 | 약품검색 중앙값 | LLM 안전성 | 누적 PR |
|---|---|---|---|
| **99×** (3.3→331 req/s) | **580→7ms** (캐시 히트) | **3.25→4.5** / 5 | **139** |

- **외부 의존성 격리** — 약품 검색이 매번 식약처 공공 API를 호출하던 걸 Redis에 캐싱. 응답 중앙값 580ms→7ms도 있지만, 핵심은 식약처 API가 503으로 죽어도 캐싱된 검색어는 정상 응답한다는 걸 테스트로 증명한 것(장애 격리).
- **비동기 전환** — 동기 LLM 호출이 이벤트 루프를 막던 구간을 async + ARQ로 바꿔 동시 100건 처리량을 99배 개선 (locust 측정).
- **2단계 안전 필터** — 키워드 필터 v1 → 문맥 보강 v2로, LLM-as-judge 안전성 점수 3.25→4.5.
- **RAG 실험(#171)** — 임베딩 검색 vs 프롬프트 전체 주입을 비교해 품질은 동등(4.98/5.0)하면서 토큰 73.6% 절감, 챗봇에 임베딩 코사인 유사도 검색으로 연동.
- *시행착오*: 처음 잰 P95 7초가 더미 키 때문에 나온 비정상값이었음을 멘토 피드백으로 발견 → 진짜 키로 재측정(캐시 미스 P95 2초, 히트 12ms). 측정은 환경부터 의심해야 한다는 걸 배움.

`FastAPI` `Tortoise ORM` `PostgreSQL 16` `Redis` `ARQ` `OpenAI gpt-4o-mini` `text-embedding-3-small` `식약처 OpenAPI` `Docker` `GitHub Actions` `Codecov`

---

## 2. 난임 임신 성공 예측 — IVF 이진분류
`2026.04 (6일)` · 데이콘 × 오즈코딩스쿨 해커톤 · 8조 팔로리안 3인 · **팀장**

25만 건 × 69 피처, 양성 26% 불균형, ROC-AUC. 3인 팀 팀장으로 모델 전략과 제출 결정을 맡았습니다.

- num_leaves(31/63/95)·seed(42/2024/777)를 다르게 한 **LightGBM 3-model 앙상블**을 StratifiedKFold 5겹으로 학습 후 rank averaging.
- leakage 차단(결측 보정을 fold train 기준으로만 fit), Adversarial Validation, OOF↔LB 갭 모니터링으로 과적합 제출 회피.
- **결과: 팀 Public LB 0.74218 (1위 0.74246, 0.0003 차)**. OOF가 좋아진다고 LB가 따라오는 게 아니라는 걸 체득.

`LightGBM` `scikit-learn` `StratifiedKFold` `rank averaging` `Adversarial Validation`

---

## 3. 칼로리 소모 예측 — 회귀
`2026.02` · 10조의 행복 3인 · 팀원

정수 RMSE 회귀. 처음엔 피처 30개+ 2단 스태킹으로 갔다가 한 번 크게 엎었습니다.

- log1p OOF RMSE가 0.0236으로 좋게 나왔지만 **expm1 역변환에서 고칼로리 오차가 증폭**돼 LB가 망가짐(v18 0.886→v19 4.06). raw target으로 전환.
- 물리 구조 기반 2-Stage(선형 변수 → 신체 조건 보정) + Ridge 메타모델, 정수 RMSE를 직접 노린 Shift Optimization(−0.49~+0.49, step 0.001).
- **결과: Validation RMSE 0.857 → 0.204**. 복잡한 게 항상 좋은 건 아니라는 걸 확실히 느낌.

`Python` `scikit-learn` `Ridge Stacking` `2-Stage 모델링` `Shift Optimization`

---

## 4. 흡연 여부 데이터 분석 — EDA·통계 검정
`2026.01` · 9조 3인 · 데이터 분석 담당

건강검진 데이터(7천 명·17지표)로 흡연자/비흡연자 지표 차이를 통계 검정. 가설 수립·검정 파트 담당.

- 교란 변수를 통제하려고 BMI를 정상 범위로 고정. 정상 BMI 흡연자의 TG/HDL이 더 높은지 t-test(5.61 vs 4.79, t=6.96, p&lt;0.001).
- BMI 군별 헤모글로빈/간효소 분포 차이는 Mann–Whitney U(전 군 p&lt;0.001), 정상 BMI 흡연자 크레아티닌(로그)도 유의.
- **결과: 가설 3건 모두 p&lt;0.001**. 체형을 통제해도 흡연 신호가 일관됨. 데이터를 돌리기 전에 뭐가 진짜 차이를 만드는지 의심하는 습관이 생김.

`Python` `pandas` `t-test` `Mann–Whitney U` `seaborn`

---

## 기술 스택

| 분류 | |
|---|---|
| **언어 / ML** | Python · scikit-learn · LightGBM · XGBoost · CatBoost · pandas |
| **백엔드** | FastAPI · Tortoise ORM · ARQ(비동기 워커) · Redis · PostgreSQL 16 |
| **LLM / AI** | OpenAI gpt-4o-mini · text-embedding-3-small · RAG(임베딩 코사인 유사도) · LLM-as-judge 평가 |
| **협업 / 품질** | Docker · GitHub Actions · Codecov · Git Flow + PR 리뷰 · locust |
| **방법론** | 통계적 가설 검정 · Adversarial Validation · OOF 후처리 보정 |

---

## 경력

- **SK바이오사이언스** 품질관리 이화학파트 (인턴, 2021.06~2022.10)
- **경북바이오산업연구원** (인턴, 2020.06~2020.08)

제약·바이오 QC 실무에서 익힌 기준 준수와 기록 습관을, 지금은 의료 데이터의 정량적 검증으로 이어가고 있습니다.
