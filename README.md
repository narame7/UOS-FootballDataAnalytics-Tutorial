# ⚽ UOS02057 축구 데이터 분석 (Introduction to Football Data Analytics) — 2026년 2학기

서울시립대학교 인공지능학과 **축구 데이터 분석** 수업의 실습 저장소입니다.
매주 강의에서 배운 모델(xG, xT, VAEP, 피치 컨트롤, EPV, OBSO, 플레이 스타일, 포메이션 검출, 축구 LLM …)을
실제 축구 데이터 위에서 직접 재현해 보는 Jupyter 노트북과 보조 코드, 데이터가 들어 있습니다.

| 항목 | 내용 |
|---|---|
| 강의 | 매주 월요일 14:00–16:00 |
| 실습 | 매주 월요일 16:00–18:00 (강의 직후, 조교 진행) |
| 첫 수업 | 2026년 9월 7일 (월) |
| 담당 교수 | 고상기 (sangkiko@uos.ac.kr) · [CIDA Lab](https://sites.google.com/site/sangkikotoc/) |
| 실습 조교 | 개강 후 공지 |
| 언어 · 환경 | Python 3.10+, Jupyter Notebook / Google Colab |

<br>

## 🌿 브랜치 안내

| 브랜치 | 용도 |
|---|---|
| [`main`](../../tree/main) | **올해(2026년 2학기) 자료** — 실습 전 주에 노트북이 갱신됩니다. 항상 이 브랜치를 보세요. |
| [`2026`](../../tree/2026) | 올해 자료의 작업 브랜치 (`main`과 동일하게 유지) |
| [`2025`](../../tree/2025) | 2025년 2학기 자료 보관용 |

<br>

## 🗓️ 2026년 2학기 일정과 실습 폴더

강의 주차와 이 저장소의 폴더 번호는 다릅니다(폴더 번호는 작년 기준이며, 노트북을 갱신하면서 차례로 정리할 예정입니다).
아래 표의 **실습 폴더** 열을 보고 그 주의 노트북을 여시면 됩니다.

| 주차 | 날짜 | 강의 주제 | 실습 폴더 (노트북) |
|---|---|---|---|
| 1 | 9/7 | 강의 소개 | — (Colab 준비) |
| 2 | 9/14 | 데이터 수집 방법과 파이썬 기초 | `Week 2` — 통계·이벤트·트래킹 데이터 불러오기 |
| 3 | 9/21 | 이벤트 데이터 · **과제 1** | `Week 4` — 이벤트 데이터 분석과 시각화 |
| 4 | 9/28 | 기계학습 기초 | `Week 5` — 이벤트 데이터 기계학습 |
| 5 | 10/5 | 휴강 (개천절 대체휴일) | — |
| 6 | 10/12 | 트래킹 데이터 기초 | `Week 6` — 트래킹 데이터 시각화 |
| 7 | 10/19 | 딥러닝 기초 · **과제 2** | (트래킹 데이터 실습 계속) |
| 8 | 10/26 | 기대 득점(xG)과 기대 위협(xT) | `Week 8` — Expected Threat |
| 9 | 11/2 | 적용 사례, **프로젝트 주제 발표** | — |
| 10 | 11/9 | 선수 플레이 가치 평가 · **과제 3** | `Week 9` — SPADL과 VAEP |
| 11 | 11/16 | 오프더볼 기여도 평가 | `Week 11` — 피치 컨트롤 · EPV · OBSO |
| 12 | 11/23 | **프로젝트 중간 발표** (10/5 보강) | — |
| 13 | 11/30 | 플레이 스타일 분석 · **과제 4** | `Week 13` — SoccerMix, `Week 14` — SoccerCPD 포메이션 클러스터링 |
| 14 | 12/7 | 세트피스 분석 | (스타일 실습 계속) |
| 15 | 12/14 | 축구 데이터 분석의 미래 | `Week 15` — 축구 LLM 앱 |
| 16 | 12/21 | **프로젝트 최종 발표** · **과제 5** | — |

<br>

## 📂 실습 폴더 소개

각 폴더는 **노트북(`*.ipynb`) + 보조 코드(`*.py`) + 데이터(`data.zip`, `*.tar.xz`)** 로 이루어져 있습니다.
노트북은 번호 순서대로 실행하세요.

### `Week 2` — 데이터 불러오기
- `1-load-statistic-data.ipynb` — `ScraperFC`·`soccerdata`로 ClubElo, Understat, Football-Data 통계 데이터를 불러와 살펴보기 (2026/27 시즌 기준)
  - FBref는 봇 차단(Cloudflare) 때문에 브라우저 없는 Colab에서 크롤링이 안 되어 다루지 않습니다.
  - ClubElo(api.clubelo.com)와 Football-Data 서버가 응답하지 않으면 해당 셀은 안내 메시지만 출력하고 넘어갑니다.
- `2-load-event-and-tracking-data.ipynb` — Metrica Sports 샘플 이벤트·트래킹 데이터(CSV)를 불러와 좌표 변환, 속도 계산, 장면 시각화, 영상 저장
- 필요 라이브러리: `pandas`, `matplotlib`, `seaborn`, `scipy`, `ScraperFC`, `soccerdata` (영상 저장에는 `ffmpeg` 필요)

### `Week 4` — 이벤트 데이터 분석과 시각화
- `1-event-data-analysis-and-visualization.ipynb` — 패스 맵, 슈팅 맵, 히트맵, 패스 네트워크를 `mplsoccer`로 그리기
- `data_utils.py`, `plot_utils.py` — 데이터 로딩·시각화 헬퍼 (`data.zip`을 같은 폴더에 풀어 두세요)

### `Week 5` — 이벤트 데이터 기계학습
- `1-event-data-machine-learning.ipynb` — 이벤트 데이터로 특징을 만들고 `scikit-learn`으로 분류·회귀 모델(기대 득점의 첫걸음) 학습
- `data.zip`을 같은 폴더에 풀어 두세요

### `Week 6` — 트래킹 데이터 시각화
- `1-tracking-data-visualization.ipynb` — 22명의 좌표를 애니메이션으로 그리고 속도·거리 같은 파생 지표 계산
- `data.tar.xz`, `dfl_data.tar.xz` — Metrica Sports 샘플과 DFL(분데스리가) 공개 트래킹 데이터, `functions/` — 헬퍼 함수

### `Week 8` — 기대 위협(Expected Threat, xT)
- `1-run-expected-threat.ipynb` — 이동·슈팅 확률 격자를 만들고 xT 값을 반복 계산해 선수·팀 기여도 산출
- `socceraction/`, `matplotsoccer.py` — SPADL 변환과 시각화 (KU Leuven의 socceraction 라이브러리 포함)

### `Week 9` — SPADL과 VAEP
- `1-run-spadl-vaep.ipynb` — 이벤트를 SPADL로 변환하고, 득점·실점 확률 모델을 학습해 모든 액션의 VAEP 값 계산
- `2-run-atomic-spadl-vaep.ipynb` — Atomic-SPADL 버전
- 참고: Decroos et al., *Actions Speak Louder than Goals*, KDD 2019

### `Week 11` — 피치 컨트롤 · EPV · OBSO
- `1-run-obso.ipynb` — Metrica 트래킹 데이터로 피치 컨트롤(Spearman)과 EPV 격자를 결합해 오프볼 득점 기회(OBSO) 계산
- `Metrica_IO.py`, `Metrica_Velocities.py`, `Metrica_PitchControl.py`, `Metrica_EPV.py` — Friends of Tracking 기반 코드
- `calculate_obso.py`, `obso_player.py`, `trace_snapshot.py`, `EPV_grid.csv`, `Transition_gauss.csv` — OBSO 계산과 장면 스냅샷
- 이 폴더만의 `requirements.txt`가 있습니다 (`torch`, `xgboost` 등)

### `Week 13` — SoccerMix로 본 플레이 스타일
- `notebooks/1-create-mixture-models.ipynb` — 액션 위치·방향의 혼합 모델(mixture model) 학습
- `notebooks/2-case-study.ipynb` — 선수·팀 스타일 비교 사례
- `mixture.py`, `feature.py`, `vis.py`, `data/`

### `Week 14` — SoccerCPD 포메이션 클러스터링
- `notebooks/formation_clustering.ipynb` — 트래킹 데이터에서 역할 배정과 포메이션 변화점 검출 (Kim et al., KDD 2022)
- `src/`, `data/`, `img/`

### `Week 15` — 축구 LLM
- `1-run-football-llm.ipynb` — 이벤트 데이터 위에서 검색 증강 생성(RAG)으로 자연어 질의응답 만들기
- `app.py` — Streamlit 데모 앱, `README.md` — 설치 안내, `requirements.txt` (LangChain, Chroma, Anthropic/OpenAI 클라이언트 등)

<br>

## 🚀 시작하기

### 방법 1 — Google Colab (권장, 설치 없음)

1. 이 저장소에서 열고 싶은 노트북을 클릭합니다.
2. 주소의 `github.com`을 `colab.research.google.com/github`로 바꾸거나, 아래 형식의 링크를 씁니다.

   ```
   https://colab.research.google.com/github/narame7/UOS-FootballDataAnalytics-Tutorial/blob/main/Week%202/1-load-statistic-data.ipynb
   ```
3. 노트북 첫 셀에 있는 `pip install ...` 셀을 먼저 실행합니다.
4. 데이터 파일(`data.zip` 등)이 필요한 노트북은 첫 셀에서 저장소를 클론한 뒤 압축을 풉니다.

   ```python
   !git clone https://github.com/narame7/UOS-FootballDataAnalytics-Tutorial.git
   %cd "UOS-FootballDataAnalytics-Tutorial/Week 4"
   !unzip -q data.zip
   ```

### 방법 2 — 로컬 설치

```bash
git clone https://github.com/narame7/UOS-FootballDataAnalytics-Tutorial.git
cd UOS-FootballDataAnalytics-Tutorial

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install jupyter pandas numpy matplotlib seaborn scikit-learn \
            statsbombpy kloppy mplsoccer matplotsoccer socceraction

jupyter lab
```

- `Week 11`과 `Week 15`는 폴더 안의 `requirements.txt`로 별도 환경을 만드는 것을 권합니다 (PyTorch·LangChain 등 무거운 의존성).
- Python 3.10 이상을 권장합니다 (`Week 15`는 3.10.18에서 검증).
- 압축 데이터는 각 폴더 안에서 풉니다: `unzip data.zip`, `tar -xJf data.tar.xz`.

<br>

## 📊 사용하는 데이터

| 데이터 | 종류 | 출처 |
|---|---|---|
| StatsBomb Open Data | 이벤트 + 360 프리즈 프레임 | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) (`statsbombpy`로 접근) |
| Metrica Sports Sample Data | 이벤트 + 트래킹 (25 Hz) | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| DFL / Sportec 공개 데이터 | 분데스리가 이벤트 + 트래킹 | Bassek et al., *Scientific Data* 2025 (`kloppy` 로더) |
| 수업용 가공 데이터 | 각 폴더의 `data.zip` / `*.tar.xz` | 이 저장소 |

데이터는 **수업 목적으로만** 사용해 주세요. 각 원 출처의 라이선스를 따릅니다.

<br>

## 📝 과제와 프로젝트

- **과제 5회 (40%)** — 3·7·10·13·16주차에 공지. 그 주까지의 실습 노트북을 확장하는 형태입니다.
- **프로젝트 (제안 10% + 최종 30%)** — 9주차 주제 발표, 12주차 중간 발표, 16주차 최종 발표. 실습에서 쓴 코드와 데이터를 그대로 재사용할 수 있습니다.
- **출석·참여 (10% + 10%)**
- 제출 방법과 마감은 수업 시간과 학교 LMS(uclass)에서 공지합니다.

<br>

## 📚 함께 읽으면 좋은 자료

- David Sumpter, *Soccermatics* (Bloomsbury, 2016) — 이 수업의 교과서에 가장 가까운 책
- Ian Graham, *How to Win the Premier League* (Century, 2024) — 리버풀 데이터 혁명의 내막
- Chris Anderson & David Sally, *The Numbers Game* (Penguin, 2013)
- Karun Singh, [Introducing Expected Threat (xT)](https://karun.in/blog/expected-threat.html) (2019)
- [Friends of Tracking](https://www.youtube.com/@friendsoftracking) — 트래킹 데이터 분석 무료 강의 시리즈
- [socceraction 문서](https://socceraction.readthedocs.io/) · [mplsoccer 갤러리](https://mplsoccer.readthedocs.io/en/latest/gallery/index.html) · [kloppy 문서](https://kloppy.pysport.org/)

<br>

## 🙋 문의

- 실습·과제 관련: 실습 시간에 조교에게, 또는 uclass 게시판
- 그 외: sangkiko@uos.ac.kr
- 저장소 오류나 개선 제안은 GitHub **Issues**로 남겨 주세요. Pull Request도 환영합니다.

<br>

## 🛠️ 기술 스택

* **언어:** Python
* **주요 라이브러리:** `pandas`, `NumPy`, `Matplotlib` & `Seaborn`, `scikit-learn`, `PyTorch`(일부), `statsbombpy`, `kloppy`, `mplsoccer`, `socceraction`, `LangChain`(Week 15)
* **개발 환경:** Jupyter Notebook / JupyterLab, Google Colab, VS Code
