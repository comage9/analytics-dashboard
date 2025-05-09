# Analytics Dashboard 프로젝트

## 프로젝트 개요

**Analytics Dashboard**는 출고/판매 데이터를 기반으로 실시간 현황, 트렌드, 예측, AI 인사이트 분석을 제공하는 통합 대시보드입니다. 구글 시트 데이터를 자동으로 동기화하여, 웹 기반으로 시각화 및 심층 분석을 지원합니다.

---

## 주요 기능
- **구글 시트 연동**: 출고/판매 데이터를 주기적으로 자동 동기화(증분 upsert)
- **현황 분석**: 일별/주간/월간 출고량, 판매금액, 품목/분류별 트렌드 시각화
- **예측 분석**: Prophet 기반 미래 출고량 예측, 신뢰구간 제공
- **AI 인사이트**: (옵션) LLM 기반 심층 보고서 자동 생성(현재 중지됨)
- **실시간/요일별 출고 현황**: 당일 및 최근 4주 요일별 시간대별 출고량 분석
- **Looker Studio 연동**: 외부 BI 리포트 바로가기

---

## 기술 스택
- **백엔드**: Python, FastAPI, pandas, sqlite3, APScheduler
- **프론트엔드**: React, TypeScript, Chart.js, axios
- **AI/예측**: Prophet, scikit-learn, (옵션) Ollama LLM
- **데이터 소스**: Google Sheets (CSV API)

---

## 폴더 구조
```
├── app/                # FastAPI 백엔드 및 데이터 처리
│   └── main.py         # API, 데이터 동기화, 분석 로직
├── frontend/           # React 프론트엔드
│   └── src/            # 주요 컴포넌트, 스타일 등
├── forecast.py         # 예측 관련 함수
├── requirements.txt    # Python 의존성
├── README.md           # 프로젝트 설명 및 안내
└── ...
```

---

## 설치 및 실행 방법
1. **Python 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```
2. **프론트엔드 설치/실행**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. **백엔드 실행**
   ```bash
   uvicorn app.main:app --reload
   ```
4. **웹 브라우저에서 접속**
   - 기본 주소: http://localhost:3000 (프론트)
   - API 문서: http://localhost:8000/docs

---

## 데이터 흐름 및 동기화 구조
- **구글 시트 → CSV 다운로드 → pandas DataFrame → sqlite3 DB upsert**
- 리프레시 시 변경분만 upsert하여 속도 최적화
- DB → API → 프론트엔드로 데이터 전달 및 시각화

---

## 주요 특징 및 장점
- **증분 동기화**: 전체 데이터가 아닌 변경분만 반영하여 빠른 리프레시
- **모듈화**: 백엔드/프론트엔드 분리, 유지보수 용이
- **확장성**: AI 분석, 외부 BI 연동 등 기능 확장 가능
- **실시간성**: 구글 시트 데이터가 변경되면 빠르게 반영
- **사용자 친화적 UI**: 현대적 대시보드 스타일, 다양한 시각화 제공

---

## 활용 예시
- 물류/유통/제조사의 출고/판매 실적 모니터링
- 실시간 트렌드 및 예측 기반 의사결정 지원
- AI 기반 심층 분석 보고서 자동화(옵션)

---

## 유지보수 및 확장 가이드
- **구글 시트 구조 변경 시**: app/main.py의 컬럼 매핑/키 조정 필요
- **AI 분석 재활성화**: /api/insight 엔드포인트 주석 해제 및 LLM 서버 준비
- **DB 교체**: sqlite3 → MySQL/PostgreSQL 등으로 확장 가능
- **프론트엔드 커스터마이즈**: src/components 내 컴포넌트 수정

---

## 기여 방법
1. 이슈 등록 또는 Pull Request 제출
2. 주요 변경 시 README, 주석, 예제 코드 보강 권장
3. 문의: comage9@gmail.com

---

# Analytics Dashboard 프로젝트 깃허브 업로드 안내

이 문서는 **다른 컴퓨터에서 이 프로젝트를 깃허브에 업로드(푸시)하는 방법**과 필요한 준비 사항을 안내합니다.

---

## 1. 사전 준비

1. **Git 설치**
   - [Git 공식 다운로드](https://git-scm.com/downloads)에서 운영체제에 맞게 설치

2. **깃허브 계정**
   - [GitHub](https://github.com/)에서 계정 생성 및 로그인

3. **깃허브 Personal Access Token(토큰) 준비**
   - [토큰 생성 가이드](https://docs.github.com/ko/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
   - 토큰은 비밀번호 대신 사용됨 (권장: repo 권한 포함)

---

## 2. 프로젝트 폴더에서 깃허브 업로드(푸시) 방법

### 1) 터미널(명령 프롬프트, PowerShell, Git Bash 등) 실행 후 프로젝트 폴더로 이동
```bash
cd 프로젝트_폴더_경로
```

### 2) git 초기화(이미 되어 있다면 생략)
```bash
git init
```

### 3) 원격 저장소(origin) 설정 (이미 있다면 변경)
```bash
git remote remove origin  # 기존 origin이 있다면 삭제
```
```bash
git remote add origin https://github.com/comage9/analytics-dashboard.git
```

### 4) 변경사항 스테이징 및 커밋
```bash
git add .
git commit -m "프로젝트 업로드"
```

### 5) 강제 푸시(기존 내용 덮어쓰기)
```bash
git push -f origin main
```
- **토큰 입력 요청 시**: 아이디 대신 토큰을 비밀번호 자리에 붙여넣기

---

## 3. 자주 발생하는 문제 및 해결법

- **인증 오류**: 토큰이 만료되었거나 권한이 부족할 수 있음 → 새 토큰 생성 후 사용
- **브랜치 이름(main/master) 불일치**: `git branch -M main` 명령으로 main 브랜치로 통일
- **권한 오류**: 저장소 Collaborator로 추가되어 있는지 확인

---

## 4. 참고
- [깃허브 공식 문서](https://docs.github.com/ko)
- [토큰 인증 안내](https://docs.github.com/ko/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

---

**문의: comage9@gmail.com 또는 깃허브 이슈 등록** 