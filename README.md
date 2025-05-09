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