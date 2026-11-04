# AI 물 음용 가능성 분류 웹 앱

Streamlit과 scikit-learn의 의사결정나무를 이용한 교육용 프로젝트입니다.

## 파일 구성
- `model.py`: 데이터 검증, 결측치 중앙값 대체, 학습, 평가
- `app.py`: 파일 업로드 및 결과 표시
- `requirements.txt`: 설치할 라이브러리
- `water_potability.csv`: **구조 확인용 예시 데이터(실제 전체 데이터 아님)**

## Windows 실행 방법
```bat
cd %USERPROFILE%\Desktop\ML_Project
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## GitHub / Streamlit Community Cloud 배포
1. GitHub에서 `ML_Project` 저장소를 생성합니다.
2. 프로젝트 파일을 저장소 최상위 경로에 업로드합니다.
3. Streamlit Community Cloud에서 해당 저장소와 `main` 브랜치를 선택합니다.
4. Main file path를 `app.py`로 설정한 뒤 배포합니다.

## 데이터 형식
`Potability`가 0 또는 1인 숫자형 열이어야 하며 나머지 입력 열도 숫자형이어야 합니다. 각 클래스는 최소 2개 이상의 행이 필요합니다. 예시 CSV는 형식만 보여주므로 실제 분석에는 충분한 크기의 데이터를 사용하세요.

## 주의
이 앱은 업로드한 정답 포함 데이터로 새 모델을 학습하고 테스트합니다. 정답이 없는 새 데이터만으로 예측하는 기능은 포함하지 않습니다. 학습 및 평가 결과는 실제 수질 안전성 판정에 사용할 수 없습니다.
