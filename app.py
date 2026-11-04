"""Streamlit app for teaching water potability classification."""
import pandas as pd
import streamlit as st
from model import train_model

st.set_page_config(page_title="AI 물 음용 가능성 예측", page_icon="💧", layout="wide")
st.title("💧 AI 물 음용 가능성 예측 시스템")
st.write("CSV 또는 엑셀 파일을 업로드하여 의사결정나무 분류 모델을 학습하고 평가합니다.")
st.info("데이터에는 정답 열인 Potability(0: 음용 부적합, 1: 음용 가능)가 필요합니다. 결과는 학습용이며 실제 음용 안전성을 보증하지 않습니다.")
file = st.file_uploader("학습 데이터 업로드", type=["csv", "xlsx"])
if file is None:
    st.caption("예시: 프로젝트 폴더의 water_potability.csv 파일을 업로드하세요. 예시 파일은 구조 확인용 소량 데이터로 모델 성능을 판단하기에 적합하지 않습니다.")
else:
    try:
        df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        st.subheader("1. 데이터 확인")
        st.write(f"행: {len(df):,}개 / 열: {len(df.columns):,}개")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("2. 결측치 확인")
        st.dataframe(df.isna().sum().rename("결측치 수"), use_container_width=True)
        depth = st.slider("의사결정나무 최대 깊이", min_value=1, max_value=20, value=5)
        if st.button("모델 학습 및 평가", type="primary"):
            with st.spinner("모델을 학습하고 있습니다..."):
                model, metrics, report, matrix = train_model(df, max_depth=depth)
            st.subheader("3. 분류 성능 평가")
            cols = st.columns(4)
            for col, (label, score) in zip(cols, metrics.items()):
                col.metric(label, f"{score:.1%}")
            st.subheader("4. 혼동행렬")
            st.caption("행: 실제값 / 열: 예측값")
            matrix_df = pd.DataFrame(matrix, index=["실제 0", "실제 1"], columns=["예측 0", "예측 1"])
            st.dataframe(matrix_df, use_container_width=True)
            st.subheader("5. 상세 분류 보고서")
            st.code(report, language="text")
            st.success("모델 학습과 평가가 완료되었습니다.")
    except (ValueError, KeyError, pd.errors.ParserError, ImportError) as exc:
        st.error(f"파일 또는 데이터 처리 오류: {exc}")
    except Exception as exc:
        st.error(f"처리 중 오류가 발생했습니다: {exc}")
