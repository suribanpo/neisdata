# app.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
import streamlit as st
from io import BytesIO
import openpyxl

# ── 페이지 설정 ──
st.set_page_config(page_title="파일 업로드 및 등급 분석", page_icon="🧮")
st.title("🧮 나이스 등급 분석")

# ── 사이드바 설정 ──
with st.sidebar:
    st.header("🔧 설정")
    st.markdown("⚠️ **파일명에 한글/공백/특수문자가 포함되지 않도록 저장 후 업로드하세요.**")
    uploaded_file = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type="xlsx")
    top_n = st.number_input("상위 N명 보기", min_value=1, max_value=300, value=20)
    bin_width = st.slider("히스토그램 bin 너비", min_value=1, max_value=20, value=10)
    st.markdown("---")

# ── 데이터 업로드 처리 ──
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    df_raw = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")

    # 1. 데이터 전처리
    def preprocess_data(df):
        df = df.iloc[7:37, [1, 2, 3, 4, 6, 7, 9, 10, 12, 15, 16]]
        df.columns = ['번호', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.melt(id_vars='번호', value_name='점수', var_name='반')
        df = df.dropna()
        return df

    df = preprocess_data(df_raw)

    # 2. 등급 부여
    def assign_grade(df_sorted, thresholds):
        total = len(df_sorted)
        df_sorted['누적 비율'] = (df_sorted.index + 1) / total
        def get_grade(row):
            for i, t in enumerate(thresholds):
                if row['누적 비율'] <= t:
                    return i + 1
            return 9
        df_sorted['등급'] = df_sorted.apply(get_grade, axis=1)
        return df_sorted

    # 3. 등급 컷 계산
    def calculate_grade_cutoffs(df_sorted):
        cutoffs = []
        prev_grade = None
        for i, row in df_sorted.iterrows():
            curr_grade = row['등급']
            if prev_grade is not None and curr_grade != prev_grade:
                cutoffs.append({
                    "등급": f"{prev_grade}등급",
                    "컷 점수": df_sorted.iloc[i - 1]["점수"]
                })
            prev_grade = curr_grade
        return pd.DataFrame(cutoffs)

    thresholds = [0.04, 0.11, 0.23, 0.40, 0.60, 0.77, 0.89, 0.96, 1.00]
    df_sorted = df.sort_values("점수", ascending=False).reset_index(drop=True)
    df_sorted = assign_grade(df_sorted, thresholds)
    df_sorted = df_sorted[["반", "번호", "점수", "누적 비율", "등급"]]
    cutoffs_df = calculate_grade_cutoffs(df_sorted)

    # ── 상위 성적자 표시 ──
    st.subheader(f"🏅 상위 {top_n}명 성적")
    st.dataframe(df_sorted.head(top_n), use_container_width=True)

    # ── 평균 점수 표시 ──
    # ── 평균, 표준편차, 학생 수 메트릭 카드 ──
    avg_score = df_sorted['점수'].mean()
    std_score = df_sorted['점수'].std()
    n_students = len(df_sorted)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("전체 평균 점수", f"{avg_score:.2f}점")
    with c2:
        st.metric("표준편차", f"{std_score:.2f}점")
    with c3:
        st.metric("전체 학생 수", f"{n_students}명")

    # ── 시각화 ──
    st.subheader("📊 점수 분포 및 반별 분포")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.histplot(df_sorted["점수"], binwidth=bin_width, ax=ax1, edgecolor="white", color="#3498db", alpha=0.85)
        ax1.set_title("점수 분포 (등급 컷 포함)", fontsize=14)
        ax1.set_xlabel("점수", fontsize=12)
        ax1.set_ylabel("학생 수", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.3)

        # 등급 컷 시각화
        colors = sns.color_palette("Set1", n_colors=len(cutoffs_df))
        for i, row in enumerate(cutoffs_df.itertuples()):
            score = row._2  # 컷 점수
            grade = row.등급
            ax1.axvline(x=score, color=colors[i], linestyle='--', linewidth=2, label=grade)
            ax1.text(score, ax1.get_ylim()[1]*0.9, grade,
                    color=colors[i], fontsize=9, ha='right', rotation=90)

        ax1.legend(title="등급 컷", loc="upper right", fontsize=8, frameon=False)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df_sorted['반'] = pd.Categorical(df_sorted['반'], categories=[str(i) for i in range(1, 11)], ordered=True)
        sns.boxplot(x="반", y="점수", data=df_sorted, ax=ax2, palette="pastel")
        ax2.set_title("반별 점수 분포", fontsize=14)
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig2)

    # ── 등급 컷 테이블 추가 표시 ──
    st.subheader("📋 등급 컷 표")
    cutoffs_table = cutoffs_df.rename(columns={"등급": "Grade", "컷 점수": "Cut-off Score"})
    cutoffs_table["Cut-off Score"] = cutoffs_table["Cut-off Score"].round(2)
    st.dataframe(cutoffs_table.style.format({"Cut-off Score": "{:.2f}"}), use_container_width=True)


else:
    st.warning("엑셀 파일(.xlsx)을 업로드해주세요. ⚠️ 파일명은 영어/숫자만 사용하세요.")
