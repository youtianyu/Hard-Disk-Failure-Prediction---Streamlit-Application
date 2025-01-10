import streamlit as st
import pandas as pd
import joblib
import numpy as np
def map_to_seven(value):
    if value < 0:
        value = 0
    elif value > 100:
        value = 100
    
    # 将0-100的范围映射到0-7
    mapped_value = value * 7 / 100
    # 向下取整得到最终结果
    return int(mapped_value)
st.set_page_config(page_title="硬盘故障预测器",page_icon=":computer:",layout="wide")
Language = st.sidebar.selectbox("Language:",["中文","English"],index=1)
st.title("硬盘故障预测器"if Language=="中文" else "Hard drive failure predictor")
smart_values = {}
if Language == "English":
    st.write("Please enter the original SMART value of the hard disk:")
else:
    st.write("请输入硬盘的原始SMART值：")
col11,col12,col13,col14,col15 = st.columns(5)
col21,col22,col23,col24,col25 = st.columns(5)
with col11:
    smart_values["smart_5_raw"] = st.number_input("smart_5_raw:",min_value=0,max_value=9007199254740991)
with col12:
    smart_values["smart_9_raw"] = st.number_input("smart_9_raw:",min_value=0,max_value=9007199254740991)
with col13:
    smart_values["smart_187_raw"] = st.number_input("smart_187_raw:",min_value=0,max_value=9007199254740991)
with col14:
    smart_values["smart_188_raw"] = st.number_input("smart_188_raw:",min_value=0,max_value=9007199254740991)
with col15:
    smart_values["smart_193_raw"] = st.number_input("smart_193_raw:",min_value=0,max_value=9007199254740991)
with col21:
    smart_values["smart_194_raw"] = st.number_input("smart_194_raw:",min_value=0,max_value=9007199254740991)
with col22:
    smart_values["smart_197_raw"] = st.number_input("smart_197_raw:",min_value=0,max_value=9007199254740991)
with col23:
    smart_values["smart_198_raw"] = st.number_input("smart_198_raw:",min_value=0,max_value=9007199254740991)
with col24:
    smart_values["smart_241_raw"] = st.number_input("smart_241_raw:",min_value=0,max_value=9007199254740991)
with col25:
    smart_values["smart_242_raw"] = st.number_input("smart_242_raw:",min_value=0,max_value=9007199254740991)

if st.button("预测" if Language == "中文" else "Predict"):
    with st.spinner("加载模型..." if Language == "中文" else "Loading model..."):
        model_r = joblib.load("r.joblib")
        model_cn3 = joblib.load("cn3.joblib")
        model_cn7 = joblib.load("cn7.joblib")
    with st.spinner("预测中..." if Language == "中文" else "Predicting..."):
        DataFrame = pd.DataFrame(smart_values,index=[0])
        r = model_r.predict(DataFrame)[0]
        cn3 = model_cn3.predict(DataFrame)[0]
        cn3_proba = model_cn3.predict_proba(DataFrame)[0]
        cn7 = model_cn7.predict(DataFrame)[0]
        cn7_proba = model_cn7.predict_proba(DataFrame)[0]
    st.write("预测结果: " if Language == "中文" else "Prediction result: ")
    st.write("硬盘健康度为: ",100-r)
    if r<15 and cn3 < 1.0 and cn7 < 1.0:
        st.write("硬盘健康优秀" if Language == "中文" else "The hard disk is in excellent health")
    elif r<30 and cn3 < 1.0 and cn7 < 1.0:
        st.write("硬盘健康良好" if Language == "中文" else "The hard disk is in good health")
    elif r<45 and cn3 < 1.0 and cn7 < 1.0:
        st.write("硬盘健康一般" if Language == "中文" else "The hard disk is in average health")
    elif r<60:
        st.write("硬盘健康较差"if Language == "中文" else "The hard disk is in poor health")
    else:
        st.write("硬盘健康非常差"if Language == "中文" else "The hard disk is in very poor health")
    if cn3 >= 1.0:
        st.write("并且硬盘将在大约 "if Language == "中文" else "The hard drive will fail in about ",[10000,480,48][int(cn3)]," 小时内故障" if Language == "中文" else " hours.")
    if cn7 >= 1.0:
        st.write("你的硬盘最多还能用大约 "if Language == "中文" else "The hard drive will last for about ",[10000,100,80,60,40,20,2][int(cn7)]," 天" if Language == "中文" else " days.")
    st.caption("预测结果仅供参考, 不应作为硬盘故障的判断依据" if Language == "中文" else "The prediction result is for reference only and should not be used as a basis for determining hard disk failure.")
    st.caption("模型训练数据来自 Backblaze" if Language == "中文" else "The training data of the model comes from Backblaze")
    st.caption("模型制作程序参考 https://github.com/cyyself/big-data-homework-artifacts/tree/master" if Language == "中文" else "The model making program refers to https://github.com/cyyself/big-data-homework-artifacts/tree/master")
    # 绘制概率折线图
    st.write("概率图: " if Language == "中文" else "Probability chart: ")
    # 创建一个包含两个数据集的数据框
    r_num = np.array([0.0 if i!=map_to_seven(r) else 1.0 for i in range(0, 7) ])
    if Language == "中文":
        df = pd.DataFrame({
            "磨损程度": range(0, 7),
            "cn3_proba": [cn3_proba[0],cn3_proba[0],cn3_proba[0],cn3_proba[0],cn3_proba[0],cn3_proba[1],cn3_proba[2]],
            "cn7_proba": cn7_proba,
            "R_result": r_num
        })
    else:
        df = pd.DataFrame({
            "Wear Level": range(0, 7),
            "cn3_proba": [cn3_proba[0],cn3_proba[0],cn3_proba[0],cn3_proba[0],cn3_proba[0],cn3_proba[1],cn3_proba[2]],
            "cn7_proba": cn7_proba,
            "R_result": r_num
    })
    # 使用 melt 函数将数据框转换为长格式
    if Language == "中文":
        df_melted = df.melt(id_vars="磨损程度", value_vars=["cn3_proba", "cn7_proba","R_result"], var_name="模型", value_name="概率")
    else:
        df_melted = df.melt(id_vars="Wear Level", value_vars=["cn3_proba", "cn7_proba","R_result"], var_name="Model", value_name="Probability")
    # 绘制叠加图表
    if Language == "中文":
        st.line_chart(df_melted, x="磨损程度", y="概率", color="模型")
    else:
        st.line_chart(df_melted, x="Wear Level", y="Probability", color="Model")
    