import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# 加载数据
st.title("🕜 Time Series Classification")
st.sidebar.title("⭕️ Fine Tuning")
uploaded_file = st.file_uploader("Upload Data File", type=["csv"])
if uploaded_file is not None:
    uploaded_file.seek(0)
    try:
        data = pd.read_csv(uploaded_file, encoding='utf-8')
        st.write(data.head())
    except pd.errors.EmptyDataError:
        st.error("File is null, please examine it again.")
    except Exception as e:
        st.error(f"Something wrong in reading file: {e}")

    # 配置参数
    target_column = st.sidebar.selectbox("Select Target", data.columns)
    feature_columns = st.sidebar.multiselect("Select Feature", data.columns.drop(target_column))
    test_size = st.sidebar.slider("Proportion for testing", 0.1, 0.5, 0.2)

    # 训练分类算法
    if st.button("Run the model"):
        # 选择特征列
        X = data[feature_columns].values  
        y = data[target_column].values  

        # 将特征转换为嵌套格式
        X_nested = np.array([X[i, :] for i in range(X.shape[0])]).reshape(X.shape[0], -1, len(feature_columns))

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_nested, y, test_size=test_size, random_state=42)

        # 训练
        clf = TimeSeriesForestClassifier()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # 输出结果
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy for model: {accuracy:.2f}")

        # 可视化数据和结果
        st.bar_chart(data[target_column])
        for feature in feature_columns:
            st.line_chart(data[feature])

        # 输出预测结果
        results_df = pd.DataFrame({
            'Real label': y_test,
            'Predict label': predictions
        })
        st.write("Comparison:")
        st.write(results_df)
        st.bar_chart(results_df)

        # 绘制混淆矩阵
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
        plt.xlabel('Real label')
        plt.ylabel('Predict label')
        plt.title('confusion matrix')
        st.pyplot(plt)

        # 输出分类报告
        unique_labels = np.unique(y_train)  # 获取训练集中的唯一标签
        report = classification_report(y_test, predictions, labels=unique_labels)
        st.text("Classification Report")
        st.text(report)
