import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import time
import random
import numpy as np

from wordcloud import WordCloud

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Voice of Customer Analytics",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "predict"

if "feedback_input" not in st.session_state:
    st.session_state.feedback_input = ""

# =====================================================
# LOAD MODEL
# =====================================================
model = pickle.load(open("classifier.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
train_df = pd.read_csv("final_feedback_dataset.csv")

# =====================================================
# PREMIUM CSS
# =====================================================
st.markdown("""
<style>

.stApp {
    background:
    radial-gradient(circle at top left, #1e1b4b 0%, #050816 35%),
    radial-gradient(circle at bottom right, #0f172a 0%, #050816 45%);
    color: white;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

html, body, [class*="css"] {
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 96%;
}

.hero {
    padding: 2rem 2.5rem;
    border-radius: 30px;
    background:
    linear-gradient(
        135deg,
        rgba(17,25,40,0.96),
        rgba(15,23,42,0.92)
    );
    border:
    1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    box-shadow:
    0 10px 40px rgba(168,85,247,0.18);
    margin-bottom: 1.2rem;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}

.helper-text {
    color: #cbd5e1;
    line-height: 1.9;
    font-size: 1rem;
}

.nav-btn button {
    background:
    linear-gradient(
        135deg,
        #7c3aed,
        #db2777
    ) !important;

    border: none !important;
    border-radius: 18px !important;
    color: white !important;
    height: 3.3em !important;
    font-weight: 700 !important;

    box-shadow:
    0 8px 25px rgba(168,85,247,0.35);

    transition: all 0.25s ease !important;
}

.nav-btn button:hover {
    transform: translateY(-3px);

    box-shadow:
    0 12px 30px rgba(168,85,247,0.45);
}

textarea {
    background:
    rgba(2,6,23,0.92) !important;

    color: white !important;

    border-radius: 24px !important;

    border:
    1px solid rgba(168,85,247,0.22) !important;

    padding: 1rem !important;

    font-size: 1rem !important;

    min-height: 230px !important;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 18px;
    height: 3.2em;
    font-size: 1rem;
    font-weight: 700;
    color: white;

    background:
    linear-gradient(
        135deg,
        #8b5cf6,
        #ec4899
    );

    box-shadow:
    0 6px 20px rgba(168,85,247,0.35);

    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: scale(1.02);

    box-shadow:
    0 10px 30px rgba(168,85,247,0.42);
}

[data-testid="metric-container"] {
    background:
    rgba(17,25,40,0.72);

    border:
    1px solid rgba(255,255,255,0.08);

    border-radius: 20px;

    padding: 1rem;
}

[data-testid="stFileUploader"] {
    background:
    rgba(17,25,40,0.72);

    border-radius: 24px;

    border:
    1px solid rgba(255,255,255,0.08);

    padding: 1.3rem;
}

.stPlotlyChart {
    background:
    rgba(17,25,40,0.55);

    border-radius: 24px;

    padding: 0.8rem;

    border:
    1px solid rgba(255,255,255,0.05);

    margin-bottom: 1rem;
}

[data-testid="stDataFrame"] {
    background:
    rgba(17,25,40,0.72);

    border-radius: 20px;

    padding: 0.5rem;

    border:
    1px solid rgba(255,255,255,0.06);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero">

<div class="main-title">
🚀 AI-Powered Customer Feedback Analytics
</div>

<div class="helper-text">
AI-powered NLP platform for customer feedback classification,
business intelligence, sentiment analytics, and actionable insight generation.
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# NAVIGATION
# =====================================================
n1, n2, n3 = st.columns(3)

with n1:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)

    if st.button("Predict Feedback"):
        st.session_state.page = "predict"

    st.markdown('</div>', unsafe_allow_html=True)

with n2:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)

    if st.button("Bulk Analytics"):
        st.session_state.page = "bulk"

    st.markdown('</div>', unsafe_allow_html=True)

with n3:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)

    if st.button("Model Insights"):
        st.session_state.page = "insights"

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    "<div style='height:1rem'></div>",
    unsafe_allow_html=True
)

# =====================================================
# PREDICT PAGE
# =====================================================
if st.session_state.page == "predict":

    c1, c2 = st.columns([1.08, 0.92], gap="large")

    with c1:

        st.markdown("## Analyze Customer Feedback")

        st.markdown("""
        <div class="helper-text">
        Enter customer feedback to generate:
        • Feedback Classification
        • AI-Powered Insights
        </div>
        """, unsafe_allow_html=True)

        examples = [

            "The delivery was delayed and customer support was terrible.",

            "Please add dark mode and better search filters.",

            "Amazing experience and very fast service!",

            "The app crashes frequently during payments.",

            "Loved the interface and overall user experience."
        ]

        if st.button("Generate Example Feedback"):
            st.session_state.feedback_input = random.choice(examples)

        user_input = st.text_area(
            "",
            key="feedback_input",
            placeholder="Enter customer feedback here..."
        )

        predict_btn = st.button("Analyze Feedback")

    with c2:

        st.markdown("## Live AI Insights")

        if predict_btn:

            if not st.session_state.feedback_input.strip():

                st.warning(
                    "Please enter customer feedback before analyzing."
                )

            else:

                with st.spinner(
                    "Analyzing customer feedback..."
                ):

                    time.sleep(1.4)

                    vec = vectorizer.transform(
                        [st.session_state.feedback_input]
                    )

                    prediction = model.predict(vec)[0]

                    probs = model.predict_proba(vec)[0]

                prediction_colors = {
                    "Complaint": "#ff4d6d",
                    "Suggestion": "#8b5cf6",
                    "Praise": "#06b6d4"
                }

                st.markdown(
                    f"""
                    <div style='
                    padding:1.5rem;
                    border-radius:24px;
                    background:linear-gradient(
                        135deg,
                        {prediction_colors[prediction]},
                        #111827
                    );
                    color:white;
                    text-align:center;
                    font-size:1.5rem;
                    font-weight:700;
                    margin-bottom:1rem;
                    '>

                    {prediction}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                prob_df = pd.DataFrame({
                    "Category": model.classes_,
                    "Probability": probs
                })

                fig_prob = px.bar(
                    prob_df,
                    x="Category",
                    y="Probability",
                    text=prob_df["Probability"].round(2),
                    color="Category",
                    template="plotly_dark",
                    color_discrete_map={
                        "Complaint": "#ff4d6d",
                        "Suggestion": "#8b5cf6",
                        "Praise": "#06b6d4"
                    }
                )

                fig_prob.update_layout(

                    title=dict(
                        text="Live Prediction Probability Analysis",
                        x=0.5,
                        xanchor='center',
                        font=dict(
                            size=20,
                            color='white'
                        )
                    ),

                    paper_bgcolor='rgba(0,0,0,0)',

                    plot_bgcolor='rgba(15,23,42,0.35)',

                    font=dict(
                        color='white',
                        size=13
                    ),

                    margin=dict(
                        l=20,
                        r=20,
                        t=80,
                        b=20
                    ),

                    height=360
                )

                st.plotly_chart(
                    fig_prob,
                    use_container_width=True
                )

                # =========================================
                # BUSINESS RECOMMENDATIONS
                # =========================================
                if prediction == "Complaint":

                    recommendation_color = "#ff4d6d"

                    recommendation_title = (
                        "Critical Customer Recovery Action"
                    )

                    recommendation_text = """
                    • Escalate issue to support team<br><br>
                    • Investigate operational bottlenecks<br><br>
                    • Prioritize customer retention workflow<br><br>
                    • Trigger service recovery response<br><br>
                    • Monitor recurring complaint patterns
                    """

                elif prediction == "Suggestion":

                    recommendation_color = "#8b5cf6"

                    recommendation_title = (
                        "Product Improvement Opportunity"
                    )

                    recommendation_text = """
                    • Forward suggestions to product team<br><br>
                    • Evaluate feature enhancement feasibility<br><br>
                    • Analyze recurring user requests<br><br>
                    • Add insights to roadmap planning<br><br>
                    • Improve customer-centric innovation
                    """

                else:

                    recommendation_color = "#06b6d4"

                    recommendation_title = (
                        "Positive Customer Experience"
                    )

                    recommendation_text = """
                    • Encourage public reviews and ratings<br><br>
                    • Use feedback for testimonial campaigns<br><br>
                    • Identify loyalty reward opportunities<br><br>
                    • Promote customer advocacy programs<br><br>
                    • Reinforce high-performing services
                    """

                recommendation_html = f"""
                <div style='
                padding:1.4rem;
                border-radius:24px;
                background:linear-gradient(
                    135deg,
                    rgba(17,25,40,0.92),
                    rgba(15,23,42,0.92)
                );
                border-left:6px solid {recommendation_color};
                border:1px solid rgba(255,255,255,0.08);
                margin-top:1rem;
                line-height:1.9;
                box-shadow:0 10px 30px rgba(0,0,0,0.25);
                '>

                <div style='
                font-size:1.2rem;
                font-weight:700;
                margin-bottom:0.8rem;
                color:{recommendation_color};
                '>

                {recommendation_title}

                </div>

                <div style='
                color:#e2e8f0;
                font-size:0.96rem;
                '>

                {recommendation_text}

                </div>

                </div>
                """

                st.markdown(
                    recommendation_html,
                    unsafe_allow_html=True
                )

# =====================================================
# BULK ANALYTICS
# =====================================================
elif st.session_state.page == "bulk":

    st.markdown("## Bulk Customer Feedback Analytics")

    st.markdown("""
    <div style='
    padding:1.2rem;
    border-radius:20px;
    background:rgba(17,25,40,0.72);
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:1rem;
    line-height:1.9;
    '>

    <div style='
    font-size:1.1rem;
    font-weight:700;
    margin-bottom:0.6rem;
    '>
    Bulk Analytics Upload Guide
    </div>

    • Upload CSV files only<br>
    • Dataset must contain a column named:
    <b>feedback</b><br>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        with st.spinner(
            "Processing dataset..."
        ):

            time.sleep(2)

            bulk_df = pd.read_csv(
                uploaded_file,
                quotechar='"',
                skipinitialspace=True
            )

        if "feedback" not in bulk_df.columns:

            st.error(
                "CSV must contain a 'feedback' column."
            )

        else:

            bulk_vec = vectorizer.transform(
                bulk_df["feedback"]
            )

            bulk_df["Predicted_Label"] = model.predict(
                bulk_vec
            )

            # =========================================
            # KPI ROW
            # =========================================
            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Total Records",
                len(bulk_df)
            )

            m2.metric(
                "Complaints",
                (bulk_df["Predicted_Label"] == "Complaint").sum()
            )

            m3.metric(
                "Suggestions",
                (bulk_df["Predicted_Label"] == "Suggestion").sum()
            )

            m4.metric(
                "Praise",
                (bulk_df["Predicted_Label"] == "Praise").sum()
            )

            st.markdown(
                "<div style='height:1rem'></div>",
                unsafe_allow_html=True
            )

            # =========================================
            # CHARTS
            # =========================================
            g1, g2 = st.columns(2)

            with g1:

                fig_pie = px.pie(
                    bulk_df,
                    names="Predicted_Label",
                    hole=0.72,
                    template="plotly_dark",
                    color="Predicted_Label",
                    color_discrete_map={
                        "Complaint": "#ff4d6d",
                        "Suggestion": "#8b5cf6",
                        "Praise": "#06b6d4"
                    }
                )

                fig_pie.update_layout(

                    title=dict(
                        text="Customer Sentiment Distribution",
                        x=0.5,
                        xanchor='center',
                        font=dict(
                            size=20,
                            color='white'
                        )
                    ),

                    paper_bgcolor='rgba(0,0,0,0)',

                    plot_bgcolor='rgba(15,23,42,0.35)',

                    font=dict(
                        color='white',
                        size=13
                    ),

                    margin=dict(
                        l=20,
                        r=20,
                        t=80,
                        b=20
                    ),

                    height=360
                )

                st.plotly_chart(
                    fig_pie,
                    use_container_width=True
                )

            with g2:

                count_df = (
                    bulk_df["Predicted_Label"]
                    .value_counts()
                    .reset_index()
                )

                count_df.columns = [
                    "Category",
                    "Count"
                ]

                fig_bar = px.bar(
                    count_df,
                    x="Category",
                    y="Count",
                    text="Count",
                    color="Category",
                    template="plotly_dark",
                    color_discrete_map={
                        "Complaint": "#ff4d6d",
                        "Suggestion": "#8b5cf6",
                        "Praise": "#06b6d4"
                    }
                )

                fig_bar.update_layout(

                    title=dict(
                        text="Customer Feedback Category Analysis",
                        x=0.5,
                        xanchor='center',
                        font=dict(
                            size=20,
                            color='white'
                        )
                    ),

                    paper_bgcolor='rgba(0,0,0,0)',

                    plot_bgcolor='rgba(15,23,42,0.35)',

                    font=dict(
                        color='white',
                        size=13
                    ),

                    margin=dict(
                        l=20,
                        r=20,
                        t=80,
                        b=20
                    ),

                    height=360
                )

                st.plotly_chart(
                    fig_bar,
                    use_container_width=True
                )

            st.markdown(
                "<div style='height:1rem'></div>",
                unsafe_allow_html=True
            )

            # =========================================
            # TABLE
            # =========================================
            display_df = bulk_df.copy()

            display_df.columns = [
                col.replace("_", " ").title()
                for col in display_df.columns
            ]

            st.markdown(
                "### Processed Customer Feedback"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

# =====================================================
# MODEL INSIGHTS
# =====================================================
elif st.session_state.page == "insights":

    st.markdown("## Model Performance Analytics")

    X_eval = train_df["feedback"]
    y_eval = train_df["label"]

    X_eval_vec = vectorizer.transform(X_eval)

    y_pred_eval = model.predict(X_eval_vec)

    accuracy = (
        y_pred_eval == y_eval
    ).mean()

    report = classification_report(
        y_eval,
        y_pred_eval,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    # =========================================
    # ANIMATED KPI ROW
    # =========================================
    m1, m2, m3, m4 = st.columns(4)

    accuracy_percent = accuracy * 100
    training_samples = len(train_df)
    class_count = len(model.classes_)
    feature_count = len(
        vectorizer.get_feature_names_out()
    )

    with m1:

        acc_placeholder = st.empty()

        for i in np.linspace(
            0,
            accuracy_percent,
            35
        ):

            acc_placeholder.metric(
                "Accuracy",
                f"{i:.2f}%"
            )

            time.sleep(0.01)

    with m2:

        sample_placeholder = st.empty()

        for i in range(
            0,
            training_samples + 1,
            max(1, training_samples // 30)
        ):

            sample_placeholder.metric(
                "Training Samples",
                i
            )

            time.sleep(0.005)

        sample_placeholder.metric(
            "Training Samples",
            training_samples
        )

    with m3:

        class_placeholder = st.empty()

        for i in range(1, class_count + 1):

            class_placeholder.metric(
                "Classes",
                i
            )

            time.sleep(0.08)

    with m4:

        feature_placeholder = st.empty()

        animated_steps = np.linspace(
            0,
            feature_count,
            40
        )

        for i in animated_steps:

            feature_placeholder.metric(
                "TF-IDF Features",
                int(i)
            )

            time.sleep(0.008)

        feature_placeholder.metric(
            "TF-IDF Features",
            feature_count
        )

    st.markdown(
        "<div style='height:1.5rem'></div>",
        unsafe_allow_html=True
    )

    # =========================================
    # DISTRIBUTION + PIPELINE
    # =========================================
    a1, a2 = st.columns([0.78, 1.22], gap="large")

    with a1:

        st.markdown("### Overall Model Accuracy")

        acc_col1, acc_col2 = st.columns([0.88, 0.12])

        with acc_col1:

            st.progress(float(accuracy))

        with acc_col2:

            st.markdown(
                f"""
                <div style='
                display:flex;
                align-items:center;
                justify-content:center;
                height:28px;
                margin-top:-2px;
                '>

                <span style='
                font-size:1.55rem;
                font-weight:800;
                color:#60a5fa;
                line-height:1;
                white-space:nowrap;
                '>

                {accuracy*100:.2f}%

                </span>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style='
            padding:1rem;
            border-radius:18px;
            background:rgba(17,25,40,0.72);
            border:1px solid rgba(255,255,255,0.08);
            line-height:1.8;
            color:#e2e8f0;
            margin-top:1rem;
            '>

            <b>Pipeline Summary</b><br><br>

            • Text Preprocessing<br>
            • TF-IDF Vectorization<br>
            • Machine Learning Classification<br>
            • Probability-Based Prediction

            </div>
            """,
            unsafe_allow_html=True
        )

    with a2:

        dist_df = (
            train_df["label"]
            .value_counts()
            .reset_index()
        )

        dist_df.columns = [
            "Category",
            "Count"
        ]

        fig_dist = px.pie(
            dist_df,
            names="Category",
            values="Count",
            hole=0.72,
            template="plotly_dark",
            color="Category",
            color_discrete_map={
                "Complaint": "#ff4d6d",
                "Suggestion": "#8b5cf6",
                "Praise": "#06b6d4"
            }
        )

        fig_dist.update_layout(

            title=dict(
                text="Training Dataset Distribution",
                x=0.5,
                xanchor='center',
                font=dict(
                    size=20,
                    color='white'
                )
            ),

            paper_bgcolor='rgba(0,0,0,0)',

            plot_bgcolor='rgba(15,23,42,0.35)',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=80,
                b=20
            ),

            height=360
        )

        st.plotly_chart(
            fig_dist,
            use_container_width=True
        )

    st.markdown(
        "<div style='height:1.5rem'></div>",
        unsafe_allow_html=True
    )

    # =========================================
    # CONFUSION MATRIX + METRICS
    # =========================================
    g1, g2 = st.columns(2, gap="large")

    with g1:

        st.markdown("### Confusion Matrix")

        cm = confusion_matrix(
            y_eval,
            y_pred_eval,
            labels=model.classes_
        )

        fig_cm, ax = plt.subplots(
            figsize=(6,4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="mako",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            linewidths=1,
            linecolor='#111827',
            ax=ax
        )

        fig_cm.patch.set_facecolor('#050816')

        st.pyplot(fig_cm)

    with g2:

        st.markdown("### Classification Metrics")

        metrics_df = report_df.loc[
            ["Complaint", "Praise", "Suggestion"],
            ["precision", "recall", "f1-score"]
        ]

        metrics_df = metrics_df.reset_index().melt(
            id_vars="index",
            var_name="Metric",
            value_name="Score"
        )

        fig_metrics = px.bar(
            metrics_df,
            x="index",
            y="Score",
            color="Metric",
            barmode="group",
            template="plotly_dark",
            color_discrete_map={
                "precision": "#8b5cf6",
                "recall": "#06b6d4",
                "f1-score": "#ff4d6d"
            }
        )

        fig_metrics.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.35)',
            font=dict(color='white', size=13),
            margin=dict(l=20, r=20, t=40, b=20),
            height=360
        )

        st.plotly_chart(
            fig_metrics,
            use_container_width=True
        )

    st.markdown(
        "<div style='height:1.5rem'></div>",
        unsafe_allow_html=True
    )

    # =========================================
    # PERFORMANCE TREND
    # =========================================
    st.markdown("### Class-wise Performance Trend")

    perf_df = report_df.loc[
        ["Complaint", "Praise", "Suggestion"],
        ["precision", "recall", "f1-score"]
    ]

    perf_df = perf_df.reset_index()

    fig_line = px.line(
        perf_df.melt(
            id_vars="index",
            var_name="Metric",
            value_name="Value"
        ),
        x="index",
        y="Value",
        color="Metric",
        markers=True,
        template="plotly_dark"
    )

    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.35)',
        font=dict(color='white', size=13),
        margin=dict(l=20, r=20, t=40, b=20),
        height=360
    )

    st.plotly_chart(
        fig_line,
        use_container_width=True
    )

    st.markdown(
        "<div style='height:1.5rem'></div>",
        unsafe_allow_html=True
    )

    # =========================================
    # WORD CLOUD
    # =========================================
    st.markdown("### Customer Feedback Word Cloud")

    all_text = " ".join(
        train_df["feedback"]
        .astype(str)
        .tolist()
    )

    wordcloud = WordCloud(
        width=1200,
        height=500,
        background_color="#050816",
        colormap="cool",
        max_words=120,
        contour_width=0
    ).generate(all_text)

    fig_wc, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    fig_wc.patch.set_facecolor("#050816")

    st.pyplot(fig_wc)

    # =========================================
    # CLEAN REPORT TABLE
    # =========================================
    clean_report = report_df.copy()

    clean_report = clean_report.round(3)

    clean_report = clean_report.fillna("")

    clean_report.index = [
        str(idx).replace("-", " ").title()
        for idx in clean_report.index
    ]

    clean_report.columns = [
        str(col).replace("-", " ").title()
        for col in clean_report.columns
    ]

    st.markdown("### Detailed Performance Report")

    st.dataframe(
        clean_report,
        use_container_width=True,
        hide_index=False
    )
