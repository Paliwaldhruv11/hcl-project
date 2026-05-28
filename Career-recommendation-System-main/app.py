import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title="Career Recommendation", page_icon="🎯", layout="wide")

BUNDLE_PATH = Path("saved_models") / "career_streamlit_bundle.pkl"
EDU_PATH = Path("career_dataset_large.xlsx")
RIASEC_PATH = Path("career_data_extended.csv")
EDUCATION_BOOST_FACTOR = 5.0
EDUCATION_PRIORITY_FEATURES = [
    "Education Level",
    "Specialization",
    "Certifications",
    "CGPA/Percentage",
]


def parse_skills(value):
    if isinstance(value, list):
        candidates = value
    else:
        candidates = str(value).split(",")
    return [s.strip() for s in candidates if str(s).strip() and str(s).strip().lower() != "none"]


@st.cache_resource
def load_bundle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Bundle not found at '{path}'. Run Phase 10 in notebook first."
        )
    return joblib.load(path)


@st.cache_resource
def build_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_reference_data():
    edu_df = pd.read_excel(EDU_PATH)
    riasec_df = pd.read_csv(RIASEC_PATH)

    for col in ["Education Level", "Specialization", "Certifications", "Skills"]:
        if col in edu_df.columns:
            edu_df[col] = edu_df[col].fillna("None").astype(str).str.strip()

    stats = {
        "Math_Score": (float(riasec_df["Math_Score"].min()), float(riasec_df["Math_Score"].max()), float(riasec_df["Math_Score"].mean())),
        "Science_Score": (float(riasec_df["Science_Score"].min()), float(riasec_df["Science_Score"].max()), float(riasec_df["Science_Score"].mean())),
        "Programming_Skill": (float(riasec_df["Programming_Skill"].min()), float(riasec_df["Programming_Skill"].max()), float(riasec_df["Programming_Skill"].mean())),
        "Communication_Skill": (float(riasec_df["Communication_Skill"].min()), float(riasec_df["Communication_Skill"].max()), float(riasec_df["Communication_Skill"].mean())),
        "Logical_Ability": (float(riasec_df["Logical_Ability"].min()), float(riasec_df["Logical_Ability"].max()), float(riasec_df["Logical_Ability"].mean())),
        "R_score": (float(riasec_df["R_score"].min()), float(riasec_df["R_score"].max()), float(riasec_df["R_score"].mean())),
        "I_score": (float(riasec_df["I_score"].min()), float(riasec_df["I_score"].max()), float(riasec_df["I_score"].mean())),
        "A_score": (float(riasec_df["A_score"].min()), float(riasec_df["A_score"].max()), float(riasec_df["A_score"].mean())),
        "S_score": (float(riasec_df["S_score"].min()), float(riasec_df["S_score"].max()), float(riasec_df["S_score"].mean())),
        "E_score": (float(riasec_df["E_score"].min()), float(riasec_df["E_score"].max()), float(riasec_df["E_score"].mean())),
        "C_score": (float(riasec_df["C_score"].min()), float(riasec_df["C_score"].max()), float(riasec_df["C_score"].mean())),
        "CGPA/Percentage": (float(edu_df["CGPA/Percentage"].min()), float(edu_df["CGPA/Percentage"].max()), float(edu_df["CGPA/Percentage"].mean())),
    }

    options = {
        "Education Level": sorted(edu_df["Education Level"].unique().tolist()),
        "Specialization": sorted(edu_df["Specialization"].unique().tolist()),
        "Certifications": sorted(edu_df["Certifications"].unique().tolist()),
        "Skills": sorted(
            {
                skill
                for raw in edu_df["Skills"].tolist()
                for skill in parse_skills(raw)
            }
        ),
    }
    label_maps = {
        col: {val: idx for idx, val in enumerate(options[col])}
        for col in ["Education Level", "Specialization", "Certifications"]
    }
    return stats, options, label_maps


def build_feature_frame(
    math_score,
    science_score,
    programming_skill,
    communication_skill,
    logical_ability,
    r_score,
    i_score,
    a_score,
    s_score,
    e_score,
    c_score,
    education_level,
    specialization,
    certifications,
    cgpa,
    selected_skills,
    model,
    label_maps,
    mlb,
):
    feature_names = list(model.feature_names_in_)

    processed_input = pd.DataFrame(
        [
            {
                "Math_Score": math_score,
                "Science_Score": science_score,
                "Programming_Skill": programming_skill,
                "Communication_Skill": communication_skill,
                "Logical_Ability": logical_ability,
                "R_score": r_score,
                "I_score": i_score,
                "A_score": a_score,
                "S_score": s_score,
                "E_score": e_score,
                "C_score": c_score,
                "Education Level": education_level,
                "Specialization": specialization,
                "Certifications": certifications,
                "CGPA/Percentage": cgpa,
            }
        ]
    )

    for col in ["Education Level", "Specialization", "Certifications"]:
        val = str(processed_input[col].iloc[0])
        processed_input[col] = label_maps.get(col, {}).get(val, 0)

    if mlb is not None:
        skills_list = [str(s).strip() for s in selected_skills if str(s).strip()]
        skills_encoded = mlb.transform([skills_list]) if skills_list else np.zeros((1, len(mlb.classes_)))
        skills_df = pd.DataFrame(skills_encoded, columns=list(mlb.classes_), index=processed_input.index)
        final_input = pd.concat([processed_input, skills_df], axis=1).astype(float)
    else:
        final_input = processed_input.copy().astype(float)

    final_input = final_input.reindex(columns=feature_names).fillna(0).astype(float)
    for col in EDUCATION_PRIORITY_FEATURES:
        if col in final_input.columns:
            final_input[col] = final_input[col] * EDUCATION_BOOST_FACTOR
    return final_input


def randomize_form_state(stats, education_opts, specialization_opts, cert_opts, skill_opts):
    st.session_state.math_score = float(random.randint(int(stats["Math_Score"][0]), int(stats["Math_Score"][1])))
    st.session_state.science_score = float(random.randint(int(stats["Science_Score"][0]), int(stats["Science_Score"][1])))
    st.session_state.programming_skill = float(
        random.randint(int(stats["Programming_Skill"][0]), int(stats["Programming_Skill"][1]))
    )
    st.session_state.communication_skill = float(
        random.randint(int(stats["Communication_Skill"][0]), int(stats["Communication_Skill"][1]))
    )
    st.session_state.logical_ability = float(
        random.randint(int(stats["Logical_Ability"][0]), int(stats["Logical_Ability"][1]))
    )
    st.session_state.r_score = float(random.randint(int(stats["R_score"][0]), int(stats["R_score"][1])))
    st.session_state.i_score = float(random.randint(int(stats["I_score"][0]), int(stats["I_score"][1])))
    st.session_state.a_score = float(random.randint(int(stats["A_score"][0]), int(stats["A_score"][1])))
    st.session_state.s_score = float(random.randint(int(stats["S_score"][0]), int(stats["S_score"][1])))
    st.session_state.e_score = float(random.randint(int(stats["E_score"][0]), int(stats["E_score"][1])))
    st.session_state.c_score = float(random.randint(int(stats["C_score"][0]), int(stats["C_score"][1])))
    st.session_state.cgpa = round(random.uniform(stats["CGPA/Percentage"][0], stats["CGPA/Percentage"][1]), 1)
    if education_opts:
        st.session_state.education_level = random.choice(education_opts)
    if specialization_opts:
        st.session_state.specialization = random.choice(specialization_opts)
    if cert_opts:
        st.session_state.certifications = random.choice(cert_opts)
    if skill_opts:
        sample_size = min(len(skill_opts), random.randint(1, min(3, len(skill_opts))))
        st.session_state.selected_skills = random.sample(skill_opts, sample_size)


def main():
    st.title("Career Recommendation System")
    st.caption("One-page UI for your hybrid model (RIASEC + Education datasets) with SHAP explainability.")

    try:
        bundle = load_bundle(BUNDLE_PATH)
    except Exception as exc:
        st.error(str(exc))
        st.info("Run the notebook Phase 10 cell, then refresh this app.")
        return

    model = bundle.get("model")
    mlb = bundle.get("mlb")
    if model is None:
        st.error("Bundle is missing required key: 'model'.")
        return
    explainer = build_explainer(model)

    stats, ui_options, label_maps = load_reference_data()
    education_opts = ui_options.get("Education Level", [])
    specialization_opts = ui_options.get("Specialization", [])
    cert_opts = ui_options.get("Certifications", [])
    skill_opts = ui_options.get("Skills", [])

    # Initialize form state once.
    defaults = {
        "math_score": stats["Math_Score"][2],
        "science_score": stats["Science_Score"][2],
        "programming_skill": stats["Programming_Skill"][2],
        "communication_skill": stats["Communication_Skill"][2],
        "logical_ability": stats["Logical_Ability"][2],
        "r_score": stats["R_score"][2],
        "i_score": stats["I_score"][2],
        "a_score": stats["A_score"][2],
        "s_score": stats["S_score"][2],
        "e_score": stats["E_score"][2],
        "c_score": stats["C_score"][2],
        "education_level": education_opts[0] if education_opts else "Not specified",
        "specialization": specialization_opts[0] if specialization_opts else "Not specified",
        "certifications": cert_opts[0] if cert_opts else "None",
        "cgpa": stats["CGPA/Percentage"][2],
        "selected_skills": [skill_opts[0]] if skill_opts else [],
        "include_alt": True,
        "include_shap": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    with st.form("career_form"):
        st.subheader("Profile Input")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Psychometric & Ability")
            math_score = st.number_input("Math_Score", min_value=stats["Math_Score"][0], max_value=stats["Math_Score"][1], step=1.0, key="math_score")
            science_score = st.number_input("Science_Score", min_value=stats["Science_Score"][0], max_value=stats["Science_Score"][1], step=1.0, key="science_score")
            programming_skill = st.number_input(
                "Programming_Skill", min_value=stats["Programming_Skill"][0], max_value=stats["Programming_Skill"][1], step=1.0, key="programming_skill"
            )
            communication_skill = st.number_input(
                "Communication_Skill", min_value=stats["Communication_Skill"][0], max_value=stats["Communication_Skill"][1], step=1.0, key="communication_skill"
            )
            logical_ability = st.number_input(
                "Logical_Ability", min_value=stats["Logical_Ability"][0], max_value=stats["Logical_Ability"][1], step=1.0, key="logical_ability"
            )

        with c2:
            st.markdown("#### RIASEC Profile")
            r_score = st.number_input("R_score", min_value=stats["R_score"][0], max_value=stats["R_score"][1], step=1.0, key="r_score")
            i_score = st.number_input("I_score", min_value=stats["I_score"][0], max_value=stats["I_score"][1], step=1.0, key="i_score")
            a_score = st.number_input("A_score", min_value=stats["A_score"][0], max_value=stats["A_score"][1], step=1.0, key="a_score")
            s_score = st.number_input("S_score", min_value=stats["S_score"][0], max_value=stats["S_score"][1], step=1.0, key="s_score")
            e_score = st.number_input("E_score", min_value=stats["E_score"][0], max_value=stats["E_score"][1], step=1.0, key="e_score")
            c_score = st.number_input("C_score", min_value=stats["C_score"][0], max_value=stats["C_score"][1], step=1.0, key="c_score")

        with c3:
            st.markdown("#### Education & Skills")
            education_level = st.selectbox(
                "Education Level",
                options=education_opts if education_opts else ["Not specified"],
                key="education_level",
            )
            specialization = st.selectbox(
                "Specialization",
                options=specialization_opts if specialization_opts else ["Not specified"],
                key="specialization",
            )
            certifications = st.selectbox(
                "Certifications",
                options=cert_opts if cert_opts else ["None"],
                key="certifications",
            )
            cgpa = st.number_input(
                "CGPA/Percentage",
                min_value=stats["CGPA/Percentage"][0],
                max_value=stats["CGPA/Percentage"][1],
                step=0.1,
                key="cgpa",
            )
            selected_skills = st.multiselect("Skills", options=skill_opts, key="selected_skills")
            include_alt = st.checkbox("Show top 3 career matches", key="include_alt")
            include_shap = st.checkbox("Show SHAP explanation", key="include_shap")


        action_col1, action_col2 = st.columns(2)
        with action_col1:
            st.form_submit_button(
                "Random Input",
                type="secondary",
                use_container_width=True,
                on_click=randomize_form_state,
                args=(stats, education_opts, specialization_opts, cert_opts, skill_opts),
            )
        with action_col2:
            submitted = st.form_submit_button(
                "Predict Career",
                type="primary",
                use_container_width=True,
            )

    if not submitted:
        st.info("Fill the form and click 'Predict Career'.")
        return

    try:
        final_input = build_feature_frame(
            math_score=math_score,
            science_score=science_score,
            programming_skill=programming_skill,
            communication_skill=communication_skill,
            logical_ability=logical_ability,
            r_score=r_score,
            i_score=i_score,
            a_score=a_score,
            s_score=s_score,
            e_score=e_score,
            c_score=c_score,
            education_level=education_level,
            specialization=specialization,
            certifications=certifications,
            cgpa=cgpa,
            selected_skills=selected_skills,
            model=model,
            label_maps=label_maps,
            mlb=mlb,
        )

        probs = model.predict_proba(final_input)[0]
        classes = model.classes_
        best_indices = np.argsort(probs)[::-1]

        top_career = classes[best_indices[0]]
        suitability = float(probs[best_indices[0]] * 100)
        success_prob = float(min(max((suitability + (suitability - np.mean(probs) * 100)) / 2, 0), 100))

        # Lightweight input echo for user trust/debugging.
        with st.expander("View processed input sent to model", expanded=False):
            st.dataframe(final_input, width="stretch")
            st.caption(
                f"Education-priority boost applied: x{EDUCATION_BOOST_FACTOR:.1f} on "
                + ", ".join(EDUCATION_PRIORITY_FEATURES)
            )

        st.subheader("Enhanced Career Prediction Dashboard")
        m1, m2, m3 = st.columns(3)
        m1.metric("Primary Recommendation", str(top_career))
        m2.metric("Suitability Score", f"{suitability:.2f}%")
        m3.metric("Success Probability", f"{success_prob:.2f}%")

        if include_alt:
            st.markdown("### Alternative Career Matches")
            # Match notebook structure: show next 2 options after primary recommendation.
            top_n = min(2, max(len(best_indices) - 1, 0))
            alt_df = pd.DataFrame(
                {
                    "Career": [str(classes[best_indices[i]]) for i in range(1, top_n + 1)],
                    "Match %": [round(float(probs[best_indices[i]] * 100), 2) for i in range(1, top_n + 1)],
                }
            )
            if not alt_df.empty:
                st.dataframe(alt_df, width="stretch", hide_index=True)

        if include_shap:
            st.markdown("### SHAP Explanation")
            shap_vals = explainer.shap_values(final_input)
            if isinstance(shap_vals, list):
                this_class_shap = shap_vals[best_indices[0]][0]
                base_val = explainer.expected_value[best_indices[0]]
            else:
                this_class_shap = shap_vals[0, :, best_indices[0]]
                base_val = explainer.expected_value[best_indices[0]]

            feature_names = list(model.feature_names_in_)
            top_factors_idx = np.argsort(np.abs(this_class_shap))[-3:][::-1]
            top_factors_df = pd.DataFrame(
                {
                    "Feature": [feature_names[i] for i in top_factors_idx],
                    "SHAP Value": [float(this_class_shap[i]) for i in top_factors_idx],
                    "Influence": [
                        "Positive" if this_class_shap[i] > 0 else "Negative" for i in top_factors_idx
                    ],
                }
            )
            st.markdown("#### Top Influencing Factors")
            st.dataframe(top_factors_df, width="stretch", hide_index=True)

            target_bg_features = ["Education Level", "Specialization", "Certifications"]
            bg_indices = [feature_names.index(f) for f in target_bg_features if f in feature_names]
            if bg_indices:
                st.markdown("#### Impact of Background")
                bg_labels = [feature_names[i] for i in bg_indices]
                filtered_shap = this_class_shap[bg_indices]
                bg_df = pd.DataFrame({"Feature": bg_labels, "SHAP Value": [float(v) for v in filtered_shap]})
                bg_df = bg_df.sort_values("SHAP Value")

                fig_bg, ax_bg = plt.subplots(figsize=(10, 4))
                colors = ["forestgreen" if v > 0 else "crimson" for v in bg_df["SHAP Value"]]
                ax_bg.barh(bg_df["Feature"], bg_df["SHAP Value"], color=colors)
                ax_bg.set_title(f"Impact of Background on '{top_career}' Recommendation")
                ax_bg.axvline(x=0, color="black", linestyle="--")
                ax_bg.set_xlabel("Contribution Strength (SHAP value)")
                st.pyplot(fig_bg, clear_figure=True)

            # Force plot rendered as matplotlib figure for Streamlit.
            # Removed to keep SHAP section focused on requested table + background graph only.

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.code(
            "Tip: Re-run the training notebook and Phase 10 export cell to keep bundle/model consistent."
        )

if __name__ == "__main__":
    main()
