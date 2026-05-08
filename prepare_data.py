# prepare_data.py

import pandas as pd

# LOAD DATASET
df = pd.read_csv("sentiment-analysis.csv", sep=",", engine="python")

# FIX CSV IF NEEDED
if len(df.columns) == 1:
    df = df[df.columns[0]].str.split(",", expand=True)

    df.columns = [
        "Text",
        "Sentiment",
        "Source",
        "DateTime",
        "UserID",
        "Location",
        "ConfidenceScore"
    ]

# KEEP REQUIRED COLUMNS
df = df[["Text", "Sentiment"]]

# CLEAN DATA
df["Text"] = df["Text"].astype(str).str.replace('"', '').str.strip()

df["Sentiment"] = df["Sentiment"].astype(str).str.strip()

# MAP LABELS
label_map = {
    "Positive": "Praise",
    "Negative": "Complaint"
}

df["label"] = df["Sentiment"].map(label_map)

base_df = df[["Text", "label"]]

base_df.columns = ["feedback", "label"]

# ADD SUGGESTION DATA
suggestion_texts = [
    "Please add dark mode",
    "Would love better battery life",
    "Could you improve loading speed",
    "Add more payment options",
    "Please include export to PDF",
    "Need more customization settings",
    "Would be better with offline mode",
    "Wish there was a search feature",
    "Please add multi-language support",
    "Need better navigation in the app",
    "Please improve the user interface",
    "Could you add fingerprint login",
    "Need advanced search filters",
    "Would like more personalization options",
    "Please improve customer support timing",
    "Could use better documentation",
    "Need a tutorial for beginners",
    "Please add notification settings",
    "Would be helpful to have analytics export",
    "Please add chat support",
    "Could improve app responsiveness",
    "Need better dashboard customization",
    "Would love theme options",
    "Please support voice commands",
    "Need more payment gateway options",
    "Please add file upload support",
    "Would like a mobile version",
    "Need integration with Excel",
    "Could improve performance speed",
    "Please add auto-save feature",
    "Would like keyboard shortcuts",
    "Need better filter functionality",
    "Please add sorting options",
    "Could improve menu organization",
    "Would love offline access",
    "Need more report templates",
    "Please improve navigation layout",
    "Could add API access",
    "Need better onboarding process",
    "Would like dark/light theme toggle"
]

suggestion_df = pd.DataFrame({
    "feedback": suggestion_texts,
    "label": "Suggestion"
})

# MERGE DATA
final_df = pd.concat([base_df, suggestion_df], ignore_index=True)

# SHUFFLE
final_df = final_df.sample(frac=1, random_state=42)

# SAVE
final_df.to_csv("final_feedback_dataset.csv", index=False)

print(final_df.head())

print("\nDataset Created Successfully!")
print(final_df["label"].value_counts())