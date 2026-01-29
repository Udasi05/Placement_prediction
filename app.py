# 🎓 Student Placement Prediction Dashboard


# Step 2: Import libraries
import pandas as pd 
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 3: Load dataset
file_path = "placementdata.csv"
df = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# Step 4: Basic cleanup
df = df.dropna()

#  Drop non-predictive ID columns automatically
for col in df.columns:
    if 'id' in col.lower():
        print(f" Dropping column: {col}")
        df = df.drop(columns=[col])

# Step 5: Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 6: Define features & target
target_col = "PlacementStatus"
X = df.drop(columns=[target_col])
y = df[target_col]

# Step 7: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {acc*100:.2f}%")


# 📊 Placement Statistics
total_students = len(df)
placed_students = df[df[target_col] == 1].shape[0]
not_placed_students = df[df[target_col] == 0].shape[0]
placement_rate = (placed_students / total_students) * 100

def get_pie_plot():
    labels = ['Placed', 'Not Placed']
    values = [placed_students, not_placed_students]
    plt.figure(figsize=(3,3))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Placement Distribution")
    plt.tight_layout()
    plt.savefig("/content/placement_pie.png")
    plt.close()
    return "/content/placement_pie.png"

# Extra chart: correlation between one numeric column & placement (if exists)
def get_feature_plot():
    numeric_cols = df.select_dtypes(include='number').columns.drop(target_col)
    if len(numeric_cols) == 0:
        return None
    col = numeric_cols[0]
    plt.figure(figsize=(4,3))
    df.groupby(target_col)[col].mean().plot(kind='bar', color=['red','green'])
    plt.title(f"Average {col} vs Placement")
    plt.xlabel("Placement (0=Not Placed, 1=Placed)")
    plt.ylabel(f"Average {col}")
    plt.tight_layout()
    plt.savefig("/content/feature_bar.png")
    plt.close()
    return "/content/feature_bar.png"

# =========================
# 🔮 Prediction Function
# =========================
def predict_placement(*inputs):
    data = pd.DataFrame([inputs], columns=X.columns)
    for col in data.columns:
        if col in label_encoders:
            le = label_encoders[col]
            data[col] = le.transform([data[col][0]]) if data[col][0] in le.classes_ else -1
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    return "🎯 The student WILL get Placed ✅" if pred == 1 else "❌ The student will NOT get Placed"

# =========================
# 🎨 Build Gradio Interface
# =========================
inputs = []
for col in X.columns:
    if col in label_encoders:
        inputs.append(gr.Dropdown(choices=list(label_encoders[col].classes_), label=col))
    else:
        inputs.append(gr.Number(label=col))

# Tab 1: Prediction
predict_tab = gr.Interface(
    fn=predict_placement,
    inputs=inputs,
    outputs="text",
    title="🎓 Student Placement Prediction",
    description=f"Enter student details to predict placement.\nModel Accuracy: {acc*100:.2f}%"
)

# Tab 2: Stats
def show_stats():
    pie_path = get_pie_plot()
    feat_path = get_feature_plot()
    stats_text = f"""
    ### 📊 Placement Statistics
    - Total Students: {total_students}
    - Placed: {placed_students}
    - Not Placed: {not_placed_students}
    - Placement Rate: {placement_rate:.2f}%
    - Model Accuracy: {acc*100:.2f}%

    Below are placement-related charts:
    """
    return stats_text, pie_path, feat_path

stats_tab = gr.Interface(
    fn=show_stats,
    inputs=None,
    outputs=[gr.Markdown(), gr.Image(), gr.Image()],
    title="📈 Placement Statistics"
)

# Tab 3: Info
info_text = f"""
### ℹ About This Project
This ML mini-project predicts whether a student will get placed based on academic and personal attributes.

#### 🧠 Model Info
- Algorithm: Random Forest Classifier
- Accuracy: {acc*100:.2f}%

#### 💡 How It Works
1. Data is preprocessed (missing values removed, IDs dropped, categorical values encoded).
2. Model trained → new student data tested.
3. Predicts *Placed / Not Placed* outcome.
4. Stats and graphs shown for better interpretation.

#### 🧰 Tech Stack
- Python (Pandas, Scikit-learn)
- Gradio (Interactive UI)
- Matplotlib (Charts)
"""

info_tab = gr.Interface(fn=lambda: info_text, inputs=None, outputs=gr.Markdown(), title="ℹ Information")

# Combine all tabs
dashboard = gr.TabbedInterface(
    [predict_tab, stats_tab, info_tab],
    tab_names=["🔮 Predict", "📊 Stats", "ℹ Info"]
)

# Step 10: Launch app
dashboard.launch(share=True)