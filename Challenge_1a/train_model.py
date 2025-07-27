# train_model.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load your labeled data
try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: train.csv not found.")
    print("Please run 'python create_training_data.py' first to generate it.")
    exit()

# 2. Prepare data for training
# Convert text labels (H1, H2, etc.) into numbers the model can use
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Feature engineering (must match the features used in solution.py/main.py)
df['relative_font_size'] = df['font_size'] / df['font_size'].median()
df = pd.get_dummies(df, columns=['text_case'], drop_first=True)

# 3. Define features and target
features = [
    'font_size', 'relative_font_size', 'is_bold',
    'is_centered', 'word_count', 'starts_with_numbering'
]
# Add the new one-hot encoded columns to the feature list
if 'text_case_UPPER' in df.columns: features.append('text_case_UPPER')
if 'text_case_Title' in df.columns: features.append('text_case_Title')

# Ensure all feature columns exist, fill missing with False (0)
for col in features:
    if col not in df.columns:
        df[col] = False

target = 'label_encoded'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# 4. Train the LightGBM model
params = {
    'objective': 'multiclass',
    'num_class': len(df['label'].unique()),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 100
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# 5. Save the trained model
model.booster_.save_model('model.txt')

print("\n✅ Model trained successfully and saved as model.txt")
print("\nIMPORTANT: The model expects the following class order:")
# Print the mapping of number to label so you can verify it in classification_agent.py
class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
print(class_mapping)