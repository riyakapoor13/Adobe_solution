# agents/classification_agent.py
import os
import lightgbm as lgb
import pandas as pd

class HeadingClassificationAgent:
    """
    Classifies blocks as Title, H1, H2, H3, or Other.
    """
    def __init__(self, model_path="model.txt"):
        self.model = None
        if os.path.exists(model_path):
            print("ML model found. Loading...")
            self.model = lgb.Booster(model_file=model_path)
            # IMPORTANT: This map must match the output of your train_model.py script
            self.class_map = {0: "H1", 1: "H2", 2: "H3", 3: "Other", 4: "Title"}
        else:
            print("ML model not found. Falling back to rules-based classification.")

    def predict(self, featured_blocks: list) -> list:
        if not featured_blocks:
            return []
        
        if self.model:
            return self._predict_with_ml(featured_blocks)
        else:
            return self._predict_with_rules(featured_blocks)

    def _predict_with_ml(self, featured_blocks: list) -> list:
        df = pd.DataFrame(featured_blocks)
        features_to_use = ['font_size', 'relative_font_size', 'is_bold', 'is_centered', 'word_count', 'starts_with_numbering']
        
        df = pd.get_dummies(df, columns=['text_case'], drop_first=True)
        expected_cols = features_to_use + ['text_case_UPPER', 'text_case_Title']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = False

        predictions = self.model.predict(df[features_to_use], num_iteration=self.model.best_iteration)
        predicted_class_indices = predictions.argmax(axis=1)

        for i, block in enumerate(featured_blocks):
            block['level'] = self.class_map.get(predicted_class_indices[i], "Other")
        return featured_blocks

    def _predict_with_rules(self, featured_blocks: list) -> list:
        for block in featured_blocks:
            level = "Other"
            rel_size = block['relative_font_size']
            
            if rel_size > 1.8 and block['is_centered']:
                level = "Title"
            elif rel_size > 1.4 and block['is_bold']:
                level = "H1"
            elif rel_size > 1.2 and (block['is_bold'] or block['starts_with_numbering']):
                level = "H2"
            elif rel_size > 1.0 and (block['is_bold'] or block['starts_with_numbering']):
                level = "H3"
            
            block['level'] = level
        return featured_blocks