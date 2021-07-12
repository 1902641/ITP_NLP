import pandas as pd

class NLPModel:
    def load_label_list(self, in_label_list: list):
        pass

    def set_output_model_directory(self, in_directory: str):
        pass

    def load_model(self):
        pass

    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
        pass

    def create_tokenizer(self):
        pass

    def fit(self, dataframe: pd.DataFrame, labels: list):
        pass

    def train(self):
        pass

    def predict(self,in_sentences, single_prediction=False):
        pass
