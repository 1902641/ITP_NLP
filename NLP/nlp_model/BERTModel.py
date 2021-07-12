from NLP.nlp_model.NLPModel import NLPModel
# For ML
import numpy as np
import re
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf

import datetime
from datetime import datetime

# BERT
import bert.optimization as optimization
import bert.run_classifier as run_classifier
import bert.tokenization as tokenization
import tensorflow_hub as hub


class CustomLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


class BERTModel(NLPModel):
    def __init__(self):
        """
        Output directory for checkpoints default to ./bert_model
        Use func set_output_model_directory() to load/save different models
        """
        self.DATA_COLUMN = 'text'
        self.LABEL_COLUMN = 'label'
        # Run the below line to download for the first run
        nltk.download('stopwords')
        self.bert_model_to_load_from_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        # self.OUTPUT_DIR = './bert_model'
        self.OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'bert_model'))
        self.SAVE_DIR = './saved_model'
        self.LOAD_DIR = './saved_model/1623831954'
        self.full_dataframe = pd.DataFrame()
        self.dataframe = pd.DataFrame()
        self.numeric_label_list = []
        self.label_list = []
        self.estimator = tf.estimator.Estimator
        self.MAX_SEQ_LENGTH = 100
        self.tokenizer = self.create_tokenizer()
        self.label_encoder = CustomLabelEncoder()
        self.trained_length = 12

    def load_label_list(self, in_label_list: list):
        self.label_list = in_label_list
        self.trained_length = len(self.label_list)
        self.label_encoder.fit(self.label_list)
        self.numeric_label_list = self.label_encoder.transform(self.label_list)

    def set_output_model_directory(self, in_directory: str):
        """
        Set this to custom model directory. Use it if you have other models to load/save
        """
        self.OUTPUT_DIR = in_directory

    def load_model(self):
        """
        Model must be loaded via an existing model directory prior to calling this function
        """

        BATCH_SIZE = 6
        LEARNING_RATE = 2e-5
        NUM_TRAIN_EPOCHS = 1.0
        WARMUP_PROPORTION = 0.1
        SAVE_CHECKPOINTS_STEPS = 200
        SAVE_SUMMARY_STEPS = 80

        num_train_steps = 1
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        run_config = tf.estimator.RunConfig(model_dir=self.OUTPUT_DIR, save_summary_steps=SAVE_SUMMARY_STEPS,
                                            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS, session_config=sess_config)

        model_function = self.model_function_builder(num_labels=self.trained_length,
                                                     learning_rate=LEARNING_RATE,
                                                     num_train_steps=num_train_steps,
                                                     num_warmup_steps=num_warmup_steps)
        self.estimator = tf.estimator.Estimator(model_fn=model_function,
                                                config=run_config,
                                                params={"batch_size": BATCH_SIZE})

    def fit(self, dataframe: pd.DataFrame, in_labels: list):
        """
        Insert training dataframe to model
        """
        self.full_dataframe = dataframe
        self.dataframe = self.full_dataframe.filter(['file', 'text', 'label'], axis=1)
        # self.numeric_label_list = [x for x in np.unique(self.dataframe.label)]
        # Convert labels to be numeric for the model to classify
        train_label_list = [x for x in np.unique(self.dataframe.label)]
        self.label_list = in_labels
        for label in train_label_list:
            if label not in self.label_list:
                self.label_list.append(label)
        self.label_encoder.fit(self.label_list)
        self.trained_length = len(self.label_list)
        test_labels = [x for x in np.unique(self.dataframe.label)]
        print('test labels: ...')
        print(test_labels)
        self.numeric_label_list = self.label_encoder.transform(self.label_list)
        self.dataframe['label'] = self.label_encoder.transform(self.dataframe['label'])
        test_labels = [x for x in np.unique(self.dataframe.label)]
        print('test labels: ...')
        print(test_labels)
        # Preprocess the text by cleaning the text fit for reading by model
        self.dataframe['text'] = self.dataframe['text'].apply(self.clean_text)
        self.dataframe['text'] = self.dataframe['text'].str.replace('\d+', '')
        self.dataframe['text_split'] = self.dataframe['text'].apply(self.split_text)

    def train(self):
        # Split into training and validation
        train_set, validation_set = train_test_split(self.dataframe, test_size=0.1, random_state=35)
        train_set.reset_index(drop=True, inplace=True)

        label_list = [x for x in np.unique(train_set.label)]
        print("Self label list: ")
        print(self.label_list)
        print("Encoded label list: ")
        print(self.numeric_label_list)

        # self.numeric_label_list = label_list

        print("training label list: ")
        print(label_list)
        val_lbls = [x for x in np.unique(validation_set.label)]
        validation_set.reset_index(drop=True, inplace=True)

        train_list = []
        train_label_list = []
        train_index_list = []
        train_filename_list = []
        for index, row in train_set.iterrows():
            for split_part in row['text_split']:
                train_list.append(split_part)
                train_label_list.append(row['label'])
                train_index_list.append(index)
                train_filename_list.append(row['file'])

        validation_list = []
        validation_label_list = []
        validation_index_list = []
        validation_filename_list = []
        for index, row in validation_set.iterrows():
            for split_part in row['text_split']:
                validation_list.append(split_part)
                validation_label_list.append(row['label'])
                validation_index_list.append(index)
                validation_filename_list.append(row['file'])

        # # Convert training set and validation set into Dataframes
        # train_dataframe = pd.DataFrame({self.DATA_COLUMN: train_list,
        #                                 self.LABEL_COLUMN: train_label_list})
        #
        # validation_dataframe = pd.DataFrame({self.DATA_COLUMN: validation_list,
        #                                      self.LABEL_COLUMN: validation_label_list})
        # Prepare input data
        train_input = train_set.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                            text_a=x[self.DATA_COLUMN],
                                                                            text_b=None,
                                                                            label=x[self.LABEL_COLUMN]), axis=1)
        validation_input = validation_set.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                                      text_a=x[self.DATA_COLUMN],
                                                                                      text_b=None,
                                                                                      label=x[self.LABEL_COLUMN]), axis=1)
        train_features = run_classifier.convert_examples_to_features(train_input, label_list, self.MAX_SEQ_LENGTH, self.tokenizer)
        validation_features = run_classifier.convert_examples_to_features(validation_input, label_list, self.MAX_SEQ_LENGTH,
                                                                          self.tokenizer)

        BATCH_SIZE = 6
        LEARNING_RATE = 2e-5
        NUM_TRAIN_EPOCHS = 1.0
        WARMUP_PROPORTION = 0.1
        SAVE_CHECKPOINTS_STEPS = 200
        SAVE_SUMMARY_STEPS = 80

        num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        run_config = tf.estimator.RunConfig(model_dir=self.OUTPUT_DIR, save_summary_steps=SAVE_SUMMARY_STEPS,
                                            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS, session_config=sess_config)

        model_function = self.model_function_builder(num_labels=self.trained_length,
                                                     learning_rate=LEARNING_RATE,
                                                     num_train_steps=num_train_steps,
                                                     num_warmup_steps=num_warmup_steps)
        estimator = tf.estimator.Estimator(model_fn=model_function,
                                           config=run_config,
                                           params={"batch_size": BATCH_SIZE})

        train_input_fn = run_classifier.input_fn_builder(features=train_features,
                                                         seq_length=self.MAX_SEQ_LENGTH,
                                                         is_training=True,
                                                         drop_remainder=False)
        validation_input_fn = run_classifier.input_fn_builder(features=validation_features,
                                                              seq_length=self.MAX_SEQ_LENGTH,
                                                              is_training=False,
                                                              drop_remainder=False)
        # Training the model
        print("Beginning to train the model...")
        current_time = datetime.now()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Total training time : ", datetime.now() - current_time)

        print("Evaluating the model with validation set")
        result = estimator.evaluate(input_fn=validation_input_fn, steps=None)
        print("Results")
        print(result)
        # self.estimator = estimator
        # self.save_model()

    def clean_text(self, text):
        """
        :param text: targeted text to be cleaned
        :return: cleaned text string
        """
        SPACE_REGEX = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_REGEX = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))
        text = text.lower()
        text = SPACE_REGEX.sub(' ', text)
        text = BAD_SYMBOLS_REGEX.sub('', text)
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)
        return text

    def split_text(self, text):
        """
        :param text: targeted text to be split
        :return: text split
        """
        l_total = []
        l_parcial = []
        if len(text.split()) // 150 > 0:
            n = len(text.split()) // 150
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = text.split()[:200]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = text.split()[w * 150:w * 150 + 200]
                l_total.append(" ".join(l_parcial))
        return l_total

    def output_model_directory(self, directory):
        """
        :param directory: location to save the bert model to
        """

        DELETE_FLAG = True

        if DELETE_FLAG:
            try:
                # tf.gfile.DeleteRecursively(directory)
                tf.io.gfile.rmtree(directory)
            except:
                print("Unable to delete")

        # tf.gfile.MakeDirs(directory)
        tf.io.gfile.makedirs(directory)
        print('***** Model output directory: {} *****'.format(directory))

    def create_tokenizer(self):
        # Retrieve necessary info from hub e.g. vocab file for tokenizer etc
        with tf.Graph().as_default():
            bert_module = hub.Module(self.bert_model_to_load_from_hub)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.compat.v1.Session() as session:
                vocab_file, do_lower_case = session.run([tokenization_info["vocab_file"],
                                                         tokenization_info["do_lower_case"]])
            return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
        bert_module = hub.Module(self.bert_model_to_load_from_hub, trainable=True)
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids= segment_ids)
        bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

        output_layer = bert_outputs["pooled_output"]
        output_layer_predict = bert_outputs["pooled_output"]
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.compat.v1.get_variable("output_weights", [num_labels, hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.compat.v1.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.compat.v1.variable_scope("loss"):
            output_layer = tf.nn.dropout(output_layer, rate=1-0.8)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))

            # If predicting
            if is_predicting:
                return predicted_labels, log_probs, output_layer_predict

            # If training
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return loss, predicted_labels, log_probs

    def model_function_builder(self, num_labels, learning_rate, num_train_steps, num_warmup_steps):
        """ Returns 'model_function' closure for TPUEstimator """
        def model_function(features, labels, mode, params):
            """ The 'module_function' for TPUEstimator """
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # If training
            if not is_predicting:
                loss, predicted_labels, log_probs = self.create_model(is_predicting, input_ids,
                                                                       input_mask, segment_ids,
                                                                       label_ids, num_labels)
                train_optimizer = optimization.create_optimizer(loss, learning_rate,
                                                                num_train_steps, num_warmup_steps, use_tpu=False)

                # Calculate evaluation metrics
                def metric_function(label_ids, predicted_labels):
                    accuracy = tf.compat.v1.metrics.accuracy(label_ids, predicted_labels)
                    true_positives = tf.compat.v1.metrics.true_positives(label_ids, predicted_labels)
                    true_negatives = tf.compat.v1.metrics.true_negatives(label_ids, predicted_labels)
                    false_positives = tf.compat.v1.metrics.false_positives(label_ids, predicted_labels)
                    false_negatives = tf.compat.v1.metrics.false_negatives(label_ids, predicted_labels)

                    return {"eval_accuracy": accuracy, "true_positives": true_positives,
                            "true_negatives": true_negatives, "false_positives": false_positives,
                            "false_negatives": false_negatives
                            }
                eval_metrics = metric_function(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_optimizer)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
            else:
                predicted_labels, log_probs, output_layer = self.create_model(is_predicting, input_ids, input_mask,
                                                                              segment_ids, label_ids, num_labels)
                predictions = {"probabilities": log_probs,
                               "labels": predicted_labels,
                               "pooled_output": output_layer
                               }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        return model_function

    def predict(self,in_sentences, single_prediction=False):
        # A list to map the actual labels to the predictions
        labels = self.numeric_label_list
        # train_input = train_set.apply(lambda x: run_classifier.InputExample(guid=None,
        #                                                                     text_a=x[self.DATA_COLUMN],
        #                                                                     text_b=None,
        #                                                                     label=x[self.LABEL_COLUMN]), axis=1)
        if single_prediction:
            input_examples = [run_classifier.InputExample(guid="", text_a=in_sentences, text_b=None, label=0)]
        else:
            input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in in_sentences]

        input_features = run_classifier.convert_examples_to_features(input_examples, self.numeric_label_list, self.MAX_SEQ_LENGTH,
                                                                     self.tokenizer)
        # Predicting the classes
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=self.MAX_SEQ_LENGTH,
                                                           is_training=False, drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn, yield_single_examples=not single_prediction)
        if single_prediction:
            prediction = next(predictions)
            return [in_sentences, [f'{x*100:0.2f}%' for x in np.exp(prediction['probabilities'])[0]], prediction['labels'], self.label_list[prediction['labels']], np.exp(prediction['probabilities']).sum()]
        else:
            return ([(sentence, [f'{x*100:0.2f}%' for x in np.exp(prediction['probabilities'])],
                      prediction['labels'], self.label_list[prediction['labels']]) for sentence, prediction in
                     zip(in_sentences, predictions)])

    def save_model(self):
        def serving_input_fn():
            label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
            input_ids = tf.placeholder(tf.int32, [None, self.MAX_SEQ_LENGTH], name='input_ids')
            input_mask = tf.placeholder(tf.int32, [None, self.MAX_SEQ_LENGTH], name='input_mask')
            segment_ids = tf.placeholder(tf.int32, [None, self.MAX_SEQ_LENGTH], name='segment_ids')
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
                {
                    'label_ids': label_ids,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids
                }
            )()
            return input_fn
        self.estimator._export_to_tpu = False
        self.estimator.export_saved_model(export_dir_base=self.SAVE_DIR, serving_input_receiver_fn=serving_input_fn)
