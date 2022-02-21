import json
from collections import defaultdict
import numpy as np
import pandas as pd
import itertools
import math

'''
This file creates feedback files for each worker in a batch, in comparison to the expert batch. 
'''

QUESTIONS_DICT = {1: "What is the [PROPERTY] of (the) [W]?", 2: "Whose [W]?", 3: "Where is the [W]?",
                  4: "How much /How many [W]?",
                  5: "What is the [W] a part/member of?", 6: "What/Who is a part/member of [W]?",
                  7: "What/Who is (the) [W]?",
                  8: "What kind of [W]?", 9: "When is the [W]?"}


class FeedbackCalculator:
    def __init__(self, results_file, expert_file, worker_score_filename, question_disagreement_template, answer_csv_template):
        self.batch_results_file = results_file
        self.results_df = pd.read_csv(self.batch_results_file)
        self.expert_results_file = expert_file
        self.expert_df = pd.read_csv(self.expert_results_file)
        self.dict_by_worker = self.create_dictionary_by_worker()
        self.expert_dict = self.create_expert_dictionary_by_instance()
        self.worker_score_filename = worker_score_filename
        self.answer_csv_template = answer_csv_template
        self.question_disagreement_template = question_disagreement_template

    def create_dictionary_by_worker(self):
        workers_dict = dict()
        workers = self.results_df["WorkerId"].unique()
        for worker_id in workers:
            workers_dict[worker_id] = self.create_answers_dict_for_each_worker(worker_id)
        return workers_dict

    def create_expert_dictionary_by_instance(self):
        expert_dict = dict()
        sentence_ids = self.expert_df["Input.sentenceId"].unique()
        for sentence_id in sentence_ids:
            sentence_id_df = self.expert_df[self.expert_df["Input.sentenceId"] == sentence_id]
            word_indices = sentence_id_df["Input.index"].unique()
            for word_index in word_indices:
                instance_df = sentence_id_df[sentence_id_df["Input.index"] == word_index]
                instance_key = self.create_instance_key(sentence_id, word_index)
                expert_dict[instance_key] = self.create_answers_dict_for_each_instance(instance_df)
        return expert_dict

    @staticmethod
    def create_instance_key(sentence_id, index):
        return str(sentence_id) + ':' + str(index)

    def create_answers_dict_for_each_instance(self, instance_df):
        answers_dict = dict()
        for index, row in instance_df.iterrows():
            answers_dict[row["WorkerId"]] = self.parser_json_answer(row["Answer.taskAnswers"], row["Input.sentence"],
                                                            row["Input.index"], row["Input.sentenceId"])
        return answers_dict

    def create_answers_dict_for_each_worker(self, worker_id):
        answers_dict = dict()
        filtered_df = self.results_df[self.results_df["WorkerId"] == worker_id]
        df_for_hit = filtered_df[["HITId", "Input.sentenceId", "Answer.taskAnswers", "Input.sentence", "Input.index"]]
        for index, row in df_for_hit.iterrows():
            instance_key = self.create_instance_key(row["Input.sentenceId"], row["Input.index"])
            answers_dict[instance_key] = self.parser_json_answer(row["Answer.taskAnswers"], row["Input.sentence"],
                                                            row["Input.index"], row["Input.sentenceId"])
        return answers_dict

    @staticmethod
    def parser_json_answer(json_answer, sentence, index, sentenceid):
        json_answer_loaded = json.loads(json_answer)[0]
        questions_answers_dict = {"questions": dict(), "answers": dict(), "properties": dict(),
                                  "sentenceid": sentenceid, "sentence": sentence, "index": index,
                                  "start_indices": dict(), "end_indices": dict()}
        question_id_max = len(json_answer_loaded)
        for i in range(question_id_max):
            if "question-" + str(i) in json_answer_loaded:
                questions_answers_dict["questions"][i] = json_answer_loaded["question-" + str(i)]
                questions_answers_dict["answers"][i] = json_answer_loaded["answers-" + str(i)]
                questions_answers_dict["start_indices"][i] = int(json_answer_loaded["start-" + str(i)])
                questions_answers_dict["end_indices"][i] = int(json_answer_loaded["end-" + str(i)])
            if "question-" + str(i) + "-property-input" in json_answer_loaded:
                questions_answers_dict["properties"][i] = json_answer_loaded["question-" + str(i) + "-property-input"]
        return questions_answers_dict

    def calculate_answer_accuracy_for_one_worker_for_one_instance(self, worker_id, curr_worker_dict, instance_key):
        # recall = tp / (fn + tp)
        # precision = tp / (tp + fp)
        # f1 = (2 * tp) / ((2 * tp) + fp + fn)
        num_matching_answers = 0
        predicted = curr_worker_dict[instance_key]
        golds = self.expert_dict[instance_key]
        predicted_answers = predicted["answers"]
        num_golds_answers = []
        answer_id_matches_with_gold = {gold_worker_id: [] for gold_worker_id in golds}
        for predicted_answer_id in predicted_answers:
            found_gold_match = False
            for gold_option in golds:
                gold_answers = golds[gold_option]["answers"]
                for gold_answer_id in gold_answers:
                    if self.evaluate_answer_match_rate(gold_answers[gold_answer_id],
                                                       predicted_answers[predicted_answer_id]) >= 0.5:
                        found_gold_match = True
                        answer_id_matches_with_gold[gold_option].append(gold_answer_id)
                        self.add_row_to_answer_csv(worker_id, predicted, predicted_answer_id, predicted_answers,
                                                   gold_option, golds[gold_option], gold_answer_id, gold_answers)
            if found_gold_match:
                num_matching_answers += 1
            else:
                self.add_row_to_answer_csv(worker_id, predicted, predicted_answer_id, predicted_answers, '', '', '', '')
        self.add_recall_rows(answer_id_matches_with_gold, golds, worker_id)
        for gold_option in golds:
            gold_answers = golds[gold_option]["answers"]
            num_golds_answers.append(len(gold_answers))
        try:
            avg_len_gold_answers = int(math.ceil(sum(num_golds_answers) / len(num_golds_answers)))
        except:
            avg_len_gold_answers = 0
        tp = num_matching_answers
        fp = (len(predicted_answers) - num_matching_answers)
        fn = avg_len_gold_answers - num_matching_answers
        if num_matching_answers > avg_len_gold_answers:
            fn = 0
        return tp, fp, fn

    def add_recall_rows(self, answer_id_matches_with_gold, golds, worker_id):
        for gold_option in golds:
            gold_answers = golds[gold_option]["answers"]
            for gold_answer_id in gold_answers:
                if gold_answer_id not in answer_id_matches_with_gold[gold_option]:
                    self.add_row_to_answer_csv(worker_id, '', '', '',
                                               gold_option, golds[gold_option], gold_answer_id, gold_answers)

    def calculate_answer_matches_for_one_worker_for_one_instance(self, curr_worker_dict, instance_key):
        golds = self.expert_dict[instance_key]
        predicted = curr_worker_dict[instance_key]
        predicted_answers = predicted["answers"]
        # gold_answer_starts = gold["start_indices"]
        # gold_answer_ends = gold["end_indices"]
        # predicted_answer_starts = predicted["start_indices"]
        # predicted_answer_ends = predicted["end_indices"]
        answer_matches = defaultdict(list)
        for predicted_answer_id in predicted_answers:
            for gold_option in golds:
                gold_answers = golds[gold_option]["answers"]
                for gold_answer_id in gold_answers:
                    # index_eval = self.evaluate_answer_match_rate_with_indices(gold_answer_starts[answer1_id],
                    #                                     predicted_answer_starts[answer2_id], gold_answer_ends[answer1_id],
                    #                                     predicted_answer_ends[answer2_id])
                    string_eval = self.evaluate_answer_match_rate(gold_answers[gold_answer_id],
                                                                  predicted_answers[predicted_answer_id])
                    if string_eval >= 0.5:
                        answer_matches[predicted_answer_id].append((gold_option, gold_answer_id))
        return answer_matches

    def calculate_question_by_answer_accuracy_for_one_worker_for_one_instance(self, worker_id, curr_worker_dict, instance_key):
        answer_matches = self.calculate_answer_matches_for_one_worker_for_one_instance(curr_worker_dict, instance_key)
        golds = self.expert_dict[instance_key]
        predicted = curr_worker_dict[instance_key]
        # gold_questions = gold["questions"]
        predicted_questions = predicted["questions"]
        num_matching_questions = 0
        for predicted_answer_id in answer_matches:
            found_gold_match = False
            for gold_option, gold_answer_id in answer_matches[predicted_answer_id]:
                gold_questions = golds[gold_option]["questions"]
                if gold_questions[gold_answer_id] == predicted_questions[predicted_answer_id]:
                    found_gold_match = True
            if found_gold_match:
                num_matching_questions += 1
            else:
                # that means none of experts agree
                for gold_option, gold_answer_id in answer_matches[predicted_answer_id]:
                    gold_questions = golds[gold_option]["questions"]
                    self.add_question_disagreement(worker_id, golds[gold_option], gold_answer_id, predicted,
                                                       predicted_answer_id, gold_questions, predicted_questions,
                                                       gold_option)
        num_golds_questions = []
        for gold_option in golds:
            gold_questions = golds[gold_option]["questions"]
            num_golds_questions.append(len(gold_questions))
        try:
            avg_len_gold_questions = int(math.ceil(sum(num_golds_questions) / len(num_golds_questions)))
        except:
            avg_len_gold_questions = 0
        tp = num_matching_questions
        fp = (len(predicted_questions) - num_matching_questions)
        fn = avg_len_gold_questions - num_matching_questions
        if fn < 0:
            fn = 0
        return tp, fp, fn

    def create_question_disagreement_dataframe(self):
        worker_ids = self.results_df["WorkerId"].unique()
        for worker_id in worker_ids:
            df = pd.DataFrame([], columns=['sentence', "predicate", 'expert_worker_id',"gold_property",  'gold_question', 'gold_answer',
                                           "predicted_property", "predicted_question", "predicted_answer"])
            df.to_csv(self.question_disagreement_template.format(worker_id), index=False, encoding="utf-8")

    def create_answer_csv_dataframe(self):
        worker_ids = self.results_df["WorkerId"].unique()
        for worker_id in worker_ids:
            df = pd.DataFrame([], columns=['sentence', "predicate", 'expert_worker_id', "gold_property", 'gold_question',
                                       'gold_answer', "predicted_property", "predicted_question", "predicted_answer"])
            df.to_csv(self.answer_csv_template.format(worker_id), index=False, encoding="utf-8")

    def get_predicate_from_sentence(self, sentence, index):
        return sentence.split(' ')[index]

    def add_row_to_answer_csv(self, worker_id, predicted, predicted_answer_id, predicted_answers,
                              gold_option, gold, gold_answer_id, gold_answers):
        file_df = pd.read_csv(self.answer_csv_template.format(worker_id))
        sentence = predicted["sentence"] if predicted else gold["sentence"]
        index = predicted["index"] if predicted else gold["index"]
        predicate = self.get_predicate_from_sentence(sentence, index)
        predicted_question = predicted["questions"][predicted_answer_id] if predicted else ''
        predicted_property = ''
        if predicted and predicted_answer_id in predicted["properties"]:
            predicted_property = predicted["properties"][predicted_answer_id]
        predicted_answer = predicted_answers[predicted_answer_id] if predicted else ''
        gold_question = gold["questions"][gold_answer_id] if gold else ''
        gold_property = ''
        if gold and gold_answer_id in gold["properties"]:
            gold_property = gold["properties"][gold_answer_id]
        gold_answer = gold_answers[gold_answer_id] if gold else ''
        row_to_add = [[sentence, predicate, gold_option, gold_property, gold_question, gold_answer,
                       predicted_property, predicted_question, predicted_answer]]
        row_df_to_add = pd.DataFrame(row_to_add, columns=list(file_df.columns))
        file_df = file_df.append(row_df_to_add)
        file_df.to_csv(self.answer_csv_template.format(worker_id), index=False, encoding="utf-8")

    def add_question_disagreement(self, worker_id, gold, gold_answer_id, predicted, predicted_answer_id,
                                  gold_questions, predicted_questions, gold_option):
        file_df = pd.read_csv(self.question_disagreement_template.format(worker_id))
        sentence = gold["sentence"]
        predicate = self.get_predicate_from_sentence(sentence, gold["index"])
        gold_question = gold_questions[gold_answer_id]
        gold_property = ''
        if gold_answer_id in gold["properties"]:
            gold_property = gold["properties"][gold_answer_id]
        gold_answer = gold["answers"][gold_answer_id]
        predicted_question = predicted_questions[predicted_answer_id]
        predicted_property = ''
        if predicted_answer_id in predicted["properties"]:
            predicted_property = predicted["properties"][predicted_answer_id]
        predicted_answer = predicted["answers"][predicted_answer_id]
        row_to_add = [[sentence, predicate, gold_option, gold_property, gold_question, gold_answer, predicted_property,
                       predicted_question, predicted_answer]]
        row_df_to_add = pd.DataFrame(row_to_add, columns=file_df.columns)
        file_df = file_df.append(row_df_to_add)
        file_df.to_csv(self.question_disagreement_template.format(worker_id), index=False, encoding="utf-8")

    def calculate_answer_accuracy_for_worker(self, worker_id):
        curr_worker_dict = self.dict_by_worker[worker_id]
        answers_tp = 0
        answers_fp = 0
        answers_fn = 0
        for instance_key in curr_worker_dict:
            tp, fp, fn = self.calculate_answer_accuracy_for_one_worker_for_one_instance(worker_id, curr_worker_dict, instance_key)
            answers_tp += tp
            answers_fp += fp
            answers_fn += fn
        return answers_tp, answers_fp, answers_fn

    def calculate_question_accuracy_for_worker(self, worker_id):
        curr_worker_dict = self.dict_by_worker[worker_id]
        questions_tp = 0
        questions_fp = 0
        questions_fn = 0
        for instance_key in curr_worker_dict:
            tp, fp, fn = self.calculate_question_by_answer_accuracy_for_one_worker_for_one_instance(worker_id, curr_worker_dict, instance_key)
            questions_tp += tp
            questions_fp += fp
            questions_fn += fn
        return questions_tp, questions_fp, questions_fn

    @staticmethod
    def evaluate_answer_match_rate(answer1, answer2):
        answer1_words = set(answer1.split(" "))
        answer2_words = set(answer2.split(" "))
        return len(answer1_words.intersection(answer2_words)) / len(answer1_words.union(answer2_words))

    @staticmethod
    def evaluate_answer_match_rate_with_indices(answer1_start, answer2_start, answer1_end, answer2_end):
        answer1_indices = set(range(answer1_start, answer1_end + 1))
        answer2_indices = set(range(answer2_start, answer2_end + 1))
        return len(answer1_indices.intersection(answer2_indices)) / len(answer1_indices.union(answer2_indices))

    def create_worker_score_dataframe(self):
        worker_ids = self.results_df["WorkerId"].unique()
        df = pd.DataFrame([], columns=['worker_id', 'recall', 'precision', 'f1',
                                       'question_recall', 'question_precision', 'question_f1',
                                       "num_hits", "avg_qa_per_hit"])
        df['worker_id'] = worker_ids
        df.to_csv(self.worker_score_filename, index=False, encoding="utf-8")

    def add_workers_question_accuracy_to_df(self, worker_ids, score_df):
        for worker_id in worker_ids:
            tp, fp, fn = self.calculate_question_accuracy_for_worker(worker_id)
            worker_question_recall = tp / (fn + tp)
            if worker_question_recall > 1:
                print('error: recall greater than 1')
            try:
                worker_question_precision = tp / (tp + fp)
            except:
                worker_question_precision = 0
            worker_answer_f1 = (2 * tp) / ((2 * tp) + fp + fn)
            score_df.loc[score_df["worker_id"] == worker_id, "question_recall"] = worker_question_recall
            score_df.loc[score_df["worker_id"] == worker_id, "question_precision"] = worker_question_precision
            score_df.loc[score_df["worker_id"] == worker_id, "question_f1"] = worker_answer_f1

    def add_workers_answer_accuracy_to_df(self, worker_ids, score_df):
        for worker_id in worker_ids:
            tp, fp, fn = self.calculate_answer_accuracy_for_worker(worker_id)
            worker_answer_recall = tp / (fn + tp)
            if worker_answer_recall > 1:
                print('error: recall greater than 1')
            try:
                worker_answer_precision = tp / (tp + fp)
            except:
                worker_answer_precision = 0
            worker_answer_f1 = (2 * tp) / ((2 * tp) + fp + fn)
            score_df.loc[score_df["worker_id"] == worker_id, "recall"] = worker_answer_recall
            score_df.loc[score_df["worker_id"] == worker_id, "precision"] = worker_answer_precision
            score_df.loc[score_df["worker_id"] == worker_id, "f1"] = worker_answer_f1

    def calculate_score_for_all_workers(self):
        self.create_worker_score_dataframe()
        self.create_question_disagreement_dataframe()
        self.create_answer_csv_dataframe()
        score_df = pd.read_csv(self.worker_score_filename)
        worker_ids = self.results_df["WorkerId"].unique()
        self.add_workers_answer_accuracy_to_df(worker_ids, score_df)
        self.add_workers_question_accuracy_to_df(worker_ids, score_df)
        self.add_worker_statistics(worker_ids, score_df)
        score_df.to_csv(self.worker_score_filename, index=False)

    def add_worker_statistics(self, worker_ids, score_df):
        results_df = pd.read_csv(self.batch_results_file)
        num_hits_per_worker = dict()
        num_qas_per_worker = defaultdict(int)
        num_qas_per_hit_per_worker = dict()
        num_of_workers = len(worker_ids)
        num_of_hits = len(results_df)
        average_hit_per_worker = num_of_hits / num_of_workers
        for worker in worker_ids:
            hits_by_worker = results_df[results_df["WorkerId"] == worker]
            num_hits_per_worker[worker] = len(hits_by_worker)
            score_df.loc[score_df["worker_id"] == worker, "num_hits"] = len(hits_by_worker)
            for index, row in hits_by_worker.iterrows():
                json_answer_loaded = json.loads(row["Answer.taskAnswers"])[0]
                question_id_max = len(json_answer_loaded)
                for i in range(question_id_max):
                    if "question-" + str(i) in json_answer_loaded:
                        num_qas_per_worker[worker] += 1
                        if worker not in num_qas_per_hit_per_worker:
                            num_qas_per_hit_per_worker[worker] = dict()
                        if row["HITId"] not in num_qas_per_hit_per_worker[worker]:
                            num_qas_per_hit_per_worker[worker][row["HITId"]] = 0
                        num_qas_per_hit_per_worker[worker][row["HITId"]] += 1
        average_num_of_qa_per_hit_for_worker = dict()
        for worker in worker_ids:
            average_num_of_qa_per_hit_for_worker[worker] = num_qas_per_worker[worker] / num_hits_per_worker[worker]
            score_df.loc[score_df["worker_id"] == worker, "avg_qa_per_hit"] = average_num_of_qa_per_hit_for_worker[worker]
        average_qa_per_hit_per_worker = sum(average_num_of_qa_per_hit_for_worker.values()) / num_of_workers
        total_qa_num = sum(num_qas_per_worker.values())
        average_qas_per_worker = total_qa_num / num_of_workers
        # print("\nWorker Statistics:\n")
        # print("Num of workers: " + str(num_of_workers))
        # print("Num of hits: " + str(num_of_hits))
        # print("Avg hits per worker: " + str(average_hit_per_worker))
        # print("Num of QAs total: " + str(total_qa_num))
        # print("Avg QAs per worker: " + str(average_qas_per_worker))
        # print("Avg QAs per hit per worker: " + str(average_qa_per_hit_per_worker))
        #
        # print("Hits per worker: " + str(num_hits_per_worker) + '\n')
        # print("Avg QA per hit per worker: " + str(average_num_of_qa_per_hit_for_worker))



'''
Usage example: need to specify all paths and run the function that creates all the analysis files
'''

batch_results_file = "../batches/training/crowd_batch1/crowd_batch1_results.csv"
expert_results_file = "../batches/training/crowd_batch1/expert_batch1_results.csv"
worker_score_filename = "../batches/training/crowd_batch1/worker_score_batch1.csv"
question_disagreement_template = '../batches/training/crowd_batch1/question_disagreement_batch1_{}.csv'
answer_csv_template = '../batches/training/crowd_batch1/batch1_report_{}.csv'
feedback_calculator1 = FeedbackCalculator(batch_results_file, expert_results_file, worker_score_filename,
                                          question_disagreement_template, answer_csv_template)
feedback_calculator1.calculate_score_for_all_workers()