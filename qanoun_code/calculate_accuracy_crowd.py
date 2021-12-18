import json
from collections import defaultdict
import numpy as np
import pandas as pd
import itertools


batch_results_file = "crowd_batch2/crowd_batch2_results.csv"
of_what_filename = "crowd_batch2/of_what_matches.txt"
what_kind_vs_filename = "crowd_batch2/what_kind_vs_consist_and_part.txt"
test_sll_same_file = "crowd_batch/test_accuracy_all_same.csv"
lab_batch_results_file = "lab_batch/batch_results_after_fix.csv"

QUESTIONS_DICT = {1: "What is the [PROPERTY] of (the) [W]?", 2: "Whose [W]?", 3: "Where is the [W]?",
                  4: "How much /How many [W]?",
                  5: "What is (the) [W] part of?", 6: "Who/What does [W] consist of?",
                  7: "Which entity/organization does the [W] belong to?", 8: "What/Who is (the) [W]?",
                  9: "What kind of [W]?",
                  10: "[W] of what?"}


class AccuracyCalculator:
    def __init__(self, results_file, confusion_matrix_name, worker_score_filename):
        self.batch_results_file = results_file
        self.results_df = pd.read_csv(self.batch_results_file)
        self.dict_by_hit = self.create_dictionary_by_hit()
        self.confusion_matrix_name = confusion_matrix_name
        self.worker_score_filename = worker_score_filename
        self.disagreement_dict = self.create_dictionary_by_hit()
        self.property_couples = list()

    def create_dictionary_by_hit(self):
        hit_dict = dict()
        hits = self.results_df["HITId"].unique()
        for hit_id in hits:
            hit_dict[hit_id] = self.create_answers_dict_for_each_hit(hit_id)
        return hit_dict

    def create_answers_dict_for_each_hit(self, hit_id):
        answers_dict = dict()
        filtered_df = self.results_df[self.results_df["HITId"] == hit_id]
        df_for_hit = filtered_df[["WorkerId", "Input.sentenceId", "Answer.taskAnswers", "Input.sentence", "Input.index"]]
        for index, row in df_for_hit.iterrows():
            answers_dict[row["WorkerId"]] = self.parser_json_answer(row["Answer.taskAnswers"], row["Input.sentence"],
                                                            row["Input.index"], row["Input.sentenceId"])
        return answers_dict

    @staticmethod
    def parser_json_answer(json_answer, sentence, index, sentenceid):
        json_answer_loaded = json.loads(json_answer)[0]
        questions_answers_dict = {"questions": dict(), "answers": dict(), "properties": dict(),
                                                        "sentenceid": sentenceid, "sentence": sentence, "index": index}
        question_id_max = len(json_answer_loaded)
        for i in range(question_id_max):
            if "question-" + str(i) in json_answer_loaded:
                questions_answers_dict["questions"][i] = json_answer_loaded["question-" + str(i)]
                questions_answers_dict["answers"][i] = json_answer_loaded["answers-" + str(i)]
            if "question-" + str(i) + "-property-input" in json_answer_loaded:
                questions_answers_dict["properties"][i] = json_answer_loaded["question-" + str(i) + "-property-input"]
        return questions_answers_dict

    def create_worker_score_dataframe(self):
        worker_ids = self.results_df["WorkerId"].unique()
        df = pd.DataFrame([], columns=['worker_id', 'recall', 'precision', 'f1', "num_hits", "avg_qa_per_hit"])
        df['worker_id'] = worker_ids
        df.to_csv(self.worker_score_filename, index=False, encoding="utf-8")

    def calculate_score_for_all_workers(self):
        worker_ids = self.results_df["WorkerId"].unique()
        score_df = pd.read_csv(self.worker_score_filename)
        for worker_id in worker_ids:
            worker_answer_tp = 0
            worker_answer_fp = 0
            worker_answer_fn = 0
            filtered_df_for_worker = self.results_df[self.results_df["WorkerId"] == worker_id]
            hits_for_worker = filtered_df_for_worker["HITId"].unique()
            for hit_id in hits_for_worker:
                filtered_df_for_hit = self.results_df[self.results_df["HITId"] == hit_id]
                filtered_df_for_hit = filtered_df_for_hit[filtered_df_for_hit["WorkerId"] != worker_id]
                other_worker_ids = filtered_df_for_hit["WorkerId"].unique()
                for gold_worker_id in other_worker_ids:
                    tp, fp, fn = self.calculate_answer_accuracy_for_two_workers_for_one_hit(gold_worker_id, worker_id,
                                                                        self.dict_by_hit[hit_id], hit_id)
                    worker_answer_tp += tp
                    worker_answer_fp += fp
                    worker_answer_fn += fn
            worker_answer_recall = worker_answer_tp / (worker_answer_fn + worker_answer_tp)
            try:
                worker_answer_precision = worker_answer_tp / (worker_answer_tp + worker_answer_fp)
            except:
                worker_answer_precision = 0
            worker_answer_f1 = (2 * worker_answer_tp) / ((2 * worker_answer_tp) + worker_answer_fp + worker_answer_fn)
            score_df.loc[score_df["worker_id"] == worker_id, "recall"] = worker_answer_recall
            score_df.loc[score_df["worker_id"] == worker_id, "precision"] = worker_answer_precision
            score_df.loc[score_df["worker_id"] == worker_id, "f1"] = worker_answer_f1
        score_df.to_csv(self.worker_score_filename, index=False)


    def calculate_accuracy_for_all_hits(self):
        questions_tp = 0
        questions_fp = 0
        questions_fn = 0
        answers_tp = 0
        answers_fp = 0
        answers_fn = 0
        for hit_id in self.dict_by_hit:
            questions_result, answers_result = self.calculate_accuracy_for_hit(hit_id)
            questions_tp += questions_result[0]
            questions_fp += questions_result[1]
            questions_fn += questions_result[2]
            answers_tp += answers_result[0]
            answers_fp += answers_result[1]
            answers_fn += answers_result[2]
        questions_recall = questions_tp / (questions_fn + questions_tp)
        questions_precision = questions_tp / (questions_tp + questions_fp)
        questions_f1 = (2 * questions_tp) / ((2 * questions_tp) + questions_fp + questions_fn)
        answers_recall = answers_tp / (answers_fn + answers_tp)
        answers_precision = answers_tp / (answers_tp + answers_fp)
        answers_f1 = (2 * answers_tp) / ((2 * answers_tp) + answers_fp + answers_fn)
        print("Answers Recall: " + str(answers_recall))
        print("Answers Precision " + str(answers_precision))
        print("Answers F1: " + str(answers_f1))
        print("\n")
        print("Questions Recall: " + str(questions_recall))
        print("Questions Precision: " + str(questions_precision))
        print("Questions F1: " + str(questions_f1))


    def calculate_accuracy_for_hit(self, hit_id):
        curr_hit_dict = self.dict_by_hit[hit_id]
        # hit_id : {gold_ans_id: list of predicted that match}
        worker_ids_for_hit = list(curr_hit_dict.keys())
        worker_id_pairs = [(worker_ids_for_hit[id1], worker_ids_for_hit[id2]) for id1 in range(len(worker_ids_for_hit))
                            for id2 in range(id1+1, len(worker_ids_for_hit))]
        questions_tp = 0
        questions_fp = 0
        questions_fn = 0
        answers_tp = 0
        answers_fp = 0
        answers_fn = 0
        for worker_pair in worker_id_pairs:
            result_values = self.calculate_answer_accuracy_for_two_workers_for_one_hit(worker_pair[0], worker_pair[1],
                                                                                       curr_hit_dict, hit_id)
            answers_tp += result_values[0]
            answers_fp += result_values[1]
            answers_fn += result_values[2]
            result_values = self.calculate_question_by_answer_accuracy_for_two_workers_for_hit(worker_pair[0],
                                                                                          worker_pair[1], curr_hit_dict)
            questions_tp += result_values[0]
            questions_fp += result_values[1]
            questions_fn += result_values[2]
        return [questions_tp, questions_fp, questions_fn], [answers_tp, answers_fp, answers_fn]

    def calculate_answer_accuracy_for_two_workers_for_one_hit(self, worker_id1, worker_id2, curr_hit_dict, hit_id):
        num_matching_answers = 0
        gold = curr_hit_dict[worker_id1]
        predicted = curr_hit_dict[worker_id2]
        gold_answers = gold["answers"]
        predicted_answers = predicted["answers"]
        for answer1_id in gold_answers:
            for answer2_id in predicted_answers:
                if self.evaluate_answer_match_rate(gold_answers[answer1_id], predicted_answers[answer2_id]) >= 0.5:
                    num_matching_answers += 1
                    if answer1_id in self.disagreement_dict[hit_id][worker_id1]["answers"]:
                        del self.disagreement_dict[hit_id][worker_id1]["answers"][answer1_id]
                    if answer2_id in self.disagreement_dict[hit_id][worker_id2]["answers"]:
                        del self.disagreement_dict[hit_id][worker_id2]["answers"][answer2_id]
        tp = num_matching_answers
        fp = (len(predicted_answers) - num_matching_answers)
        fn = len(gold_answers) - num_matching_answers
        return tp, fp, fn
        # recall = tp / (fn + tp)
        # precision = tp / (tp + fp)
        # f1 = (2 * tp) / ((2 * tp) + fp + fn)

    def print_answers_without_match(self):
        print("\nAnswers without match:\n")
        for hit_id in self.disagreement_dict:
            for worker_id in self.disagreement_dict[hit_id]:
                for answer_id in self.disagreement_dict[hit_id][worker_id]["answers"]:
                    print("HIT id: ", hit_id)
                    print("Worker id: ", worker_id)
                    print("Sentence: " + self.disagreement_dict[hit_id][worker_id]["sentence"])
                    print("Q: " + self.disagreement_dict[hit_id][worker_id]["questions"][answer_id])
                    if answer_id in self.disagreement_dict[hit_id][worker_id]["properties"]:
                        print("PROPERTY: " + self.disagreement_dict[hit_id][worker_id]["properties"][answer_id])
                    print("Answer without match: " + self.disagreement_dict[hit_id][worker_id]["answers"][answer_id])
                    print("\n")

    def evaluate_answer_match_rate(self, answer1, answer2):
        answer1_words = set(answer1.split(" "))
        answer2_words = set(answer2.split(" "))
        return len(answer1_words.intersection(answer2_words)) / len(answer1_words.union(answer2_words))

    def calculate_answer_matches_for_two_workers_for_hit(self, worker_id1, worker_id2, curr_hit_dict):
        gold = curr_hit_dict[worker_id1]
        predicted = curr_hit_dict[worker_id2]
        gold_answers = gold["answers"]
        predicted_answers = predicted["answers"]
        answer_matches = defaultdict(list)
        for answer1_id in gold_answers:
            for answer2_id in predicted_answers:
                if self.evaluate_answer_match_rate(gold_answers[answer1_id], predicted_answers[answer2_id]) >= 0.5:
                    answer_matches[answer1_id].append(answer2_id)
        return answer_matches

    def calculate_question_by_answer_accuracy_for_two_workers_for_hit(self, worker_id1, worker_id2, curr_hit_dict):
        answer_matches = self.calculate_answer_matches_for_two_workers_for_hit(worker_id1, worker_id2, curr_hit_dict)
        gold = curr_hit_dict[worker_id1]
        predicted = curr_hit_dict[worker_id2]
        gold_questions = gold["questions"]
        predicted_questions = predicted["questions"]
        num_matching_questions = 0
        for answer_id1 in answer_matches:
            if gold_questions[answer_id1] == predicted_questions[answer_matches[answer_id1][0]]:
                num_matching_questions += 1
                # self.find_of_what_matches(of_what_filename, gold, predicted, answer_id1, answer_matches[answer_id1][0],
                #                           gold_questions[answer_id1], gold["sentence"], gold["index"])
                if self.translate_questions_to_template(gold["sentence"], gold["index"], gold_questions[answer_id1]) \
                    == QUESTIONS_DICT[1] and \
                    self.translate_questions_to_template(predicted["sentence"], predicted["index"], predicted_questions[answer_matches[answer_id1][0]])\
                    == QUESTIONS_DICT[1]:
                    self.add_to_property_pairs(gold, predicted, answer_id1, answer_matches[answer_id1][0])

            else:
                # self.find_what_kind_vs_consist_and_part_of(what_kind_vs_filename, gold, predicted, answer_id1, answer_matches[answer_id1][0],
                #         gold_questions[answer_id1], predicted_questions[answer_matches[answer_id1][0]], gold["sentence"], gold["index"])
                self.print_question_disagreement(gold, answer_id1, predicted, gold_questions,
                                                                            predicted_questions, answer_matches, True)
        tp = num_matching_questions
        fp = (len(predicted_questions) - num_matching_questions)
        fn = len(gold_questions) - num_matching_questions
        return tp, fp, fn

    def find_of_what_matches(self, filename, gold, predicted, gold_id, predicted_id, question, sentence, index):
        if self.translate_questions_to_template(sentence, index, question) == QUESTIONS_DICT[10]:
            with open(filename, 'a') as file:
                file.write('\n====================\n')
                file.write("Sentence: " + sentence + '\n')
                file.write("Question: " + question + '\n')
                file.write("Answer1: " + gold["answers"][gold_id] + '\n')
                file.write("Answer2: " + predicted["answers"][predicted_id] + '\n')

    def find_what_kind_vs_consist_and_part_of(self, filename, gold, predicted, gold_id, predicted_id, question_gold,
                                              question_predicted, sentence, index):
        translated_q_gold = self.translate_questions_to_template(sentence, index, question_gold)
        translated_q_predicted = self.translate_questions_to_template(sentence, index, question_predicted)
        what_kind = QUESTIONS_DICT[9]
        consist = QUESTIONS_DICT[6]
        part = QUESTIONS_DICT[5]
        if translated_q_gold == what_kind and translated_q_predicted in (consist, part) or\
            translated_q_predicted == what_kind and translated_q_gold in (consist, part):
            with open(filename, 'a') as file:
                file.write('\n====================\n')
                file.write("Sentence: " + sentence + '\n')
                file.write("Question1: " + question_gold + '\n')
                file.write("Answer1: " + gold["answers"][gold_id] + '\n')
                file.write("Question2: " + question_predicted + '\n')
                file.write("Answer2: " + predicted["answers"][predicted_id] + '\n')


    def add_to_property_pairs(self, gold, predicted, gold_id, predicted_id):
        self.property_couples.append({"sentence": gold["sentence"], "question": gold["questions"][gold_id],
                                      "property1": gold["properties"][gold_id], "property2": predicted["properties"][predicted_id],
                                      "answer1": gold["answers"][gold_id], "answer2": predicted["answers"][predicted_id]})

    def print_property_pairs(self):
        print("\nProperty Pairs:\n")
        for property_pair in self.property_couples:
            print("Sentence: " + property_pair["sentence"])
            print("Question: " + property_pair["question"])
            print("Property1: " + property_pair["property1"])
            print("Property2: " + property_pair["property2"])
            print("Answer1: " + property_pair["answer1"])
            print("Answer2: " + property_pair["answer2"])
            print("\n")

    def print_question_disagreement(self, gold, answer_id1, predicted, gold_questions, predicted_questions, answer_matches, flag):
        if flag:
            print("Sentence: " + gold["sentence"])
            print("Q: " + gold_questions[answer_id1])
            if answer_id1 in gold["properties"]:
                print("PROPERTY: " + gold["properties"][answer_id1])
            print("A: " + gold["answers"][answer_id1])
            print("Q: " + predicted_questions[answer_matches[answer_id1][0]])
            if answer_matches[answer_id1][0] in predicted["properties"]:
                print("PROPERTY: " + predicted["properties"][answer_matches[answer_id1][0]])
            print("A: " + predicted["answers"][answer_matches[answer_id1][0]])
            print("\n")

    def translate_questions_to_template(self, sentence, target_index, question):
        prep_questions = ["Who/What is the [W] [PREP]?", "What is someone a/the [W] [PREP]?"]
        prep_questions_start = ["Who/What is the [W]", "What is someone a/the [W]"]
        split_sent = sentence.split(" ")
        target_word = split_sent[target_index]
        question = question.replace(target_word, "[W]")
        for prep_idx in range(len(prep_questions_start)):
            if prep_questions_start[prep_idx] in question:
                return prep_questions[prep_idx]
        return question

    def create_confusion_matrix_for_two_workers_for_hit(self, worker_id1, worker_id2, curr_hit_dict):
        confusion_matrix = pd.DataFrame(np.zeros(shape=(len(QUESTIONS_DICT), len(QUESTIONS_DICT))),
                                        columns=list(QUESTIONS_DICT.values()), index=list(QUESTIONS_DICT.values()))
        gold = curr_hit_dict[worker_id1]
        predicted = curr_hit_dict[worker_id2]
        gold_questions = gold["questions"]
        predicted_questions = predicted["questions"]
        answer_matches = self.calculate_answer_matches_for_two_workers_for_hit(worker_id1, worker_id2, curr_hit_dict)
        for answer_id1 in answer_matches:
            gold_question_template = self.translate_questions_to_template(gold["sentence"], gold["index"],
                                                                                            gold_questions[answer_id1])
            predicted_question_template = self.translate_questions_to_template(gold["sentence"],gold["index"],
                                                                    predicted_questions[answer_matches[answer_id1][0]])
            confusion_matrix[gold_question_template][predicted_question_template] += 1

        # confusion_matrix.to_csv(confusion_matrix_name)
        return confusion_matrix

    def create_confusion_matrix_for_hit(self, hit_id):
        curr_hit_dict = self.dict_by_hit[hit_id]
        worker_ids_for_hit = list(curr_hit_dict.keys())
        worker_id_pairs = list(itertools.combinations(worker_ids_for_hit, 2))
        first = True
        total_matrix = None
        for worker_pair in worker_id_pairs:
            curr_confusion_matrix = self.create_confusion_matrix_for_two_workers_for_hit(worker_pair[0], worker_pair[1], curr_hit_dict)
            if first:
                total_matrix = curr_confusion_matrix
                first = False
            else:
                total_matrix += curr_confusion_matrix
        return total_matrix

    def create_confusion_matrix_for_all_hits(self):
        total_matrix = None
        first = True
        for hit_id in self.dict_by_hit:
            curr_confusion_matrix = self.create_confusion_matrix_for_hit(hit_id)
            try:
                curr_confusion_matrix.empty
                if first:
                    total_matrix = curr_confusion_matrix
                    first = False
                else:
                    total_matrix += curr_confusion_matrix
            except:
                print("this hit is not done " + str(hit_id))
        return total_matrix

    def fold_confusion_matrix(self, matrix):
        for i in range(1, len(QUESTIONS_DICT) + 1):
            for j in range(i + 1, len(QUESTIONS_DICT) + 1):
                matrix[QUESTIONS_DICT[i]][QUESTIONS_DICT[j]] = matrix[QUESTIONS_DICT[i]][QUESTIONS_DICT[j]] + \
                                                               matrix[QUESTIONS_DICT[j]][QUESTIONS_DICT[i]]
                matrix[QUESTIONS_DICT[j]][QUESTIONS_DICT[i]] = 0
        return matrix

    def create_csv_from_confusion_matrix(self):
        matrix = self.create_confusion_matrix_for_all_hits()
        # matrix.to_csv("crowd_batch2/test_unfolded.csv")
        matrix = self.fold_confusion_matrix(matrix)
        matrix.to_csv(self.confusion_matrix_name)

    def print_property_statistics(self):
        property_counter = defaultdict(int)
        for hit_id in self.dict_by_hit:
            for worker_id in self.dict_by_hit[hit_id]:
                for q_id in self.dict_by_hit[hit_id][worker_id]["questions"]:
                    if q_id in self.dict_by_hit[hit_id][worker_id]["properties"]:
                        property_counter[self.dict_by_hit[hit_id][worker_id]["properties"][q_id].lower()] += 1
        print("\nProperty Distribution:\n")
        for property in property_counter:
            print(property + ": " + str(property_counter[property]))
        print("Total usage: " + str(sum(property_counter.values())))

    def print_worker_statistics(self):
        results_df = pd.read_csv(self.batch_results_file)
        workers = results_df["WorkerId"].unique()
        num_hits_per_worker = dict()
        num_qas_per_worker = defaultdict(int)
        num_qas_per_hit_per_worker = dict()
        num_of_workers = len(workers)
        num_of_hits = len(results_df)
        average_hit_per_worker = num_of_hits / num_of_workers
        score_df = pd.read_csv(self.worker_score_filename)
        for worker in workers:
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
        for worker in workers:
            average_num_of_qa_per_hit_for_worker[worker] = num_qas_per_worker[worker] / num_hits_per_worker[worker]
            score_df.loc[score_df["worker_id"] == worker, "avg_qa_per_hit"] = average_num_of_qa_per_hit_for_worker[worker]
        average_qa_per_hit_per_worker = sum(average_num_of_qa_per_hit_for_worker.values()) / num_of_workers
        total_qa_num = sum(num_qas_per_worker.values())
        average_qas_per_worker = total_qa_num / num_of_workers
        print("\nWorker Statistics:\n")
        print("Num of workers: " + str(num_of_workers))
        print("Num of hits: " + str(num_of_hits))
        print("Avg hits per worker: " + str(average_hit_per_worker))
        print("Num of QAs total: " + str(total_qa_num))
        print("Avg QAs per worker: " + str(average_qas_per_worker))
        print("Avg QAs per hit per worker: " + str(average_qa_per_hit_per_worker))

        print("Hits per worker: " + str(num_hits_per_worker) + '\n')
        print("Avg QA per hit per worker: " + str(average_num_of_qa_per_hit_for_worker))
        score_df.to_csv(self.worker_score_filename, index=False)


# accuracy_calculator = AccuracyCalculator(batch_results_file, "crowd_batch/confusion_matrix_crowd.csv")
# accuracy_calculator.calculate_accuracy_for_all_hits()
# accuracy_calculator.print_answers_without_match()
# accuracy_calculator.create_csv_from_confusion_matrix()

# test_all_same = AccuracyCalculator(test_sll_same_file)
# test_all_same.calculate_accuracy_for_all_hits()

# lab_batch = AccuracyCalculator(lab_batch_results_file, "lab_batch/confusion_matrix_lab_batch.csv")
# lab_batch.calculate_accuracy_for_all_hits()
# lab_batch.create_csv_from_confusion_matrix()

accuracy_calculator2 = AccuracyCalculator(batch_results_file, "crowd_batch2/confusion_matrix_crowd.csv", "crowd_batch2/workers_score.csv")
accuracy_calculator2.calculate_accuracy_for_all_hits()
# accuracy_calculator2.create_worker_score_dataframe()
# accuracy_calculator2.calculate_score_for_all_workers()
accuracy_calculator2.print_worker_statistics()
# accuracy_calculator2.print_property_statistics()
# accuracy_calculator2.print_property_pairs()
# accuracy_calculator2.print_answers_without_match()
# accuracy_calculator2.create_csv_from_confusion_matrix()

# accuracy_calculator_test = AccuracyCalculator("crowd_batch2/test/crowd_batch2_results_test.csv", "crowd_batch2/test/confusion_matrix_crowd.csv")
# accuracy_calculator_test.calculate_accuracy_for_all_hits()
# accuracy_calculator_test.print_answers_without_match()
