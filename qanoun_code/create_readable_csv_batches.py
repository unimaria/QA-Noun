import pandas as pd
import json

QUESTIONS_TEMPLATES = {1: "What is the [PROPERTY] of (the) [W]?", 2: "Whose [W]?", 3: "Where is the [W]?",
                  4: "How much /How many [W]?",
                  5: "What is (the) [W] part of?", 6: "Who/What does [W] consist of?",
                  7: "Which entity/organization does the [W] belong to?", 8: "What/Who is (the) [W]?",
                  9: "What kind of [W]?",
                  10: "[W] of what?"}


class ParsingClass:
    def __init__(self):
        self.question = ''
        self.answer = ''
        self.start = ''
        self.end = ''
        self.property_word = ''
        self.comment = ''
        self.general_comment = ''

class QANounCSVCreator:
    def __init__(self, results_file_name, final_filename):
        self.results_file_name = results_file_name
        self.final_filename = final_filename
        self.results_file_df = pd.read_csv(self.results_file_name)
        self.to_df_list = []
        self.columns = ["sentence", "sent_id", "predicate_idx", "key", "predicate", "worker_id", "question_template",
                        "question", "answer", "answer_range", "property", "comment", "general_comment"]

    def create_readable_csv(self):
        for index, row in self.results_file_df.iterrows():
            self.process_one_row(row)

    def process_one_row(self, row):
        for result in self.parser_json(row["Answer.taskAnswers"]):
            self.add_one_row_to_df(row["Input.sentence"], row["Input.sentenceId"], row["Input.index"], row["WorkerId"],
                                   result.question, result.answer, result.start, result.end, result.property_word,
                                   result.comment, result.general_comment)
        df = pd.DataFrame(self.to_df_list, columns=self.columns)
        df.to_csv(self.final_filename, index=False, encoding="utf-8")

    def parser_json(self, json_answer):
        json_answer_loaded = json.loads(json_answer)[0]
        question_id_max = len(json_answer_loaded)
        for i in range(question_id_max):
            parse_instance = ParsingClass()
            if "question-" + str(i) in json_answer_loaded:
                parse_instance.question = json_answer_loaded["question-" + str(i)]
                parse_instance.answer = json_answer_loaded["answers-" + str(i)]
                parse_instance.start = json_answer_loaded["start-" + str(i)]
                parse_instance.end = json_answer_loaded["end-" + str(i)]
                if "question-" + str(i) + "-property-input" in json_answer_loaded:
                    parse_instance.property_word = json_answer_loaded["question-" + str(i) + "-property-input"]
                if "comment-" + str(i) in json_answer_loaded:
                    parse_instance.comment = json_answer_loaded["comment-" + str(i)]
                if "comment-general" in json_answer_loaded:
                    parse_instance.general_comment = json_answer_loaded["comment-general"]
                yield parse_instance

    def get_target_word(self, sentence, target_index):
        split_sent = sentence.split(" ")
        target_word = split_sent[target_index]
        return target_word

    def get_question_template_from_question(self, target_word, question):
        question_template = question.replace(target_word, "[W]")
        return question_template

    def add_one_row_to_df(self, sentence, sent_id, predicate_idx, worker_id, question, answer, start, end,
                          property_word, comment, general_comment):
        predicate = self.get_target_word(sentence, predicate_idx)
        question_template = self.get_question_template_from_question(predicate, question)
        if "[PROPERTY]" in question:
            question = question.replace('[PROPERTY]', property_word)
        answer_range = start+":"+end
        key = sent_id + ":" + str(predicate_idx)
        self.to_df_list.append([sentence, sent_id, predicate_idx, key, predicate, worker_id, question_template,
                                question, answer, answer_range, property_word, comment, general_comment])


# crowd_batch2_creator = QANounCSVCreator("crowd_batch2/crowd_batch2_results.csv", "crowd_batch2/crowd_batch2_readable.csv")
# crowd_batch2_creator.create_readable_csv()

# crowd_batch2_creator = QANounCSVCreator("third expert batch/third_expert_batch_results.csv", "third expert batch/third_expert_batch_results_readable.csv")
# crowd_batch2_creator.create_readable_csv()

expert_batch_4th = QANounCSVCreator("4th_expert_batch/4th_expert_batch_results.csv", "4th_expert_batch/4th_expert_batch_results_readable.csv")
expert_batch_4th.create_readable_csv()