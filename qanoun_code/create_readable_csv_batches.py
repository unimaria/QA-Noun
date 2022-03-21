import pandas as pd
import json

QUESTIONS_TEMPLATES = {1: "What is the [PROPERTY] of (the) [W]?", 2: "Whose [W]?", 3: "Where is the [W]?",
                  4: "How much /How many [W]?",
                  5: "What is the [W] a part/member of?", 6: "What/Who is a part/member of [W]?", 7: "What/Who is (the) [W]?",
                  8: "What kind of [W]?", 9: "When is the [W]?"}


class ParsingClass:
    def __init__(self):
        self.question = ''
        self.answer = ''
        self.start = ''
        self.end = ''
        self.property_word = ''
        self.part_member_consist = ''
        self.part_member_partof = ''
        self.what_who_consist = ''
        self.what_who_copular = ''
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
                                   result.what_who_consist, result.what_who_copular,
                                   result.part_member_consist, result.part_member_partof,
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
                if "question-" + str(i) + "-consist-input-what-who" in json_answer_loaded:
                    parse_instance.what_who_consist = json_answer_loaded["question-" + str(i) + "-consist-input-what-who"]
                if "question-" + str(i) + "-copular-input-what-who" in json_answer_loaded:
                    parse_instance.what_who_copular = json_answer_loaded["question-" + str(i) + "-copular-input-what-who"]
                if "question-" + str(i) + "-consist-input-part-member" in json_answer_loaded:
                    parse_instance.part_member_consist = json_answer_loaded["question-" + str(i) + "-consist-input-part-member"]
                if "question-" + str(i) + "-partof-input-part-member" in json_answer_loaded:
                    parse_instance.part_member_partof = json_answer_loaded["question-" + str(i) + "-partof-input-part-member"]
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
                          property_word, what_who_consist, what_who_copular,
                          part_member_consist, part_member_partof, comment, general_comment):
        predicate = self.get_target_word(sentence, predicate_idx)
        question_template = self.get_question_template_from_question(predicate, question)
        if "[PROPERTY]" in question:
            question = question.replace('[PROPERTY]', property_word)
        question = self.remove_slash_from_question(question, question_template,
                                                   what_who_consist, what_who_copular,
                                                   part_member_consist, part_member_partof)
        answer_range = start+":"+end
        key = sent_id + ":" + str(predicate_idx)
        self.to_df_list.append([sentence, sent_id, predicate_idx, key, predicate, worker_id, question_template,
                                question, answer, answer_range, property_word, comment, general_comment])

    def remove_slash_from_question(self, question, question_template,
                                   what_who_consist, what_who_copular, part_member_consist, part_member_partof):
        if question_template == QUESTIONS_TEMPLATES[5]:
            return question.replace("part/member", part_member_partof)
        elif question_template == QUESTIONS_TEMPLATES[6]:
            question = question.replace("part/member", part_member_consist)
            return question.replace("What/Who", what_who_consist)
        elif question_template == QUESTIONS_TEMPLATES[7]:
            return question.replace("What/Who", what_who_copular)
        else:
            return question

# crowd_batch4_creator = QANounCSVCreator("../batches/crowd_batch4/crowd_batch4_results.csv", "../batches/crowd_batch4/crowd_batch4_results_readable.csv")
# crowd_batch4_creator.create_readable_csv()

# test_slash_creator = QANounCSVCreator("../test_batches/results/test_batch_buttons_result2.csv", "../test_batches/results/test_batch_buttons_result_readable2.csv")
# test_slash_creator.create_readable_csv()

crowd_batch5_creator = QANounCSVCreator("../batches/crowd_batch5/crowd_batch5_results.csv", "../batches/crowd_batch5/crowd_batch5_results_readable.csv")
crowd_batch5_creator.create_readable_csv()