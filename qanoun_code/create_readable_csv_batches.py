from typing import Iterable, Tuple, List, Dict, Callable, Any, Optional
import pandas as pd
import json

QUESTIONS_TEMPLATES = {1: "What is the [PROPERTY] of (the) [W]?", 2: "Whose [W]?", 3: "Where is the [W]?",
                  4: "How much /How many [W]?",
                  5: "What is the [W] a part/member of?", 6: "What/Who is a part/member of [W]?", 7: "What/Who is (the) [W]?",
                  8: "What kind of [W]?", 9: "When is the [W]?"}


class ParsedQAInfo:
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

class RawCSVReader:
    columns = ["sentence", "sentence_id", "target_idx", "instance_id", "target", "worker_id", 
               "question_template", "question", "answer", "answer_range", "property", "comment", 
               "general_comment"]
    def __init__(self, raw_results_file_name, final_filename):
        self.raw_results_file_name = raw_results_file_name
        self.final_filename = final_filename
        self.raw_results_df = pd.read_csv(self.raw_results_file_name)

    def create_readable_csv(self):
        df = self.to_readable_df(self.raw_results_df)
        df.to_csv(self.final_filename, index=False, encoding="utf-8")
    
    @staticmethod    
    def read_annot_csv(raw_results_fn: str) -> pd.DataFrame:
        "Extracts a readable DataFrame from a raw results CSV file (downloaded from MTurk)."
        raw_results_df = pd.read_csv(raw_results_fn)
        return RawCSVReader.to_readable_df(raw_results_df)
    
    @staticmethod    
    def to_readable_df(raw_results_df) -> pd.DataFrame:
        qa_records = []
        for index, ins_row in raw_results_df.iterrows():
            # instance_info is [sentence, sent_id, target_idx, instance_id, predicate, worker_id]
            instance_info_list = RawCSVReader.parse_instance_info(ins_row)
            predicate = instance_info_list[-2]
            qa_annotations: List[ParsedQAInfo] = list(RawCSVReader.parser_json(ins_row["Answer.taskAnswers"]))
            for qa_annotation in qa_annotations:
                # qa_info is [question_template, question, answer, answer_range, 
                #               property_word, comment, general_comment]
                qa_info = RawCSVReader.extract_qa_info(predicate,
                                    qa_annotation.question, qa_annotation.answer, qa_annotation.start, qa_annotation.end, qa_annotation.property_word,
                                    qa_annotation.what_who_consist, qa_annotation.what_who_copular,
                                    qa_annotation.part_member_consist, qa_annotation.part_member_partof,
                                    qa_annotation.comment, qa_annotation.general_comment)
                qa_as_list = instance_info_list + qa_info
                qa_records.append(qa_as_list)
            # To account for no-QA in evaluations, we will keep a row with empty Q&A fields 
            #  for instances having no QAs.
            if len(qa_annotations) == 0:
                empty_qa_info = ['','','',None,'','','']
                qa_records.append(instance_info_list + empty_qa_info)
        df = pd.DataFrame(qa_records, columns=RawCSVReader.columns)
        return df

    @staticmethod
    def parse_instance_info(ins_row: pd.Series):
        sentence, sent_id, target_idx, worker_id = (ins_row["Input.sentence"], 
                                                    ins_row["Input.sentenceId"], 
                                                    ins_row["Input.index"], 
                                                    ins_row["WorkerId"])
        predicate = RawCSVReader.get_target_word(sentence, target_idx)
        instance_id = sent_id + "_" + str(target_idx)
        return [sentence, sent_id, target_idx, instance_id, predicate, worker_id]
        
    @staticmethod
    def parser_json(json_answer):
        json_answer_loaded = json.loads(json_answer)[0]
        question_id_max = len(json_answer_loaded)
        for i in range(question_id_max):
            parse_instance = ParsedQAInfo()
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

    @staticmethod
    def get_target_word(sentence, target_index):
        split_sent = sentence.split(" ")
        target_word = split_sent[target_index]
        return target_word

    @staticmethod
    def get_question_template_from_question(target_word, question):
        question_template = question.replace(target_word, "[W]")
        return question_template

    @staticmethod
    def extract_qa_info(predicate, question, answer, start, end,
                          property_word, what_who_consist, what_who_copular,
                          part_member_consist, part_member_partof, comment, general_comment):
        question_template = RawCSVReader.get_question_template_from_question(predicate, question)
        if "[PROPERTY]" in question:
            question = question.replace('[PROPERTY]', property_word)
        question = RawCSVReader.remove_slash_from_question(question, question_template,
                                                   what_who_consist, what_who_copular,
                                                   part_member_consist, part_member_partof)
        answer_range = RawCSVReader.info_to_answer_span(int(start), int(end))
        return [question_template, question, answer, answer_range, 
                property_word, comment, general_comment]

    @staticmethod
    def info_to_answer_span(start, end) -> Tuple[int, int]:
        if end==start:
            end += 1
        elif end<start:
            raise ValueError(f"end index {end} is smaller than start index {start}")
        return (start, end)
    
    @staticmethod
    def remove_slash_from_question(question, question_template,
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
if __name__ == "__main__":
    crowd_batch5_creator = RawCSVReader("training/crowd_batch1/crowd_batch1_results.csv", "readable_example.csv")
    crowd_batch5_creator.create_readable_csv()