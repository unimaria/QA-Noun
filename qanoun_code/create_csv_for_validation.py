import pandas as pd
import json


class QuestionAndAnswer:
    def __init__(self, question, answer, start_index_answer, end_index_enswer):
        self.question = question
        self.answer = answer
        self.start_index_answer = start_index_answer
        self.end_index_answer = end_index_enswer


class CSVCreator:
    def __init__(self, file):
        self.result_file = file
        self.result_df = pd.read_csv(self.result_file)
        self.sort_result_dataframe()

    def sort_result_dataframe(self):
        # first by sentence Id and then by index
        self.result_df.sort_values(by=["Input.sentenceId", "Input.index"])

    def create_csv_files_for_validation(self, new_filename):
        curr_sentence_id = 0
        curr_index = 0
        curr_sentence = 0
        curr_questions_and_answers = []
        df_list_of_lists = []
        for index, row in self.result_df.iterrows():
            if row["Input.sentenceId"] == curr_sentence_id and row["Input.index"] == curr_index:
                self.add_row_to_qs_and_as(curr_questions_and_answers, row)
            else:
                df_list_of_lists.append(self.create_one_new_df_row(curr_questions_and_answers, curr_sentence_id,
                                                                   curr_index, curr_sentence))
                curr_questions_and_answers = []
                curr_sentence_id = row["Input.sentenceId"]
                curr_index = row["Input.index"]
                curr_sentence = row['Input.sentence']
                self.add_row_to_qs_and_as(curr_questions_and_answers, row)
        # first line is zeros
        df = pd.DataFrame(df_list_of_lists[1:], columns=['sentenceId', 'index', 'sentence', 'questions', 'answers', "ranges"])
        df.to_csv(new_filename, index=False, encoding="utf-8", line_terminator='\r\n')

    def add_row_to_qs_and_as(self, questions_and_answers, row):
        # here I used json online parser http://json.parser.online.fr
        json_loaded = json.loads(row['Answer.taskAnswers'])[0]
        question_id_max = len(json_loaded)
        for i in range(1, question_id_max):
            question_key = "question-" + str(i)
            answer_key = "answers-" + str(i)
            start_index_key = "start-" + str(i)
            property_key = "question-"+str(i)+"-property-input"
            end_answer_key = "end-" + str(i)
            if question_key in json_loaded:
                question = json_loaded[question_key]
                if "[PROPERTY]" in question:
                    question = question.replace("[PROPERTY]", json_loaded[property_key])
                curr_qa = QuestionAndAnswer(question, json_loaded[answer_key], json_loaded[start_index_key], json_loaded[end_answer_key])
                questions_and_answers.append(curr_qa)

    def create_one_new_df_row(self, curr_questions_and_answers, sentenceid, index, sentence):
        curr_questions_and_answers.sort(key=lambda x: x.start_index_answer, reverse=False)
        questions = [obj.question for obj in curr_questions_and_answers]
        questions = '&'.join(questions)
        answers = [obj.answer for obj in curr_questions_and_answers]
        answers = '&'.join(answers)
        ranges = [str(obj.start_index_answer) + '##' + str(obj.end_index_answer) for obj in curr_questions_and_answers]
        ranges = '&'.join(ranges)
        new_df_row = [sentenceid, index, sentence, questions, answers, ranges]
        return new_df_row




# csv_creator = CSVCreator("test_csv_for_validation/test_csv_validation1.csv")
# csv_creator.create_csv_files_for_validation("validation/validation_test_join.csv")

csv_creator2 = CSVCreator("validation_expert_batch/validation_input.csv")
csv_creator2.create_csv_files_for_validation("validation_expert_batch/validation_expert_batch.csv")
