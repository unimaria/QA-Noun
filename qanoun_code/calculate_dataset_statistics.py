import pandas as pd
from datetime import timedelta
from statistics import median
from create_csv_for_mturk_batch import process_file, preprocess_csv, preprocess_json, tqa_file_path, wiki_file_path, wikinews_file_path
from create_readable_csv_batches import RawCSVReader

target_folder_path = "../expenses/dataset_statistics"
NUM_SENTENCES = 200


def create_instances_file_using_json(dataset_path, base_filename, num_sentences):
    preprocess_json(dataset_path, target_folder_path + base_filename + "_sentences.txt", num_sentences)
    process_file(target_folder_path + base_filename + "_sentences.txt", target_folder_path + base_filename + "_instances.csv")


def create_instances_file_using_csv(dataset_path, base_filename, num_sentences):
    preprocess_csv(dataset_path, target_folder_path + base_filename + "_sentences.txt", num_sentences)
    process_file(target_folder_path + base_filename + "_sentences.txt", target_folder_path + base_filename + "_instances.csv")


def create_tqa_file(num_sentences):
    create_instances_file_using_json(tqa_file_path, "/tqa", num_sentences)


def create_wiki_file(num_sentences):
    create_instances_file_using_csv(wiki_file_path, "/wiki", num_sentences)


def create_wikinews_file(num_sentences):
    create_instances_file_using_csv(wikinews_file_path, "/wikinews", num_sentences)


def count_sentences_to_instances_ratio():
    create_tqa_file(NUM_SENTENCES)
    create_wiki_file(NUM_SENTENCES)
    create_wikinews_file(NUM_SENTENCES)
    count_sentences_to_instances_ratio_without_creation()


def count_sentences_to_instances_ratio_without_creation():
    tqa_df = pd.read_csv(target_folder_path + "/tqa_instances.csv")
    wiki_df = pd.read_csv(target_folder_path + "/wiki_instances.csv")
    wikinews_df = pd.read_csv(target_folder_path + "/wikinews_instances.csv")
    num_instances_tqa = tqa_df.shape[0]
    num_instances_wiki = wiki_df.shape[0]
    num_instances_wikinews = wikinews_df.shape[0]
    print("\n=============================")
    print("TQA - avg instance number from a sentence: " + str(num_instances_tqa / NUM_SENTENCES))
    print("Wikipedia - avg instance number from a sentence: " + str(num_instances_wiki / NUM_SENTENCES))
    print("Wiki-news - avg instance number from a sentence: " + str(num_instances_wikinews / NUM_SENTENCES))
    print("Total avg instance number from a sentence: " +  \
          str((num_instances_tqa + num_instances_wiki + num_instances_wikinews) / (NUM_SENTENCES * 3)))


def count_question_to_assignments_ratio(batch_result_file):
    readable_creator = RawCSVReader(batch_result_file, target_folder_path + "/readable_batch_csv.csv")
    readable_creator.create_readable_csv()
    hits_df = pd.read_csv(batch_result_file)
    questions_df = pd.read_csv(target_folder_path + "/readable_batch_csv.csv")
    num_hits = hits_df.shape[0]
    num_questions = questions_df.shape[0]
    print("Avg question per assignment: " + str(num_questions / num_hits))


def calc_avg_expenses_per_assignment_4th_crowd_batch():
    batch_df = pd.read_csv("../batches/crowd_batch4/crowd_batch4_results.csv")
    num_assignments = batch_df.shape[0]
    price_per_assignment = 8
    bonuses_df = pd.read_csv("../batches/crowd_batch4/batch4_bonuses_with_assignment.csv")
    total_bonuses = bonuses_df['bonus_in_cents'].sum()
    print("Total money per assignment 4th crowd batch (dollars): " \
          + str((num_assignments*price_per_assignment + total_bonuses) / num_assignments / 100))

def calc_avg_expenses_per_assignment_5th_crowd_batch():
    batch_df = pd.read_csv("../batches/crowd_batch5/crowd_batch5_results.csv")
    num_assignments = batch_df.shape[0]
    price_per_assignment = 10
    bonuses_df = pd.read_csv("../batches/crowd_batch5/batch5_bonuses_with_assignment.csv")
    total_bonuses = bonuses_df['bonus_in_cents'].sum()
    print("Total money per assignment 5th crowd batch (dollars): " \
          + str((num_assignments*price_per_assignment + total_bonuses) / num_assignments / 100))

def calc_avg_expenses_per_assignment_3rd_crowd_batch():
    batch_df = pd.read_csv("../batches/crowd_batch3/crowd_batch3_results.csv")
    num_assignments = batch_df.shape[0]
    price_per_assignment = 8
    bonuses_df = pd.read_csv("../batches/crowd_batch3/batch3_bonuses_with_assignment.csv")
    total_bonuses = bonuses_df['bonus_in_cents'].sum()
    print("Total money per assignment 3rd crowd batch (dollars): " \
          + str((num_assignments*price_per_assignment + total_bonuses) / num_assignments / 100))


def calculate_time_diff_in_hours(start_time, end_time):
    start_time_split = start_time.split(":")
    end_time_split = end_time.split(":")
    start_delta = timedelta(hours=int(start_time_split[0]), minutes=int(start_time_split[1]), seconds=int(start_time_split[2]))
    end_delta = timedelta(hours=int(end_time_split[0]), minutes=int(end_time_split[1]), seconds=int(end_time_split[2]))
    diff_time = end_delta - start_delta
    return diff_time / timedelta(hours=1)


def calculate_salary_per_hour_3rd_crowd_batch():
    bonuses_batch_df = pd.read_csv("../batches/crowd_batch3/batch3_bonuses_per_assignments_test.csv")
    batch_df = pd.read_csv("../batches/crowd_batch3/crowd_batch3_results.csv")
    price_per_assignment = 11
    salary_per_hour_for_assignment = []
    time_per_assignment = []
    for index, row in batch_df.iterrows():
        assignment_id = row["AssignmentId"]
        time_took_hours = calculate_time_diff_in_hours(row["AcceptTime"].split()[3], row["SubmitTime"].split()[3])
        money_paid = int(bonuses_batch_df.loc[bonuses_batch_df["assignment_id"] == assignment_id, "bonus_in_cents"]) + price_per_assignment
        salary_per_hour_for_assignment.append(money_paid / time_took_hours / 100)
        time_per_assignment.append(time_took_hours)
    print("Salary per hour for assignment in dollars: " + str(salary_per_hour_for_assignment))
    print("Median: " + str(median(salary_per_hour_for_assignment)))
    print("Average: " + str(sum(salary_per_hour_for_assignment) / len(salary_per_hour_for_assignment)))
    print("Avg time: " + str(median(time_per_assignment)))
    print("Median time: " + str(sum(time_per_assignment) / len(time_per_assignment)))



# count_sentences_to_instances_ratio_without_creation()
# count_question_to_assignments_ratio("../batches/crowd_batch2/crowd_batch2_results.csv")
# calc_avg_expenses_per_assignment_3rd_crowd_batch()
# calculate_salary_per_hour_3rd_crowd_batch()
calc_avg_expenses_per_assignment_5th_crowd_batch()
calc_avg_expenses_per_assignment_4th_crowd_batch()
calc_avg_expenses_per_assignment_3rd_crowd_batch()