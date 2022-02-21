import pandas as pd
import json

batch_result_file = '../batches/crowd_batch5/crowd_batch5_results.csv'
bonuses_file_path = '../batches/crowd_batch5/batch5_bonuses_with_assignment.csv'
bonuses_per_assignments_file_path = '../batches/crowd_batch3/batch3_bonuses_per_assignments_test.csv'

# batch_result_file = '../batches/4th_expert_batch/4th_expert_batch_results.csv'
# bonuses_file_path = '../batches/4th_expert_batch/4th_expert_batch_bonuses.csv'

batch_result_df = pd.read_csv(batch_result_file)
bonuses_df = pd.DataFrame([], columns=["worker_id", "assignment_id", "bonus_in_cents", "bonus_in_dollars"])

# all bonuses are in cents
SECOND_QA_BONUS = 5
THIRD_QA_BONUS = 4
FOURTH_TO_SIXTHS_QA_BONUS = 3


def initialize_bonus_df():
    worker_ids = batch_result_df['WorkerId'].drop_duplicates()
    bonuses_df["worker_id"] = worker_ids
    bonuses_df["bonus_in_cents"] = 0


def initialize_bonus_df_per_assignments():
    assignment_ids = batch_result_df['AssignmentId']
    bonuses_df["assignment_id"] = assignment_ids


def calculate_and_insert_bonuses_in_dollars():
    bonuses_df["bonus_in_dollars"] = bonuses_df["bonus_in_cents"] / 100


def calculate_and_insert_bonuses():
    initialize_bonus_df()
    for index, row in batch_result_df.iterrows():
        worker_id = row['WorkerId']
        assignment_id = row["AssignmentId"]
        qas = row['Answer.taskAnswers']
        total_bonus = calculate_bonus_for_hit(qas)
        bonuses_df.loc[bonuses_df["worker_id"] == worker_id, "assignment_id"] = assignment_id
        bonuses_df.loc[bonuses_df["worker_id"] == worker_id, "bonus_in_cents"] += total_bonus
    calculate_and_insert_bonuses_in_dollars()
    bonuses_df.to_csv(bonuses_file_path, index=False)


def calculate_bonus_per_assignment():
    initialize_bonus_df_per_assignments()
    for index, row in batch_result_df.iterrows():
        worker_id = row['WorkerId']
        assignment_id = row["AssignmentId"]
        qas = row['Answer.taskAnswers']
        total_bonus = calculate_bonus_for_hit(qas)
        bonuses_df.loc[bonuses_df["assignment_id"] == assignment_id, "worker_id"] = worker_id
        bonuses_df.loc[bonuses_df["assignment_id"] == assignment_id, "bonus_in_cents"] = total_bonus
    calculate_and_insert_bonuses_in_dollars()
    bonuses_df.to_csv(bonuses_per_assignments_file_path, index=False)


def calculate_bonus_for_hit(qas):
    qas_loaded = json.loads(qas)[0]
    num_questions = 0
    for output in qas_loaded:
        if "question-" in output:
            num_questions += 1
    total_bonus = 0
    if num_questions >= 2:
        total_bonus += SECOND_QA_BONUS
    num_questions -= 2
    if num_questions >= 1:
        total_bonus += THIRD_QA_BONUS
    num_questions -= 1
    max_remaining_qas_for_bonus = 3
    while num_questions > 0 and max_remaining_qas_for_bonus > 0:
        total_bonus += FOURTH_TO_SIXTHS_QA_BONUS
        num_questions -= 1
        max_remaining_qas_for_bonus -= 1
    return total_bonus

calculate_and_insert_bonuses()
# calculate_bonus_per_assignment()