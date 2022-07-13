from typing import Callable, Iterable, List, NoReturn, Tuple, Union, Any, Dict, Optional 
import pandas as pd
import math
import json

from metrics import ManyToOneMetrics
from create_readable_csv_batches import RawCSVReader

'''
This file creates feedback files for each worker in a batch, in comparison to the expert batch. 
'''

QUESTIONS_DICT = {1: "What is the [PROPERTY] of (the) [W]?", 
                  2: "Whose [W]?", 
                  3: "Where is the [W]?",
                  4: "How much /How many [W]?",
                  5: "What is the [W] a part/member of?", 
                  6: "What/Who is a part/member of [W]?",
                  7: "What/Who is (the) [W]?",
                  8: "What kind of [W]?", 
                  9: "When is the [W]?"}

def rename_key(dict_or_df: Union[Dict[str, Any], pd.DataFrame], old_name: str, new_name: str): # -> NoReturn:
    dict_or_df[new_name] = dict_or_df.pop(old_name)

def modify_key_names(data: Union[pd.DataFrame, Dict], 
                     prefix='', suffix='', preserve_keys=[]): # -> NoReturn:
    " Return the data structure after modifying its keys with prefix/suffix"
    for key in data.copy():
        if key not in preserve_keys:
            rename_key(data, key, f"{prefix}{key}{suffix}")

def inverse_merge(orig_df: pd.DataFrame, drop: pd.DataFrame, on: Optional[List[str]] = None) -> pd.DataFrame:
    " Return all rows in `orig_df` not occuring in `drop` on `on` columns. "
    on = on or drop.columns
    return (orig_df.merge(drop[on], on=on, how='left', indicator=True)
              .query('_merge == "left_only"').drop(columns='_merge'))
        
def concat_dfs_with_renamed_columns(left: pd.DataFrame, right: pd.DataFrame, 
                                    lprefix='', 
                                    rprefix='',
                                    lsuffix='',
                                    rsuffix='', 
                                    preserve_columns=[]):
    """
    Concatenate two DataFrames, but modify column names with a unique prefix/suffix for each DF.
    :param preserve_columns: column names not to modify.
    """
    left = left.copy()
    right = right.copy()
    for df, prefix, suffix in zip((left, right), (lprefix, rprefix), (lsuffix, rsuffix)):
        modify_key_names(df, prefix=prefix, suffix=suffix, preserve_keys=preserve_columns)
    return pd.concat((left, right))

def drop_duplicate_rows(rows: Iterable[pd.Series]) -> List[pd.Series]:
    ret = []
    for row in rows:
        add_row = True
        for prev_row in ret:
            if (row==prev_row).all():
                add_row = False
        if add_row:
            ret.append(row)
    return ret

class NewFeedbackCalculator:
    "Ayal's re-implementation of FeedbackCalculator"
    
    IOU_THRESHOLD = 0.5
    
    def __init__(self, results_file, expert_file, workers_score_output_fn, workers_mistakes_output_fn):
        self.workers_annot_file = results_file
        self.expert_annot_file = expert_file
        self.workers_score_output_fn = workers_score_output_fn
        self.workers_mistakes_output_fn = workers_mistakes_output_fn
        
        self.workers_annot_df = RawCSVReader.read_annot_csv(self.workers_annot_file)
        self.expert_annot_df = RawCSVReader.read_annot_csv(self.expert_annot_file)


    @staticmethod
    def consolidated_expert_set(experts_df: pd.DataFrame) -> pd.DataFrame:
        # currently picking just one expert
        Ayal_worker_id = 'AZC7J87AH18DW'
        return experts_df[experts_df.worker_id == Ayal_worker_id]
    
    def prepare_reports(self):
        """
        Evaluate each worker against expert, prepare worker statistics report
        and an aggregated "mistakes" csv for semi-automatic review.  
        """
        reference_df = self.consolidated_expert_set(self.expert_annot_df) 
        worker_mistakes_dfs = []
        worker_statistics_dicts = []
        for worker_id, worker_df in self.workers_annot_df.groupby('worker_id'):
            worker_eval_results = self.evaluate_dataset(system=worker_df,
                                                        reference=reference_df)
            worker_ans_metric, worker_q_metric, worker_match_info = worker_eval_results
            # prepare worker statistics dict
            n_hits = worker_df['instance_id'].nunique()
            is_qa_row = worker_df['question'].astype(bool)
            n_qas = is_qa_row.sum()
            n_empty = worker_df[~is_qa_row]['instance_id'].nunique()
            worker_stats = {"worker_id": worker_id,
                            "recall": worker_ans_metric.recall(),
                            "precision": worker_ans_metric.prec(),
                            "f1": worker_ans_metric.f1(),
                            "question_recall": worker_q_metric.recall(),
                            "question_precision": worker_q_metric.prec(),
                            "question_f1": worker_q_metric.f1(),
                            "num_hits": n_hits,
                            "avg_qa_per_hit": n_qas / n_hits,
                            "num_empty_instances": n_empty
                                }
            worker_statistics_dicts.append(worker_stats)
            # prepare worker mistakes csv - join dataframes
            ref_ans_unmatched_df = worker_match_info['ref_ans_unmatched_df']
            rename_key(ref_ans_unmatched_df, 'worker_id', 'expert_id')
            ref_ans_unmatched_df['worker_id'] = worker_id
            ans_mistakes = concat_dfs_with_renamed_columns(
                worker_match_info['sys_ans_unmatched_df'],
                ref_ans_unmatched_df,
                lsuffix='_worker', rsuffix='_expert',
                preserve_columns=['instance_id', 'sentence', 'sentence_id', 'target_idx', 'target', 'worker_id', 'expert_id', 'general_comment']) 
            
            # take only question mistakes that are not answer mistakes - i.e. occur in answer_matches
            ans_match_df = worker_match_info['ans_matched_df']
            q_match_df = worker_match_info['q_matched_df']
            
            sys_q_unmatched_df = worker_match_info['sys_q_unmatched_df']
            matched_sys_qas_on_ans = worker_match_info['sys_ans_matched_df'][['instance_id', 'question_template', 'answer']]
            sys_q_unmatched_df = sys_q_unmatched_df.merge(matched_sys_qas_on_ans, 
                                                          on=['instance_id', 'question_template', 'answer'])
            ref_q_unmatched_df = worker_match_info['ref_q_unmatched_df']
            matched_ref_qas_on_ans = worker_match_info['ref_ans_matched_df'][['instance_id', 'question_template', 'answer']]
            ref_q_unmatched_df = ref_q_unmatched_df.merge(matched_ref_qas_on_ans, 
                                                          on=['instance_id', 'question_template', 'answer'])
            rename_key(ref_q_unmatched_df, 'worker_id', 'expert_id')
            ref_q_unmatched_df['worker_id'] = worker_id
            # for clarity of presentation, take question-mismtaches from the joined ans-match DF
            
            rename_key(sys_q_unmatched_df, "question_template", "question_template_worker")
            rename_key(sys_q_unmatched_df, "answer", "answer_worker")
            sys_unmatched_questions = sys_q_unmatched_df[['instance_id', 'question_template_worker', 'answer_worker']]
            sys_q_unmatched_ans_matched_df = ans_match_df.merge(sys_unmatched_questions, 
                                                    on=['instance_id', 'question_template_worker', 'answer_worker'])
            rename_key(ref_q_unmatched_df, "question_template", "question_template_expert")
            rename_key(ref_q_unmatched_df, "answer", "answer_expert")
            ref_unmatched_questions = ref_q_unmatched_df[['instance_id', 'question_template_expert', 'answer_expert']]
            ref_q_unmatched_ans_matched_df = ans_match_df.merge(ref_unmatched_questions, 
                                                    on=['instance_id', 'question_template_expert', 'answer_expert'])
            
            ques_mistakes1 = pd.concat([sys_q_unmatched_ans_matched_df, ref_q_unmatched_ans_matched_df]) 
            # ques_mistakes = ques_mistakes.drop_duplicates()
            """ Debug Log: 
            Perhaps the problem is that I'm merging from `ans_match_df` into unmatched_q rows, sys & ref, 
            when the latter can include rows that also have an exact question-match in `ans_match_df`. 
            I'm not accounting for many-to-one alignment here... 
            Perhaps a better way to tackle question-mismatch is to take all rows from `ans_match_df` 
            that do *not* occur in `q_match_df` 
            """
            ques_mistakes = inverse_merge(ans_match_df, q_match_df, on=['instance_id', 'question_worker', 'answer_worker'])
            
            # mark type of error
            ans_mistakes['mismatch_type'] = "answer" 
            ques_mistakes['mismatch_type'] = "question" 
            worker_mistakes = pd.concat([ans_mistakes, ques_mistakes])
            worker_mistakes_dfs.append(worker_mistakes)
        
        # save to workers mistakes file
        all_mistakes = pd.concat(worker_mistakes_dfs)
        all_mistakes.to_csv(self.workers_mistakes_output_fn, index=False)
        # save to workers statistics file
        all_statistics = pd.DataFrame(worker_statistics_dicts)
        all_statistics.to_csv(self.workers_score_output_fn, index=False, float_format='%.3f')
    
    @staticmethod
    def evaluate_dataset(system: pd.DataFrame, reference: pd.DataFrame):
        # keep only reference for those instances (predicates) that are annotated by system
        sys_instances = system['instance_id'].drop_duplicates()
        reference = pd.merge(reference, sys_instances, on='instance_id')
        # keep only system annotation for those instances (predicates) that are annotated by reference
        ref_instances = reference['instance_id'].drop_duplicates()
        system = pd.merge(system, ref_instances, on='instance_id')
        # init metrics and lists 
        answer_metric = ManyToOneMetrics.empty()
        question_metric = ManyToOneMetrics.empty()
        sys_ans_matched_rows = []
        ref_ans_matched_rows = []
        sys_ans_unmatched_rows = []
        ref_ans_unmatched_rows = []
        sys_q_matched_rows = []
        ref_q_matched_rows = []
        sys_q_unmatched_rows = []
        ref_q_unmatched_rows = []
        ans_matched_pairs = []
        q_matched_pairs = []
        def same_q_template(row1, row2):
            return row1.question_template == row2.question_template
        
        for instance_id, instance_sys_df in system.groupby('instance_id'):
            instance_ref_df = reference[reference['instance_id'] == instance_id]
            # first align by answer only, for 'answer_f1' metric
            ins_tp, ins_fp, ins_fn = NewFeedbackCalculator.find_instance_matches_and_mismatches(
                instance_sys_df, instance_ref_df)
            # split TPs to ref unique QA and sys unique QAs
            ins_tp_sys, ins_tp_ref = zip(*ins_tp) if ins_tp else ([], [])
            ins_tp_sys = drop_duplicate_rows(ins_tp_sys)
            ins_tp_ref = drop_duplicate_rows(ins_tp_ref)
            answer_metric += ManyToOneMetrics(len(ins_tp_sys), len(ins_tp_ref), 
                                              len(ins_fp), len(ins_fn))
            # keep answer mistakes and matches
            sys_ans_unmatched_rows.extend(ins_fp)
            ref_ans_unmatched_rows.extend(ins_fn)
            sys_ans_matched_rows.extend(ins_tp_sys)
            ref_ans_matched_rows.extend(ins_tp_ref)
            ans_matched_pairs.extend(ins_tp)
            # Then align by answer & question, for 'question_f1' metric
            ins_q_tp, ins_q_fp, ins_q_fn = NewFeedbackCalculator.find_instance_matches_and_mismatches(
                instance_sys_df, instance_ref_df, same_q_template)
            # split TPs to ref unique QA and sys unique QAs
            ins_q_tp_sys, ins_q_tp_ref = zip(*ins_q_tp) if ins_q_tp else ([], [])
            ins_q_tp_sys = drop_duplicate_rows(ins_q_tp_sys)
            ins_q_tp_ref = drop_duplicate_rows(ins_q_tp_ref)
                        
            question_metric += ManyToOneMetrics(len(ins_q_tp_sys), len(ins_q_tp_ref), 
                                                len(ins_q_fp), len(ins_q_fn))
            sys_q_unmatched_rows.extend(ins_q_fp)
            ref_q_unmatched_rows.extend(ins_q_fn)
            sys_q_matched_rows.extend(ins_q_tp_sys)
            ref_q_matched_rows.extend(ins_q_tp_ref)
            q_matched_pairs.extend(ins_q_tp)
            
        # aggregate rows to dataframes
        def rows_to_df(rows: List[pd.Series]) -> pd.DataFrame:
            return pd.DataFrame([row.to_dict() for row in rows])
        # aggregate matched row-pairs for dataframe
        def row_pairs_to_df(row_pairs: List[Tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
            if not row_pairs:
                return pd.DataFrame()
            dicts = []
            for sys_row, ref_row in row_pairs:
                sys_dict = sys_row.to_dict()               
                ref_dict = ref_row.to_dict()
                rename_key(ref_dict, 'worker_id', 'expert_id')
                modify_key_names(sys_dict, suffix='_worker', preserve_keys=['instance_id', 'worker_id', 'sentence_id', 'sentence', 'target', 'target_idx', 'general_comment'])
                modify_key_names(ref_dict, suffix='_expert', preserve_keys=['instance_id', 'expert_id', 'sentence_id', 'sentence', 'target', 'target_idx', 'general_comment'])               
                dicts.append(dict(sys_dict, **ref_dict))
            return pd.DataFrame(dicts)
            
        matches_info = dict(sys_ans_matched_df = rows_to_df(sys_ans_matched_rows),
                            ref_ans_matched_df = rows_to_df(ref_ans_matched_rows),
                            sys_ans_unmatched_df = rows_to_df(sys_ans_unmatched_rows),
                            ref_ans_unmatched_df = rows_to_df(ref_ans_unmatched_rows),
                            sys_q_matched_df = rows_to_df(sys_q_matched_rows),
                            ref_q_matched_df = rows_to_df(ref_q_matched_rows),
                            sys_q_unmatched_df = rows_to_df(sys_q_unmatched_rows),
                            ref_q_unmatched_df = rows_to_df(ref_q_unmatched_rows),
                            # further "merged" match-DFs for that join matching annotation on row level (from `ins_tp` & `ins_q_tp`)
                            ans_matched_df = row_pairs_to_df(ans_matched_pairs),
                            q_matched_df = row_pairs_to_df(q_matched_pairs)
                            )
        
        return answer_metric, question_metric, matches_info    
    
    @staticmethod
    def evaluate_instance(instance_sys_df: pd.DataFrame, instance_ref_df: pd.DataFrame) -> Tuple[int, int, int]:
        tp, fp, fn = NewFeedbackCalculator.find_instance_matches_and_mismatches(instance_sys_df, instance_ref_df)
        return len(tp), len(fp), len(fn)
    
    @staticmethod
    def find_instance_matches_and_mismatches(system: pd.DataFrame, reference: pd.DataFrame,
                                             question_match_criterion: Optional[Callable[[pd.Series, pd.Series], bool]] = None):
        """ 
        Evaluates annotation per one instance (target noun) against reference annotation.
        Returns (true_positives, false_positives, false_negatives), 
         where the true-positive matches are returned as (sys_qa_row, ref_qa_row) pairs, 
         and the mismatches are lists of annotation rows (pd.Series) :
             
        """
        question_match_criterion = question_match_criterion or (lambda r1,r2: True)
        def is_no_qa_row(row) -> bool:
            return not row.question
        reference_copy = reference.copy()   # use a copy to mark "matched"/"unmatched" on ref QAs
        reference_copy['matched'] = False
        tp, fp = [], []
        for _, sys_qa_row in system.iterrows():
            sys_qa_matched = False
            for row_id, ref_qa_row in reference.iterrows():
                # handle empty instances (no-QA row)
                if is_no_qa_row(ref_qa_row):
                    reference_copy.loc[row_id, 'matched'] = True # don't count this ref_row as FN
                    continue
                
                if is_no_qa_row(sys_qa_row):
                    # ref QA is a FN, caught by "mathced" column being False
                    continue
                # if no row is empty, check for answer match
                if NewFeedbackCalculator.compute_iou(sys_qa_row.answer_range, ref_qa_row.answer_range) \
                        >= NewFeedbackCalculator.IOU_THRESHOLD:
                    # Check for question match 
                    if question_match_criterion(sys_qa_row, ref_qa_row):
                        sys_qa_matched = True
                        reference_copy.loc[row_id, 'matched'] = True
                        tp.append((sys_qa_row, ref_qa_row))
                
            if not sys_qa_matched:
                fp.append(sys_qa_row)
        fn = reference_copy[~reference_copy['matched']]
        fn.pop('matched')
        fn = [row for _,row in fn.iterrows()]
        
        return tp, fp, fn
          
    @staticmethod
    def compute_iou(range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
        indices1 = set(range(*range1))
        indices2 = set(range(*range2))
        return len(indices1.intersection(indices2)) / len(indices1.union(indices2))
        

'''
Usage example: need to specify all paths and run the function that creates all the analysis files
'''
if __name__ == "__main__":
    batch = "batch2"
    # input files
    batch_results_file = f"../training/{batch}/crowd_{batch}_annot.csv"
    expert_results_file = f"../training/{batch}/expert_{batch}_annot.csv"
    # output files
    workers_score_output_fn = f"../training/{batch}/worker_score_{batch}.csv"
    workers_mistakes_output_fn = f"../training/{batch}/worker_mistakes_{batch}.csv"

    feedback_calculator1 = NewFeedbackCalculator(batch_results_file, expert_results_file, 
                                            workers_score_output_fn, workers_mistakes_output_fn)
    feedback_calculator1.prepare_reports()
