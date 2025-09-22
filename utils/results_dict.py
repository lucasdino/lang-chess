import math

from .exceptions import ParseException, IllegalMoveException
from .parsing import extract_solution, coerce_response


class ResultsDict():
    """
    Class that aggregates result metrics for a given evaluation type. Logs and writes results to wandb if provided.

    task_type: One of 'choose_from_n', 'produce_list', or 'predict_singlemove'.
    filename: The filename being evaluated (for logging purposes).
    wandb_run: Optional wandb run object for logging.
    """
    def __init__(self, task_type, filename, wandb_run = None):
        self.task_type = task_type
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.results = self._instantiate_dict()
        self.correct_responses = []
        self.produce_list_threshold = 0.6    # IoU threshold to call a produce_list response 'correct' for rej sampling
        self.predict_answer_threshold = 0.7  # Threshold to call a predict_singlemove response 'correct' for rej sampling

    def add_result(self, prompt, model_response, info):
        try:
            self.results["Total Samples"] += 1
            ground_truth = info['answer']
            if self.task_type == "choose_from_n":
                answer = ground_truth['answer']
                candidates = ground_truth['candidates']
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                
                # Determine correctness for rej sampling / logging
                if predicted_answer == answer:
                    self.results["Correct"] += 1
                    self.correct_responses.append({
                        "prompt": prompt,
                        "model_response": model_response,
                        "info": info
                    })
                else:
                    if predicted_answer in candidates:
                        self.results["Incorrect"] += 1
                    else:
                        raise IllegalMoveException("Predicted move is not in the provided moves.")
            
            elif self.task_type == 'produce_list':
                answer = ground_truth
                self.results["Total Ground Truth Legal Moves"] += len(answer)
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)

                # Compute correctness. We can compute final IoU score using 'Total Ground Truth Legal Moves', 
                # 'Predicted Ground Truth Legal Moves', and 'Illegal Moves' later.
                num_right = 0
                already_guessed = set()
                for move in predicted_answer:
                    if move in answer and move not in already_guessed:
                        already_guessed.add(move)
                        num_right += 1
                        self.results["Predicted Ground Truth Legal Moves"] += 1
                    else:
                        self.results["Illegal Moves"] += 1

                # Keep for rej sampling if above a certain threshold
                score_iou = num_right / (len(answer) + len(predicted_answer) - num_right)
                if score_iou >= self.produce_list_threshold:
                    self.correct_responses.append({
                        "prompt": prompt,
                        "model_response": model_response,
                        "info": info
                    })
                
            elif self.task_type == 'predict_singlemove':
                answer = ground_truth
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                sorted_answers = sorted(answer.items(), key=lambda x: x[1])
                
                if predicted_answer in answer:
                    self.results["Legal Moves Provided"] += 1
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_answers) if move == predicted_answer)
                    self.results["Cumulative Rank of Moves Provided"] += predicted_move_idx/len(sorted_answers)

                    # Keep for rej sampling if above a certain threshold
                    if predicted_move_idx/len(sorted_answers) > self.predict_answer_threshold:
                        self.correct_responses.append({
                            "prompt": prompt,
                            "model_response": model_response,
                            "info": info
                        })
                else:
                    raise IllegalMoveException("Predicted move is not in the legal moves.")
        
        # Exception handling to log various errors     
        except Exception as e:
            if isinstance(e, ParseException):
                self.results["Error: Parsing"] += 1
            elif isinstance(e, IllegalMoveException):
                self.results["Error: Illegal Move"] += 1
            else:
                self.results["Error: Other"] += 1
        
    def get_final_dict(self, run_type):
        """ run_type is either 'eval' or 'rejsampling' -- used for wandb logging. """
        run_type = run_type.capitalize()

        if self.task_type == "choose_from_n":
            total = self.results["Total Samples"]
            self.results["Accuracy"] = self._safe_div(self.results["Correct"], total)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Accuracy": self.results["Accuracy"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"],
                })

        elif self.task_type == "produce_list":
            self.results["Percent Legal Moves Predicted"] = self._safe_div(self.results["Predicted Ground Truth Legal Moves"], self.results["Total Ground Truth Legal Moves"])
            self.results["Ratio of Legal to Illegal Moves"] = self._safe_div(self.results["Predicted Ground Truth Legal Moves"], self.results["Illegal Moves"])
            self.results["Intersection Over Union"] = self._safe_div(self.results["Predicted Ground Truth Legal Moves"], (self.results["Total Ground Truth Legal Moves"] + self.results["Illegal Moves"]))
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Other'], self.results["Total Samples"])
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Percent Legal Moves Predicted": self.results["Percent Legal Moves Predicted"],
                    f"{run_type} - {self.trimmed_filename}/Ratio of Legal to Illegal Moves": self.results["Ratio of Legal to Illegal Moves"],
                    f"{run_type} - {self.trimmed_filename}/IoU": self.results["Intersection Over Union"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        elif self.task_type == "predict_singlemove":
            legal = self.results["Legal Moves Provided"]
            total = self.results["Total Samples"]
            self.results["Avg. Rank of Move Provided"] = self._safe_div(self.results["Cumulative Rank of Moves Provided"], legal)
            self.results["Percent Legal Moves Provided"] = self._safe_div(legal, total)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Avg. Rank of Move Provided": self.results["Avg. Rank of Move Provided"],
                    f"{run_type} - {self.trimmed_filename}/Percent Legal Moves Provided": self.results["Percent Legal Moves Provided"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        return self.results, self.correct_responses

    # =================================================
    # Internal helper functions
    # =================================================
    def _instantiate_dict(self):
        if self.task_type == "choose_from_n":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Correct": 0,
                "Incorrect": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "produce_list":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Total Ground Truth Legal Moves": 0,
                "Predicted Ground Truth Legal Moves": 0,
                "Illegal Moves": 0,
                "Error: Parsing": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "predict_singlemove":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Legal Moves Provided": 0,
                "Cumulative Rank of Moves Provided": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        else:
            raise ValueError(f"Undefined task type: {self.task_type}")

    def _safe_div(self, x, y, default=0, default_div=None):
        if default_div is not None:
            # Use default_div if y is "close to zero"
            return x / y if not math.isclose(y, 0) else x / default_div
        else:
            return x / y if not math.isclose(y, 0) else default
    


# =============================================
# Results Dict for LLM Parsing Cases
# =============================================
class ParserResultsDict():
    """
    Class that aggregates result metrics for a given parsing evaluation type (e.g., hallucinations, reasoning strategies, reasoning quality). Logs and writes results to wandb if provided.
    
    Args:
    task_type: One of 'hallucination', 'reasoning_strategy', or 'reasoning_quality'.
    filename: The filename being evaluated (for logging purposes).
    wandb_run: Optional wandb run object for logging.
    """
    def __init__(self, task_type, filename, wandb_run = None):
        self.task_type = task_type
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.results = self._instantiate_dict()

    def add_result(self, parsed_response):
        self.results["Total Responses Parsed"] += 1
        if self.task_type == "hallucination":
            for k, v in parsed_response.items():
                self.results[k] += v

        elif self.task_type == "reasoning_strategy":
            for k, v in parsed_response.items():
                self.results[f"Count: {k}"] += v
        
        elif self.task_type == "reasoning_quality":
            for k, v in parsed_response.items():
                self.results[f"Sum: Reasoning {k}"] += v

    def get_final_dict(self):
        """ Return finalized dict and log to wandb. """
        if self.task_type == "hallucination":
            total_moves = self.results['Count: Moves Checked'] + self.results['Count: Pieces Checked']
            average_moves_per_response = self._safe_div(total_moves, self.results['Total Responses Parsed'])
            moves_accuracy = self._safe_div(self.results['Count: Moves Correct'], self.results['Count: Moves Checked'])
            pieces_accuracy = self._safe_div(self.results['Count: Pieces Correct'], self.results['Count: Pieces Checked'])
            total_accuracy = self._safe_div(self.results['Count: Moves Correct'] + self.results['Count: Pieces Correct'], total_moves)
            hallucination_percent = self._safe_div(self.results['Count: Hallucinations'],  total_moves)
            percent_reprompts = self._safe_div(self.results['Error: Reprompt'], self.results['Total Responses Parsed'])

            self.results['Moves Accuracy'] = moves_accuracy
            self.results['Pieces Accuracy'] = pieces_accuracy
            self.results['Total Accuracy'] = total_accuracy
            self.results['Hallucination Percent'] = hallucination_percent
            self.results['Ave. Moves Parsed Per Response'] = average_moves_per_response
            self.results['Percent Reprompts'] = percent_reprompts
            
            if self.wandb_run:
                self.wandb_run.log({
                    f"Hallucination / Moves Accuracy": self.results['Moves Accuracy'],
                    f"Hallucination / Pieces Accuracy": self.results['Pieces Accuracy'],
                    f"Hallucination / Total Accuracy": self.results['Total Accuracy'],
                    f"Hallucination / Hallucination Percent": self.results["Hallucination Percent"],
                    f"Hallucination / Ave. Moves Parsed Per Response": self.results["Ave. Moves Parsed Per Response"],
                    f"Hallucination / Percent Reprompts": self.results["Percent Reprompts"]                
                })
        
        elif self.task_type == "reasoning_strategy":
            self.results['Percent Enumeration'] = self._safe_div(self.results['Count: Enumeration'], self.results['Total Responses Parsed'])
            self.results['Percent Tree Search'] = self._safe_div(self.results['Count: Tree Search'], self.results['Total Responses Parsed'])
            self.results['Percent Backtracking'] = self._safe_div(self.results['Count: Backtracking'], self.results['Total Responses Parsed'])
            self.results['Percent Self Correction'] = self._safe_div(self.results['Count: Self Correction'], self.results['Total Responses Parsed'])
            self.results['Percent Subgoal Setting'] = self._safe_div(self.results['Count: Subgoal Setting'], self.results['Total Responses Parsed'])
            self.results['Percent Verification'] = self._safe_div(self.results['Count: Verification'], self.results['Total Responses Parsed'])
            self.results['Percent Reprompts'] = self._safe_div(self.results['Error: Reprompt'], self.results['Total Responses Parsed'])
            
            if self.wandb_run:
                self.wandb_run.log({
                    f"Reasoning Strategy / Percent Enumeration": self.results["Percent Enumeration"],
                    f"Reasoning Strategy / Percent Tree Search": self.results["Percent Tree Search"],
                    f"Reasoning Strategy / Percent Backtracking": self.results["Percent Backtracking"],
                    f"Reasoning Strategy / Percent Self Correction": self.results["Percent Self Correction"],
                    f"Reasoning Strategy / Percent Subgoal Setting": self.results["Percent Subgoal Setting"],
                    f"Reasoning Strategy / Percent Verification": self.results["Percent Verification"],
                    f"Reasoning Strategy / Percent Reprompts": self.results["Percent Reprompts"],
                })
        
        elif self.task_type == "reasoning_quality":
            self.results['Avg. Reasoning Efficacy'] = self._safe_div(self.results['Sum: Reasoning Efficacy'], self.results['Total Responses Parsed'])
            self.results['Avg. Reasoning Efficiency'] = self._safe_div(self.results['Sum: Reasoning Efficiency'], self.results['Total Responses Parsed'])
            self.results['Avg. Reasoning Faithfulness'] = self._safe_div(self.results['Sum: Reasoning Faithfulness'], self.results['Total Responses Parsed'])
            reasoning_score_all = self.results['Sum: Reasoning Efficacy'] + self.results['Sum: Reasoning Efficiency'] + self.results['Sum: Reasoning Faithfulness']
            self.results['Avg. Reasoning All'] = self._safe_div(reasoning_score_all, self.results['Total Responses Parsed'])
            self.results['Percent Reprompts'] = self._safe_div(self.results['Error: Reprompt'], self.results['Total Responses Parsed'])
            
            if self.wandb_run:
                self.wandb_run.log({
                    f"Reasoning Quality / Avg. Reasoning Efficacy": self.results["Avg. Reasoning Efficacy"],
                    f"Reasoning Quality / Avg. Reasoning Efficiency": self.results["Avg. Reasoning Efficiency"],
                    f"Reasoning Quality / Avg. Reasoning Faithfulness": self.results["Avg. Reasoning Faithfulness"],
                    f"Reasoning Quality / Avg. Reasoning All": self.results["Avg. Reasoning All"],
                    f"Reasoning Quality / Percent Reprompts": self.results["Percent Reprompts"],
                })

        return self.results


    # =================================================
    # Internal helper functions
    # =================================================
    def _instantiate_dict(self):
        if self.task_type == "hallucination":
            return {
                "Filename": self.filename,
                "Total Responses Parsed": 0,
                "Count: Moves Checked":  0,
                "Count: Moves Correct":  0,
                "Count: Pieces Checked": 0,
                "Count: Pieces Correct": 0,
                "Count: Hallucinations": 0,
                "Error: Reprompt": 0,
                "Error: Other": 0
            }
        elif self.task_type == "reasoning_strategy":
            return {
                "Filename": self.filename,
                "Total Responses Parsed": 0,
                "Count: Enumeration": 0,
                "Count: Tree Search": 0,
                "Count: Backtracking": 0,
                "Count: Self Correction": 0,
                "Count: Subgoal Setting": 0,
                "Count: Verification": 0,
                "Error: Reprompt": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "reasoning_quality":
            return {
                "Filename": self.filename,
                "Total Responses Parsed": 0,
                "Sum: Reasoning Efficacy": 0,
                "Sum: Reasoning Efficiency": 0,
                "Sum: Reasoning Faithfulness": 0,
                "Error: Reprompt": 0,
                "Error: Other": 0,
            }
        else:
            raise ValueError(f"Undefined task type: {self.task_type}")

    def _safe_div(self, x, y, default=0): 
        return x / y if y else default