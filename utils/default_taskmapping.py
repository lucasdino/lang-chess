# Defining defaults for tasks / etc.
TASK_MAP = {
    'bestmove': "choose_from_n",
    'worstmove': "choose_from_n",
    'legalmoves': "produce_list",
    'predictmove': "predict_singlemove",
    'blunder_explanations': "synthetic_generation",
    'goodmove_explanations': "synthetic_generation",
    'reasonablemove_explanations': "synthetic_generation",
}

RUNTYPE_SYSPROMPT_MAPPING = {
    'hallucination': 'hallucinations_sysprompt.txt', 
    'reasoning_strategy': 'reasoning_strategies_sysprompt.txt',
    'reasoning_quality': 'reasoning_quality_sysprompt.txt',
}