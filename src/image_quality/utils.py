from  typing import Dict, Tuple


def check_and_normalise_score(
    score: float,
    score_range: Tuple[float], 
    lower_better: bool,
):
    in_left_band = score >= score_range[0]
    in_right_band = score <= score_range[1]

    assert in_left_band and in_right_band
    
    norm_score = (score - score_range[0]) / (score_range[1] - score_range[0])
    if lower_better:
        norm_score = 1 - norm_score
        lower_better = False
    return norm_score


def parse_score_range(model_name: str, pyiqa_models_configs: Dict):
    score_range = pyiqa_models_configs[model_name]['score_range']

    if 'lower_better' in pyiqa_models_configs[model_name].keys():
        lower_better = pyiqa_models_configs[model_name]['lower_better']
    else:
        lower_better = False

    score_range = score_range.split(',')
    if '~' in score_range[0]:
        left_approx =  True
        score_range[0] = score_range[0].replace('~', '')
    else:
        left_approx =  False

    if '~' in score_range[1]:
        right_approx =  True

        if score_range[1] == ' ~':
            raise ValueError("Score range is undefined")
        else:
            score_range[1] = score_range[1].replace('~', '')
    else:
        right_approx =  False
    
    score_range = tuple(map(float, score_range))
    return score_range, lower_better, left_approx, right_approx