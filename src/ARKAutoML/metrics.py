from operator import pos
from pprint import pprint
from sklearn import metrics
# from .logger import logger

def pretty_scores(scores):
    scores_list = []
    for k,v in scores.items():
        scores_list.append(f'{k.capitalize()} : {round(v, 3)}')
        # print(f'{k.capitalize()} : {round(v, 3)}', end= '  ')
    return '  '.join(scores_list)
    
def get_scores(y, predictions):
    accuracy = metrics.accuracy_score(y, predictions)
    precision_0 = metrics.precision_score(y,predictions, pos_label=0)
    recall_0 = metrics.recall_score(y,predictions, pos_label=0)
    f1_score_0 = metrics.f1_score(y,predictions, pos_label=0)

    precision_1 = metrics.precision_score(y,predictions, pos_label=1)
    recall_1 = metrics.recall_score(y,predictions, pos_label=1)
    f1_score_1 = metrics.f1_score(y,predictions, pos_label=1)
    return   {
                'accuracy' : accuracy,
                'precision_1' : precision_1,
                'recall_1' : recall_1,
                'f1_score_1' : f1_score_1,
                'precision_0' : precision_0,
                'recall_0' : recall_0,
                'f1_score_0' : f1_score_0,
    }

def model_metrics(model, X, y, logger, print_metrics=False):
    predictions = model.predict(X)
    scores = get_scores(y, predictions)

    try:
        pred_proba = model.predict_proba(X)
        roc_auc = metrics.roc_auc_score(y, pred_proba[:,1])
    except:
        roc_auc = 0
    scores['roc_auc'] =  roc_auc
    if print_metrics:
        logger.info(f'{pretty_scores(scores)}')
        logger.info(f'\n {metrics.confusion_matrix(y,predictions)}')
        logger.info(f'\n {metrics.classification_report(y,predictions)}')
    return scores