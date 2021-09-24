class BaseConfig:
    PROJECT_NAME = 'MySampleProject'
    SUB_PROJECT_NAME = 'Sample_1'
    BASE_INPUT_PATH = r'../input/'
    BASE_OUTPUT_PATH = r'../output/'
    FILE_NAME = BASE_INPUT_PATH + r'train.csv'
    DATA_PATH = BASE_INPUT_PATH + FILE_NAME

    TARGET_COL = 'customer_category'
    FOLD_METHOD = 'StratifiedKFold'
    PROBLEM_TYPE = 'classification'

    cont_cols = ['customer_visit_score', 'customer_product_search_score',
                'customer_ctr_score', 'customer_stay_score', 'customer_frequency_score',
                'customer_product_variation_score', 'customer_order_score',
                'customer_affinity_score', ]

    cat_cols = ['customer_active_segment','X1']
