# vacancies_feature_pipeline
Repository contains implementation of the pipeline that can be used to generate test_proc.tsv. The implementation can be found in feat_eng_pipeline.py. It allows further extensions (as it was required) for new feature codes and other methods of normalization instead of Z-score.
This repo also contains test_features_pipeline.py with just general test for the pipeline. 
Provided data is stored in the directory /data.

To run the pipeline:
1) Use pytest to run the test and the results for test dataset will be automatically recorded to test_proc.tsv

2) Run script in test_features_pipeline.py without pytest and the results will be automatically recorded to test_proc.tsv as well.