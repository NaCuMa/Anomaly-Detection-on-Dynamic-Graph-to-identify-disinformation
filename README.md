# Anomaly-Detection-on-Dynamic-Graph-to-identify-disinformation

## Initial setting

### Downloads

Download TGN (https://github.com/twitter-research/tgn) and TADDY (https://github.com/yixinliu233/TADDY_pytorch) repositories in the same folder.
Rename them in 'TGN' and 'TADDY'.

### Files to replace

Replace the following files by those provided in the present repository:
 - For TGN:
   - 'train_self_supervised.py';
   - 'utils\preprocess_data.py';
   - 'utils\data_processing.py';
   - 'model\tgn.py';
   - 'evaluation\evaluation.py';
   - 'modules\embedding_module.py'.
 - For TADDY:
   - '0_prepare_data.py';
   - '1_train.py';
   - 'codes\AnomalyGeneration.py'.

### Files to add

Add the following files:
- For TGN:
   - 'TGN_TADDY.ipynb';
   - 'TGN_test.ipynb';
   - 'Reliability_module.ipynb';
   - 'Synthetic_dataset.ipynb'.
 - For TADDY:
   - 'data\raw\email-dnc.csv';
   - 'data\raw\AST'.

## Running the codes

### Prepare data

- For a benchmark dataset: run the TGN\TGN_TADDY notebook, choose a dataset on the second cell and execute all. Get the last training instant, printed by the last cell.
- For a synthetic dataset: run the Synthetic_dataset notebook, choose parameters on the second cell and execute all.

Preprocess data according to https://github.com/twitter-research/tgn, e.g.: python utils/preprocess_data.py --data uci_TADDY_005_nop

### TGN training

Proceed as explained in https://github.com/twitter-research/tgn and, in the case of benchmark dataset, indicate the final time of training, obtained at the previous step e.g.:
- Benchmark dataset: python train_self_supervised.py -d uci_TADDY_005_nop --use_memory --prefix tgn-attn --val_time 2946672.0
- Synthetic: python train_self_supervised.py -d synthetic --use_memory --prefix tgn-attn

### Anomaly detection

Open the TGN_test notebook, choose a dataset on the second cell and execute all.
The results are displayed in the last part of the notebook.

### Explainability module

Open the Explainability_module notebook, choose a dataset on the second cell and execute all.
The results are displayed in the last part of the notebook.
