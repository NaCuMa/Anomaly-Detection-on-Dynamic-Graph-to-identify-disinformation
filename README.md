# Anomaly-Detection-on-Dynamic-Graph-to-identify-disinformation

## Initial setting

### Downloads

Download the 'code' file of the present repository. Codes from TGN (https://github.com/twitter-research/tgn) and TADDY (https://github.com/yixinliu233/TADDY_pytorch) repositories are reused here for the sake of compatibility.
Compare to the original repositories, the following files were replaced or added:
 - For TGN:
   - 'train_self_supervised.py';
   - 'utils\preprocess_data.py';
   - 'utils\data_processing.py';
   - 'model\tgn.py';
   - 'evaluation\evaluation.py';
   - 'modules\embedding_module.py';
   - 'TGN_TADDY.ipynb'.
 - For TADDY:
   - '0_prepare_data.py';
   - '1_train.py';
   - 'codes\AnomalyGeneration.py';
   - 'data\raw\email-dnc.csv';
   - 'data\raw\AST'.

## Running the codes

Open the TGN\Main_notebook.ipynb notebook. Run all to test it on the btc_alpha benchmark dataset with 5% of anomalies. Results are available in the Anomaly detection\Results and Not to modify: reliability module\Results sections.

If you want to try another benchmark dataset, three cells need to be updated :
 - In Prepare dataset\Benchmark\To do: choice of the dataset, you need to indicate two times the name of the chosen dataset (uci, digg, email, btc_alpha, btc_otc or AST);
 - In TGN training\To do: for benchmark datasets, you need to indicate the name of the chosen dataset and the anomaly proportion with a specific format, e.g. for btc_alpha and 5% of anomalies: btc_alpha_TADDY_005_nop, you need also to precise the last instant of training obtained in at the end Prepare dataset\Benchmark (otherwise these values are given at the end of this readme);
 - In Anomaly detection\ To do: choice of the dataset, you need to specify the chosen dataset with the aforementionned format.

If you want to test the method with the Synthetic dataset, you need to decomment the cells allowing its generation in Prepare dataset\Synthetic, the training of TGN on  it in TGN training\To do: for synthetic datasets and to use 'Synthetic' as dataset name everywhere.
