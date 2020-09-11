# midl2020-cnnlstm-ich
CNN-LSTM for intracranial hemorrhage detection

# Re-produce step

- 1. run `process_ct.py` to convert original dicom to 3 view
``` 
# brain
python3 process_ct.py --window-level 40 --window-width 80 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
python3 process_ct.py --window-level 40 --window-width 80 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'
 
# subdural
python3 process_ct.py --window-level 100 --window-width 300 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
python3 process_ct.py --window-level 100 --window-width 300 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'

# bone
python3 process_ct.py --window-level 600 --window-width 2800 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
python3 process_ct.py --window-level 600 --window-width 2800 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'
```

- 2. 

# Pretrained model

```

```