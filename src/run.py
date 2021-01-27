import os

os.system("python3 ensemble_train.py name=train1 k=4 criterion='LabelSmoothing' save_filename='res_labelsmoothing'") 
os.system("python3 ensemble_train.py name=train1 k=4 criterion='FocalLoss' save_filename='res_FocalLoss'") 
os.system("python3 ensemble_train.py name=train1 k=4 criterion='FocalCosineLoss' save_filename='res_FocalCosine'")  
os.system("python3 ensemble_train.py name=train1 k=4 criterion='SymmetricCrossEntropyLoss' save_filename='res_SymmetricCrossEntropy'")
os.system("python3 ensemble_train.py name=train1 k=4 criterion='BiTemperedLoss' save_filename='res_BiTempered'")
os.system("python3 ensemble_train.py name=train1 k=4 criterion='TaylorCrossEntropyLoss' save_filename='res_TaylorCrossEntropyLoss'")

