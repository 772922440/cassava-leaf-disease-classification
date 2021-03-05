# cassava-leaf-disease-classification
- The solution of cassava leaf disease classification 2021: Top 7%, Ranking 267/3900+.

## Link
https://www.kaggle.com/c/cassava-leaf-disease-classification/overview

## Install environment
- Use environment.yaml 

## Data2 source
- The Data2 is from the data in cassava leaf disease classification 2020

## Method
- Ensemble Model
  --  We use tf_efficientnet_b4 and resnext50_32x4d to ensemble
- Optimizer
  -- We use a lot of optimizer like SGD, Adam, AdamW, AdamP... The result shows that Adam and SGD is the best two.   
- Data Aug
  -- See in the src/

## In summary
- Using Ensemble and Data augmentation is the key.

## How to run the code?
```
git clone <this repository>
cd src/
see the detail in src/
```
