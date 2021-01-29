#######################################

# This is for testing the opti

#######################################

# Try SGD or Ranger

echo "###################  Begin    ###############################"
python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=AdamW backbone=tf_efficientnet_b3
echo "###################  AdamW    ###############################"

python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=SGD backbone=tf_efficientnet_b3
echo "###################  SGD    ###############################"

python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=Ranger backbone=tf_efficientnet_b3
echo "####################    Ranger   ###################################"
#Try p

python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=AdamW backbone=tf_efficientnet_b3 p=0.1
echo "######################  p=0.1   #####################################"

python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=AdamW backbone=tf_efficientnet_b3 p=0.4
echo "######################  p=0.4   #####################################"

python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=AdamW backbone=tf_efficientnet_b3 p=0.6
echo "######################## p=0.6   ####################################"

python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=AdamW backbone=tf_efficientnet_b3 p=0.3
echo "######################## p=0.3  ####################################"

python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=AdamW backbone=tf_efficientnet_b3 p=0.7
echo "######################## p=0.7   ####################################"

python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=32 optimizer=AdamW backbone=tf_efficientnet_b3 p=0.8
echo "######################## p=0.8   ####################################"