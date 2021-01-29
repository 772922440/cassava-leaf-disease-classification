#######################################

# This is for testing the opti

#######################################

# Try SGD or Ranger
echo "###################  SGD    ###############################"
python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=20 optimizer=SGD backbone=mobilenetv3_large_100
echo "####################    Ranger   ###################################"
python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=20 optimizer=SGD backbone=mobilenetv3_large_100

#Try p
echo "######################  p=0.04   #####################################"
python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=20 optimizer=SGD backbone=mobilenetv3_large_100 p=0.04
echo "######################## p=0.06   ####################################"
python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=20 optimizer=SGD backbone=mobilenetv3_large_100 p=0.06
echo "######################## p=0.03  ####################################"
python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=20 optimizer=SGD backbone=mobilenetv3_large_100 p=0.03
echo "######################## p=0.07   ####################################"
python3 ensemble_train.py name=efficientnet_b3_train_smooth_P  k=4 lr=0.001 batch_size=20 optimizer=SGD backbone=mobilenetv3_large_100 p=0.07
