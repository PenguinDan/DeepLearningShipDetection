#Image File Location
IMAGE_FILE_LOCATION = './images/'
PR_SAVE_LOCATION = "./object_detection_test/"
SMALL_IMAGES_LOCATION = "./small_images/"

#Small Ship Data File Names
FULL_SMALL_IMAGE_DATASET = './smallShipData.json'
TEST_SMALL_IMAGE_DATASET = 'small_test_data.txt'
TRAINING_SMALL_IMAGE_DATASET = 'small_training_data.txt'
TRAINED_CNN_MODEL = 'trained_CNN.h5'

#New Small Ship Data File Names
OTHER_TEST_SMALL_IMAGE_DATASET = 'other_small_test_data.json'
OTHER_TRAINING_SMALL_IMAGE_DATASET = 'other_small_training_data.json'
OTHER_TRAINED_CNN_MODEL = 'other_trained_CNN.h5'
#Same as version4 (This one sucked)
OTHER_GENERATOR_TRAINED_CNN_MODEL = 'other_generated_trained_CNN.h5'

#Models after data augmentation
#Unless otherwise specified, learing rate = 00.1, momentum = 1e-6, decay = 0.9
#Default value I saw on website with 50 samples
GENERATOR_TRAINED_CNN_MODEL = 'generated_trained_CNN.h5'
#Same but with 100 samples
GENERATOR_TRAINED_CNN_MODEL_TWO = 'generated_version2_trained_CNN.h5'
#Adding rotational augmentation 
GENERATOR_TRAINED_CNN_MODEL_THREE = 'generated_version3_trained_CNN.h5'
#Adding featurewise_center (Not very good)
GENERATOR_TRAINED_CNN_MODEL_FOUR = 'generated_version4_trained_CNN.h5'
#Added 2 fully connected layers 100/100 
GENERATOR_TRAINED_CNN_MODEL_FIVE = 'generated_version5_trained_CNN.h5'
#Same as version 5 but 200/100 (Best for the final object detection ship predictions)
GENERATOR_TRAINED_CNN_MODEL_SIX = 'generated_version6_trained_CNN.h5'



#Models with transfer learning
# Learning rate = 0.001, decay = 0.9, epoch = 10             
# Training Values     loss: 0.0160 - binary_accuracy: 0.9990
# Validation Values   Loss:  0.1450024089826009      Accuracy:  0.9860000014305115
PRETRAINED_CNN_MODEL_ONE = "pretrained_attempt_1.h5"

#Learning rate = 0.0001, decay = 0.75, epoch = 10             
# Training Values     loss: 0.0042 - binary_accuracy: 0.9995
# Validation Values   Loss:  0.06005835090763867      Accuracy:  0.977999997138977
PRETRAINED_CNN_MODEL_TWO = "pretrained_attempt_2.h5"

# Learning rate = 0.00001, decay = 0.65, epoch = 30             
# Training Values     loss: 0.0239 - binary_accuracy: 0.9945
# Validation Values   Loss:  0.1274039216339588      Accuracy:  0.9580000042915344
PRETRAINED_CNN_MODEL_THREE = "pretrained_attempt_3.h5"

# Learning rate = 0.001, decay = 0.9, epoch = 10        with data aumentation
# Training Values     loss: 4.1746 - binary_accuracy: 0.7410
# Validation Values   val_loss: 3.7394 - val_binary_accuracy: 0.7680
PRETRAINED_CNN_MODEL_FOUR = "pretrained_attempt_4.h5"

#Learning rate = 0.0001, decay = 0.75, epoch = 10       with data augmentation         
# Training Values     loss: 0.0434 - binary_accuracy: 0.9853
# Validation Values   val_loss: 0.0395 - val_binary_accuracy: 0.9860
PRETRAINED_CNN_MODEL_FIVE = "pretrained_attempt_5.h5"

# Learning rate = 0.00001, decay = 0.65, epoch = 30     with data augmentation    
# Training Values     loss: 0.0779 - binary_accuracy: 0.9732
# Validation Values   val_loss: 0.0575 - val_binary_accuracy: 0.9700
PRETRAINED_CNN_MODEL_SIX = "pretrained_attempt_6.h5"
