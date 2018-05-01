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
#Default value I saw on website with 50 samples
GENERATOR_TRAINED_CNN_MODEL = 'generated_trained_CNN.h5'
#Same but with 100 samples
GENERATOR_TRAINED_CNN_MODEL_TWO = 'generated_version2_trained_CNN.h5'
#Adding rotational augmentation 
GENERATOR_TRAINED_CNN_MODEL_THREE = 'generated_version3_trained_CNN.h5'
#Adding featurewise_center (Not very good)
GENERATOR_TRAINED_CNN_MODEL_FOUR = 'generated_version4_trained_CNN.h5'