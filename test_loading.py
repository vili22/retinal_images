from utils import loading

#print(tf.__version__)

training_data, valid_data, test_data = loading.load_samples(2000)
print(training_data[0].shape)