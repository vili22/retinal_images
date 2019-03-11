from utils import loading

#print(tf.__version__)

training_data, valid_data = loading.load_samples()
print(training_data[0].shape)