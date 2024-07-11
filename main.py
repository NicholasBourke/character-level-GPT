from train import accuracy_test

"""
Running accuracy_test will randomly select a number (batch_size) of character sequences
(context_length long) from the test subset of the enwik8 dataset over a number (num_batches)
of iterations.

The model attempts to predict the next character for each sequence. The percentage of correct
predictions are printed once all iterations are completed.

Please direct the function to the locations of model_state_dict.pth and the test dataset with
params_dir and data_dir respectively.
"""


params_dir = "parameters/"  # change to location of test dataset
data_dir = "data/"          # change to location of model_state_dict.pth
context_length = 512
batch_size = 16             
num_batches = 1000

accuracy_test(params_dir, data_dir, batch_size, num_batches)