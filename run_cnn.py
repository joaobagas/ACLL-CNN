from network.functions import *
from network.dataset_creator import *

# input_path = "/home/bernardo/Desktop/NordicDenmark/"
# output_path = "/home/bernardo/Desktop/Images/No animals/Dataset1"

# create_dataset(input_path, output_path)

train_path = "datasets/acll/train"
test_path = "datasets/acll/test"

train(True, 5, 10, 10, train_path)
test(True, test_path)
