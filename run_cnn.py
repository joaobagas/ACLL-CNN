from network.functions import *
from network.dataset_creator import *

# input_path = "/home/bernardo/Desktop/NordicDenmark"
input_path = "/home/bernardo/Desktop/Images/No animals/og"
output_path = "/home/bernardo/Desktop/Images/No animals/Dataset1"



train_path = "datasets/acll/train"
test_path = "datasets/acll/test"

a = 3
if a == 1:
    net = train(load_model=True, batch_size=5, num_epochs=500, min_loss=0.001, train_path=train_path)
elif a == 2:
    over_fit_single_batch(load_model=False, batch_size=5, num_epochs=1000, test_every_epoch=10, train_path=train_path)
elif a == 3:
    print("\nTest path:\n")
    test(net=None, test_path=test_path)
    print("\nTrain path:\n")
    test(net=None, test_path=train_path)
elif a == 4:
    create_dataset(input_path, output_path)