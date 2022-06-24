python3 ../examples/MNNTrain/mnist/train_mnist.py
rm ./0.mnist.mnn
train_wrong=$[$? > 0]
printf "TEST_NAME_TRAIN_TEST: pymnn训练测试\nTEST_CASE_AMOUNT_TRAIN_TEST: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n" $train_wrong $[1 - $train_wrong]
