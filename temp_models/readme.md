Contains models per epoch

Using this code, we can test which model gave the best results

	for i in range(1,11):
	   with open('temp_models/brnn_model_%s_%s.pkl' % (TYPE, i), 'rb') as f:
		 BRNN = dill.load(f)
		 accuracy = BRNN.predict((testing_inputs, testing_targets), True)
		 print("Accuracy: {:.2f}%".format(accuracy * 100))

