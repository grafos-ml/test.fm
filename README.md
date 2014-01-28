Conceptual example of using the system
======================================


	import numpy as np
	import testfm
	from testfm.evaluation.evaluator import Evaluator
	from testfm.models.baseline_model import Popularity, RandomModel
	from testfm.models.tensorCoFi import TensorCoFi
	from testfm.models.content_based import LSIModel
	from pkg_resources import resource_filename

	#prepare the data
	df = pd.read_csv(..., names=['user', 'item', 'rating', 'date', 'title'])
	training, testing = testfm.split.holdoutByRandomFast(df, 0.9)

	#tell me what models we want to evaluate
	models = [  RandomModel(),
            	Popularity(),
           	 	LSIModel('title'),
            	TensorCoFi(),
	         ]

	#evaluate
	items = training.item.unique()
	evaluator = Evaluator()

	print "\n\n Multiprocessing"
	for m in models:
		m.fit(training)
		print m.getName().ljust(50), \
			list(evaluator.evaluate_model_multiprocessing(m,
				testing, all_items=items))
