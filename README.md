Test.fm is a testing framework for Collaborative Filtering factor models.
It integrates with pandas as the default data model and
gives an easy wayt to check how well your models do.

Example of using the Test.fm framework
======================================


	import numpy as np
	import testfm
	from testfm.evaluation.evaluator import Evaluator
	from testfm.models.baseline_model import Popularity, RandomModel
	from testfm.models.tensorCoFi import TensorCoFi

	#prepare the data
	df = pd.read_csv(..., names=['user', 'item', 'rating', 'date', 'title'])
	training, testing = testfm.split.holdoutByRandom(df, 0.9)

	#tell me what models we want to evaluate
	models = [  RandomModel(),
            	Popularity(),
            	TensorCoFi(),
	         ]

	#evaluate
	items = training.item.unique()
	evaluator = Evaluator()
	for m in models:
		m.fit(training)
		print m.getName().ljust(50),
		print evaluate_model(m, testing, all_items=items)
