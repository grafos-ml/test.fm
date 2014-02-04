Introduction
===========

Test.fm is (yet another) testing framework for Collaborative Filtering factor models.
It integrates well with pandas as the default data model and
gives an easy wayt to investigate how well your models perform and why.
You can build model using okapi and then check how it performs on the testing data.
Or if you have little data sets, you can use it directly.

Example of using the Test.fm framework
======================================
```python
	import pandas as pd
	import testfm
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
	for m in models:
		m.fit(training)
		print m.getName().ljust(50),
		print testfm.evaluate_model(m, testing, all_items=items)
```
Instalation
==========
1. download and extract the sources.
2. check the dependencies in conf/requirements.txt (the pyjnius could fail if you use pip)
3. run #sudo python setup.py install
4. if you are a developer of test.fm better do python setup.py develop
5. enjoy and contribute
