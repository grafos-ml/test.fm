[![Build Status](https://travis-ci.org/grafos-ml/test.fm.svg?branch=master)](https://travis-ci.org/grafos-ml/test.fm)

Introduction
============

Test.fm is (yet another) testing framework for Collaborative Filtering models.
It integrates well with pandas as the default data manipulation library and
gives an easy way to investigate how well your models perform and why.
You can build a model using [okapi](http://grafos.ml) and then check how it performs on the testing data.
Or if you have only a little data set, you can use it directly.

Example of using the Test.fm framework
======================================
```python
	import pandas as pd
	import testfm
	from testfm.models.baseline_model import Popularity, RandomModel
	from testfm.models.tensorcofi import TensorCoFi
	from testfm.evaluation.evaluator import Evaluator
	
	evaluator = Evaluator()

	# Prepare the data
	df = pd.read_csv(..., names=["user", "item", "rating", "date", "title"])
	training, testing = testfm.split.holdoutByRandom(df, 0.9)

	# Tell me what models we want to evaluate
	models = [
	    RandomModel(),
	    Popularity(),
	    TensorCoFi()
	    ]
	
	# Evaluate
	items = training.item.unique()
	for m in models:
		m.fit(training)
		print m.getName().ljust(50),
		print evaluator.evaluate_model(m, testing, all_items=items)
```

See other examples [here...](https://github.com/grafos-ml/test.fm/tree/master/src/testfm/examples)

Installation
============
You can check the official documentation [here](http://grafos-ml.github.io/test.fm).

1. download and extract the sources.
2. check the dependencies in conf/requirements.txt
3. run #sudo python setup.py install
4. if you are a developer of test.fm better do python setup.py develop
5. enjoy and contribute
6. Check travis for the latest [builds...](https://travis-ci.org/grafos-ml/test.fm)
7. Check [yaml](https://github.com/grafos-ml/test.fm/blob/master/.travis.yml) for the build script.

Nosetests
=========
$ nosetests -w src/ -vv --with-cover --cover-tests --cover-erase --cover-html --cover-package=testfm --with-doctest --doctest-tests tests testfm/evaluation testfm/models testfm/fmio testfm/splitter

Build Documentation
===================
$ sphinx-build -b html source_folder doc_folder

Similar Projects
================
1. [mrec](https://github.com/Mendeley/mrec/tree/master/mrec) from Mendeley. Good at building models. (python, ?)
2. [okapi](http://grafos.ml) from Telefonica Research. Good at distributed model building using Apache Giraph (java, giraph, apache2).
3. [graphlab](http://graphlab.org/) from CMU. Probably the richest library of modern algorithms (c++, apache2).
4. [mymedialite](http://www.mymedialite.net/) from Uni Hildesheim. Has ranking implementations. (c#, GPL).
5. [mahout](https://mahout.apache.org/) of apache. Uses hadoop to build the models. (java, hadoop, apache2)
6. [lenskit](http://lenskit.grouplens.org/) Grouplens (java, GPL2.1)
