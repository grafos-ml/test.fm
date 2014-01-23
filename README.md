Conceptual example of using the system
======================================

	import testfm
	import pandas as pd
	
	#Lets see how to build a model and test it
	df = pd.load(...)
	training, testing = testfm.split.holdout(df, fraction=0.9)
	model = testfm.buildmodel('okapi:iMF', training, user_dim=0, item_dim=1)
	print testfm.evaluate_model(model, testing, ['MAP', 'P@5'])
	#[0.9556, 0.8575]
	
	#Lets see how to do a cross-validation
	print testfm.cross_validate(df, build_model='okapi:iMF', measure='MAP', folds=5)
	#(0.9556, 0.9547, 0.9587, 0.9511, 0.9612)
	
	#Model selection (tune the hyperparameters)
	params = testfm.select_model(training, build_model='okapi:iMF', user_dim=0, item_dim=0)
	print params
	#('--lambda', 0.01)
	model = testfm.buildmodel('okapi:iMF', training, user_dim=0, item_dim=1, params=params)

