__author__ = 'linas'

'''
A module to do visual inspection of recommendations.
The idea, that we try to check what recommendations would be provided and report the good
ones and the bad ones.
'''
import cherrypy
import webbrowser

class Explorer(object):

    def get_recommendations(self, model, user, items, k=6):
        preds = [(model.getScore(user, i), i) for i in items]
        preds.sort()
        return [i for _,i in preds[:k]]

    def visualize(self, model, user, items, k=6):
        recs = self.get_recommendations(model, user, items, k)
        print recs

if __name__ == "__main__":
    import testfm
    from pkg_resources import resource_filename
    from testfm.models.baseline_model import Popularity
    import pandas as pd

    df = pd.read_csv(resource_filename(testfm.__name__,'data/movielenshead.dat'),
    sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])

    model = Popularity()
    model.fit(df)

    e = Explorer()
    recs = e.get_recommendations(model, 1, df.item.unique())
    print df[df['item'].isin(recs)]['title'].unique()

    import urwid
