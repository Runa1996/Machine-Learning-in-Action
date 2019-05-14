from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pydotplus
from sklearn.externals.six import StringIO

if __name__ == "__main__":
    with open('./lenses.txt','r') as f:
        lenses = [word.strip().split('\t') for word in f.readlines()]
    lenses_target = np.array(lenses)[:,-1].tolist()
    lenses_label = ['age','prescript','astigmatic','tearRate']
    lenses_array = np.array(lenses)[:,:-1]
    lenses_pd = pd.DataFrame(lenses_array, columns = lenses_label)
    label_encoder = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = label_encoder.fit_transform(lenses_pd[col])
    
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(),lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file = dot_data,feature_names = lenses_pd.keys(),class_names = clf.classes_, filled = True,rounded = True, special_characters = True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')
    print(clf.predict([[1,1,1,0]]))