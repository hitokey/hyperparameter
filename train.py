from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("-d","--db",required=True)
parser.add_argument("-m","--model",required=True)
parser.add_argument("-j","--jobs",type=int,default=-1)
args = vars(parser.parse_args())

db = h5py.File(args["db"],"r")
i = int(db["labels"].shape[0]*0.75)

print("hyperparameters...")
params = {"C": [0.01,0.1,1.0,10.0,100.0,1000.0,10000.0]}
model = GridSearchCV(LogisticRegression(),params,cv=3,n_jobs=args["jobs"])
model.fit(db["features"][:i],db["labels"][:i])
print("Best hyperparameters: {}".format(model.best_params_))
print("Evaluating")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:],preds,
                            target_names=db["label_names"]))

print("Save models...")
f = open(args["model"],"wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()
db.close()
