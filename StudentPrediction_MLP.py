from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#study_hours , attendence , assignments_score
X = [
    [1,40,30],
    [2,50,35],
    [3,60,40],
    [4,65,50],
    [5,70,55],
    [6,75,65],
    [7,80,70],
    [2,45,25],
    [8,90,85],
    [1,35,20],
    [3,55,20],
    [4,65,50],
    [5,70,55],
    [6,75,60],
    [7,85,75]
]

y = [0,0,0,1,1,1,1,0,1,0,0,1,1,1,1]

x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

model.fit(x_train , y_train)

y_pred = model.predict(x_test)

print("Actual output : ",y_test)
print("Predicted output : ",y_pred)

accuracy = accuracy_score(y_test , y_pred)
print("Accuracy of Deep learning model is : ",accuracy)