def My_MAE(y_true , y_pred):
    n = len(y_true)
    
    total_error = 0
    for i in range(n):
        error = abs(y_true[i] - y_pred[i])
        total_error = total_error + error

    MAE = total_error/n
    return MAE
    
y_true = [10 , 20 , 30]
y_pred = [12 , 18 , 33]

loss = My_MAE(y_true , y_pred)
print("Loss is : ",loss)
