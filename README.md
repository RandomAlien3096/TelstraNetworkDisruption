# TelstraNetworkDisruption

## Data Understanding and Data Preparation
Ahora, despues de nuestra matriz de correlacion procederemos a investigar visualmente un poco mas sobre la relevancia que tiene la ubicacion sobre el tipo de fallas que se encuentran. Para ello, utilizaremos una grafica de dispersion para visualizar la ubicacion de cada falla respecto a la informacion que se tiene. 

/*insertar grafica*/

Ya vista la grafica de dispersion, podemos notar una tendencia. Entre mayor sea el numero de la ubicacion, las fallas se hacen mas propensas y aumentan su severidad. Se puede asumir que, entre mayor sea el numero de la ubicacion, esta se encontra mas lejos. Y que por igual, que cada uno de las ubicaciones, dependiendo de su numero, se encuentran en orden y en cercania relativa al valor dado. A partir de ello, podemos plantear una hipotesis con nuestra matriz de correlacion; La severidad y cantidad de las fallas en la red de TELSTRA sera directamente proporcional a la ubicacion y distancia que estas tengan. 


## Modeling
Debido a la naturalidad categorica de nuestro complejo de datos, es necesario, ya sea, procesar profundamente nuestro set de datos o utilizar un algoritmo de machine learning capaz de poder procesar esta informacion. Dado esto, se propuso utilizar el algoritmo open-source llamado "CatBoost". El cual es un algoritmo bastante versatil y flexible. Donde tiene la capacidad de manejar una gran variedad de tipo de datos sin ningun problema proveyendo soluciones fuera de lo convencional para apoyar con los problemas dados comunmente en el analisis de negocios y big data. 

El objetivo ahora sera desplegar el algoritmo CatBoost de forma correcta. Para ello, nececitamos iniciar con el entrenamiento supervisado del complejo de datos ya procesado. Con ello utilizaremos el complejo de datos "train" donde utilizaremos el 75% de sus datos para entrenar al algoritmo y el 25% seran utilizados para validar el entrenamiento y simular a su vez como este se comportara si es dado un set de datos no antes vistos. 

-------------------------------------------------------------------------------------------

#Splitting X(data in array) and y (index of data used)

X = train_4[['id', 'location', 'severity_type', 'resource_type',
       'log_feature', 'volume', 'event_type']]
y = train_4.fault_severity
 
#divide the training set into train/validation set with 25% set aside for validation. 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

-------------------------------------------------------------------------------------------

Ahora, actualizaremos los parametros del algoritmo para poder hacer uso de la naturaleza categorica del mismo y utilizarlo en su mejor capacidad para mejores resultados

-------------------------------------------------------------------------------------------
 
categorical_features_indices = np.where(X_train.dtypes == object)[0])


 
 -------------------------------------------------------------------------------------------
 
#using pool to make the training and validation sets
train_dataset = Pool(data=X_train,
                     label=y_train,
                     cat_features=categorical_features_indices)

eval_dataset = Pool(data=X_test,
                    label=y_test,
                    cat_features=categorical_features_indices)
 
#initialize the catboost classifier
model = CatBoostClassifier(iterations=1000,
                           learning_rate=1,
                           depth=2,
                           loss_function='MultiClass',
                           random_seed=42,
                           bagging_temperature=22,
                           od_type='Iter',
                           metric_period=100,
                           od_wait=100)
#Fit model

model.fit(train_dataset, eval_set= eval_dataset, plot= True)


model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test),plot=True)

-------------------------------------------------------------------------------------------

Ahora que tenemos nuestro modelo ya entrenado, utilizaremos la funcion predict() para poder predecir los valores con el conjunto de datos de validacion que en este caso, sera el 25% de valores de nuestro train.csv Mientras que nuestra funcion predict_proba() nos dara la probabilidad de cada uno de los puntos dados.

-------------------------------------------------------------------------------------------
#predicts the actual label or class over the evaluation data set (gives us the final choice)
preds_class = model.predict(eval_dataset) 
print(preds_class)

#get predicted probabilities for each class (gives us the probabilities of each choice option that it had)
preds_proba = model.predict_proba(eval_dataset)
print(preds_proba)

-------------------------------------------------------------------------------------------

Ya que tenemos nuestro modelo entrenado y evaluado, es hora de implementar este a nuestro archivo test.cvs. Esto para poder predecir la severidad de las fallas dependiendo de la ubicacion. Crearemos nuestro conjunto de datos de test.csv

-------------------------------------------------------------------------------------------
print("The shape of the test data set without merging is: {}".format(test.shape()))

test_1 = test.merge(severity_type, how = 'left', left_on='id', right_on='id')
test_2 = test_1.merge(resource_type, how = 'left', left_on='id', right_on='id')
test_3 = test_2.merge(log_failure, how = 'left', left_on='id', right_on='id')
test_4 = test_3.merge(event_type, how = 'left', left_on='id', right_on='id')

#removing the duplicates.
test_4.drop_duplicates(subset= 'id', keep= 'first', inplace = True)
 
#checking for any null values. 
test_4.isnull().sum()

print("The shape of the merged test dataset is: {}".format(test_4.shape()))

-------------------------------------------------------------------------------------------

Ahora utilizaremos el modelo en el nuevo conjunto de datos para poder predecir la severidad de fallas y en donde estas sucederan.

-------------------------------------------------------------------------------------------

predict_test = model.predict_proba(test_4) #using the trained catboost model to get the probabilities of the choices
print(predict_test.head(15))
print("The shape of the prediction test dataset is now: {}".format(predict_test.shape()))


pred_df = pd.DataFrame(predict_test, columns = ['predict_0', 'predict_1', 'predict_2'])
print(pred_df.head(15))
print("The shape of the prediction data frame is now: {}".format(pred_df.shape()))


submission_cat = pd.concat([test[['id']],pred_df],axis=1)
submission_cat.to_csv('sub_cat_1.csv',index=False,header=True)
submission_cat.head(15)

-------------------------------------------------------------------------------------------

Para poder visualizar correctamente las predicciones del data set test, tendremos que utilizar la funcion predict() con nuestro modelo entrenado para usar la grafica de dispersion. 

-------------------------------------------------------------------------------------------

predict_telstra = model.predict(test_4)
print(predict_telstra)

-------------------------------------------------------------------------------------------

print("The shape of the final prediction is going to be: {}".format(predict_telstra.shape()))
