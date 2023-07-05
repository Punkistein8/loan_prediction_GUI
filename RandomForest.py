def obtener_prediccion_random_forest():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    # In[2]:

    # Cargamos los datos
    train = pd.read_csv('train_ctrUa4K.csv')
    test = pd.read_csv('testGUI.csv')

    # Copias de los datos originales
    train_original = train.copy()
    test_original = test.copy()

    # In[3]:

    # Reemplazando los valores de las variables categoricas
    train['Loan_Status'].replace('N', 0, inplace=True)
    train['Loan_Status'].replace('Y', 1, inplace=True)
    train['Dependents'].replace('3+', 3, inplace=True)
    train['Gender'].replace('Male', 0, inplace=True)
    train['Gender'].replace('Female', 1, inplace=True)
    train['Married'].replace('No', 0, inplace=True)
    train['Married'].replace('Yes', 1, inplace=True)
    train['Education'].replace('Not Graduate', 0, inplace=True)
    train['Education'].replace('Graduate', 1, inplace=True)
    train['Self_Employed'].replace('No', 0, inplace=True)
    train['Self_Employed'].replace('Yes', 1, inplace=True)
    train['Property_Area'].replace('Rural', 0, inplace=True)
    train['Property_Area'].replace('Semiurban', 1, inplace=True)
    train['Property_Area'].replace('Urban', 2, inplace=True)

    test['Dependents'].replace('3+', 3, inplace=True)
    test['Gender'].replace('Male', 0, inplace=True)
    test['Gender'].replace('Female', 1, inplace=True)
    test['Married'].replace('No', 0, inplace=True)
    test['Married'].replace('Yes', 1, inplace=True)
    test['Education'].replace('Not Graduate', 0, inplace=True)
    test['Education'].replace('Graduate', 1, inplace=True)
    test['Self_Employed'].replace('No', 0, inplace=True)
    test['Self_Employed'].replace('Yes', 1, inplace=True)
    test['Property_Area'].replace('Rural', 0, inplace=True)
    test['Property_Area'].replace('Semiurban', 1, inplace=True)
    test['Property_Area'].replace('Urban', 2, inplace=True)

    # In[4]:

    # Ver datos agrupados de una columna
    train.head(5)

    # In[5]:

    test

    # In[6]:

    # Llenando los valores faltantes
    train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    train['Married'].fillna(train['Married'].mode()[0], inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    train['Self_Employed'].fillna(
        train['Self_Employed'].mode()[0], inplace=True)
    train['Credit_History'].fillna(
        train['Credit_History'].mode()[0], inplace=True)

    train['Loan_Amount_Term'].fillna(
        train['Loan_Amount_Term'].mode()[0], inplace=True)
    train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

    # In[7]:

    # eliminando la columna Loan_ID
    train = train.drop('Loan_ID', axis=1)
    test = test.drop('Loan_ID', axis=1)

    # In[8]:

    # Se designa la variable objetivo y las variables independientes
    X = train.drop('Loan_Status', axis=1)
    y = train.Loan_Status

    # In[9]:

    # Se divide el conjunto de datos en entrenamiento y validacion
    from sklearn.model_selection import train_test_split
    x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

    # In[10]:

    # para hacer la validacion cruzada
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score  # para ver la precision del modelo

    i = 1
    kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(X, y):
        print('\n{} of kfold {}'.format(i, kf.n_splits))
        xtr, xvl = X.loc[train_index], X.loc[test_index]
        ytr, yvl = y[train_index], y[test_index]
        model = RandomForestClassifier(random_state=1, max_depth=10)
        model.fit(xtr, ytr)
        pred_test = model.predict(xvl)
        score = accuracy_score(yvl, pred_test)
        print('accuracy_score', score)
        i += 1
        pred_test = model.predict(test)

    # In[11]:

    # se hacen las predicciones con los datos de prueba
    pred_cv = model.predict(test)
    if pred_cv[0] == 0:
        return 'No Aprobado'
    else:
        return 'Aprobado'
