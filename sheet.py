@app.route('/bo1_modeling', methods=['GET', 'POST'])
def bo1_modeling():
    # Charger les données préparées
    prepared_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'prepared_dataAssurance.csv')
    data = pd.read_csv(prepared_filepath)

    # Diviser les données en variables d'entrée (X) et variable cible (y)
    X = data[['age', 'sex', 'bmi', 'children', 'smoker'] + 
             [col for col in data.columns if col.startswith('region_')] + 
             ['charges_per_child', 'bmi_smoker']]
    y = data['charges']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Prédiction et évaluation pour Random Forest
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    # Modèle Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)

    # Prédiction et évaluation pour Gradient Boosting
    y_pred_gb = gb_model.predict(X_test)
    r2_gb = r2_score(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

    # Passer les résultats dans le template
    return render_template(
        'bo1_modeling.html',
        r2_rf=r2_rf,
        rmse_rf=rmse_rf,
        r2_gb=r2_gb,
        rmse_gb=rmse_gb
    ) 


@app.route('/bo2_modeling', methods=['GET', 'POST'])
def bo2_modeling():
    # Charger les données préparées
    prepared_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'prepared_dataAssurance.csv')
    data = pd.read_csv(prepared_filepath)

    # Code pour la classification
    # Exécution de la modélisation et évaluation

    return render_template('bo2_modeling.html')