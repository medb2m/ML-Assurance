# app.py
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


""" BO1 : Prédire les coûts médicaux par personne.
BO2 : Analyser l’impact du tabagisme sur les coûts médicaux.
BO3 : Prédire les coûts médicaux en fonction du nombre d’enfants  """


app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Ensure the folders exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Home route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading a CSV file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Redirect to data understanding with the file path
        return redirect(url_for('data_understanding', filepath=file.filename))
    else:
        flash('Please upload a CSV file.')
        return redirect(url_for('index'))

# Route for data understanding step with visualizations
@app.route('/data_understanding/<filepath>')
def data_understanding(filepath):
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    data = pd.read_csv(full_path)

    # Stats for display
    num_rows, num_columns = data.shape
    column_info = data.dtypes.to_dict()
    summary_stats = data.describe().to_html()

    # Calculate missing data percentage
    missing_data = data.isnull().mean() * 100
    missing_data_df = missing_data[missing_data > 0].reset_index()
    missing_data_df.columns = ['Column', 'Percentage Missing']
    # Convert missing data to HTML table format for display
    missing_data_html = missing_data_df.to_html(index=False, classes="missing-data-table")

    # Duplicate data analysis
    duplicates = data.duplicated()
    num_duplicates = duplicates.sum()
    duplicates_html = data[duplicates].to_html(index=False, classes="duplicates-table") if num_duplicates > 0 else "<p>No duplicates found.</p>"

    # Clear old images
    for filename in os.listdir(app.config['IMAGE_FOLDER']):
        file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    # Generate individual visualizations
    # 1. Distribution of age
    plt.figure(figsize=(8, 6))
    sns.histplot(data['age'], bins=20, kde=True, color='blue')
    plt.title("Distribution de l'âge des bénéficiaires")
    plt.xlabel("Âge")
    plt.ylabel("Nombre de bénéficiaires")
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'age_distribution.png'))
    plt.clf()

    # 2. Smoker status distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='smoker', data=data, palette='Set1')
    plt.title("Répartition des fumeurs et non-fumeurs")
    plt.xlabel("Fumeur")
    plt.ylabel("Nombre de bénéficiaires")
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'smoker_distribution.png'))
    plt.clf()

    # 3. Medical charges distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data['charges'], bins=20, kde=True, color='green')
    plt.title("Distribution des frais médicaux")
    plt.xlabel("Charges médicales")
    plt.ylabel("Nombre de bénéficiaires")
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'charges_distribution.png'))
    plt.clf()

    # 4. BMI distribution by region
    plt.figure(figsize=(8, 6))
    data_grouped = data.groupby('region', as_index=False)['bmi'].mean()
    sns.barplot(x='region', y='bmi', data=data_grouped, palette='Set2')
    plt.title("Distribution de l'IMC par région")
    plt.xlabel("Région")
    plt.ylabel("IMC")
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'bmi_by_region.png'))
    plt.clf()

    # 5. Pair plot of all variables
    sns.pairplot(data)
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'pairplot.png'))
    plt.clf()

    # 6. Impact of smoking on medical charges
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='smoker', y='charges', data=data)
    plt.title("Impact du tabagisme sur les coûts médicaux")
    plt.xlabel("Fumeur")
    plt.ylabel("Charges médicales")
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'smoker_vs_charges.png'))
    plt.clf()

    # Noisy Data Analysis - Generating boxplots for numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for i, column in enumerate(numeric_columns):
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column}")
        plot_path = os.path.join(app.config['IMAGE_FOLDER'], f'boxplot_{column}.png')
        plt.savefig(plot_path)
        plt.close()  # Close the plot to save memory

    
    return render_template(
        'data_understanding.html', 
        filepath=filepath,
        num_rows=num_rows,
        num_columns=num_columns,
        column_info=column_info,
        summary_stats=summary_stats,
        missing_data_html=missing_data_html,
        num_duplicates=num_duplicates,
        duplicates_html=duplicates_html,
        numeric_columns=numeric_columns,
        images=['age_distribution.png', 'smoker_distribution.png', 'charges_distribution.png', 'bmi_by_region.png', 'pairplot.png', 'smoker_vs_charges.png']
    )

# Route for data preparation step
@app.route('/data_preparation/<filepath>', methods=['GET', 'POST'])
def data_preparation(filepath):
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    data = pd.read_csv(full_path)

    # Handle data preparation if the button was clicked
    if request.method == 'POST':
        data = data.dropna()  # Ensure no missing values
        # Imputation des valeurs manquantes pour les colonnes numériques
        imputer_numeric = SimpleImputer(strategy='mean')
        data[['age', 'bmi', 'children', 'charges']] = imputer_numeric.fit_transform(data[['age', 'bmi', 'children', 'charges']])

        # Imputation des valeurs manquantes pour les colonnes catégorielles avec le mode
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        data[['sex', 'smoker', 'region']] = imputer_categorical.fit_transform(data[['sex', 'smoker', 'region']])
        
        # Encodage binaire pour les colonnes `sex` et `smoker`
        data['sex'] = data['sex'].map({'female': 0, 'male': 1})
        data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})

        # Encodage one-hot pour la colonne `region`
        data = pd.get_dummies(data, columns=['region'], drop_first=True)  # `drop_first=True` pour éviter la multicolinéarité

        # Mise à l’échelle des colonnes `age`, `bmi`, et `charges`
        scaler = StandardScaler()
        data[['age', 'bmi', 'charges']] = scaler.fit_transform(data[['age', 'bmi', 'charges']])

        # BO3 : Créer la colonne `charges_per_child`
        data['charges_per_child'] = data['charges'] / (data['children'] + 1)  # Ajouter 1 pour éviter la division par zéro

        # BO2 : Créer la colonne d'interaction `bmi_smoker`
        data['bmi_smoker'] = data['bmi'] * data['smoker']

        # Détection des valeurs aberrantes en utilisant le score Z pour les colonnes numériques
        from scipy.stats import zscore
        z_scores = np.abs(zscore(data[['age', 'bmi', 'charges']]))
        data = data[(z_scores < 3).all(axis=1)]  # Retirer les lignes avec un score Z > 3 (valeur aberrante)

        # Enregistrer les données mises à jour après le traitement
        updated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'prepared_' + filepath)
        data.to_csv(updated_filepath, index=False)
        return redirect(url_for('data_preparation', filepath='prepared_' + filepath))

    # Show the columns with missing values
    missing_columns = data.isnull().sum()[data.isnull().sum() > 0].to_dict()
    duplicate_count = data.duplicated().sum()

    return render_template(
        'data_preparation.html',
        missing_columns=missing_columns,
        duplicate_count=duplicate_count,
        filepath=filepath
    )



# Route pour l'étape de modélisation
@app.route('/modeling/<filepath>')
def modeling(filepath):
    # Logique pour la modélisation (à compléter)
    return render_template('modeling.html', filepath=filepath)



@app.route('/bo1_modeling', methods=['GET', 'POST'])
def bo1_modeling():
    # Load prepared data
    prepared_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'prepared_dataAssurance.csv')
    data = pd.read_csv(prepared_filepath)


    y = data['charges']
    # Multiple Linear Regression (Age, Sex, BMI, Children, Smoker -> Charges)
    X_multiple = data[['age', 'sex', 'bmi', 'children', 'smoker']]
    X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, y, test_size=0.2, random_state=42)
    multiple_lr_model = LinearRegression()
    multiple_lr_model.fit(X_train_multiple, y_train_multiple)
    y_pred_multiple = multiple_lr_model.predict(X_test_multiple)
    r2_multiple = r2_score(y_test_multiple, y_pred_multiple)
    rmse_multiple = np.sqrt(mean_squared_error(y_test_multiple, y_pred_multiple))

    # Debugging: Manual prediction
    test_input_data = [[40, 1, 30.0, 2, 1]]  # e.g., age=40, sex=male(1), bmi=30.0, children=2, smoker=yes(1)
    try:
        test_prediction = multiple_lr_model.predict(test_input_data)[0]
        print(f"Manual test prediction for input {test_input_data}: {test_prediction}")
    except Exception as e:
        print(f"Error in manual prediction: {e}")

    # Visualization for Multiple Regression
    plt.figure()
    plt.scatter(y_test_multiple, y_pred_multiple, color='green')
    plt.title('Multiple Linear Regression (Real Charges vs Predictions)')
    plt.xlabel('Real Charges')
    plt.ylabel('Predicted Charges')
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'multiple_lr_plot.png'))
    plt.close()

    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_multiple, y_train_multiple)
    y_pred_rf = rf_model.predict(X_test_multiple)
    r2_rf = r2_score(y_test_multiple, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test_multiple, y_pred_rf))

    # Visualization for Random Forest
    plt.figure()
    plt.scatter(y_test_multiple, y_pred_rf, color='orange')
    plt.title('Random Forest Regressor (Real Charges vs Predictions)')
    plt.xlabel('Real Charges')
    plt.ylabel('Predicted Charges')
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'rf_plot.png'))
    plt.close()

    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train_multiple, y_train_multiple)
    y_pred_gb = gb_model.predict(X_test_multiple)
    r2_gb = r2_score(y_test_multiple, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test_multiple, y_pred_gb))

    # Visualization for Gradient Boosting
    plt.figure()
    plt.scatter(y_test_multiple, y_pred_gb, color='purple')
    plt.title('Gradient Boosting Regressor (Real Charges vs Predictions)')
    plt.xlabel('Real Charges')
    plt.ylabel('Predicted Charges')
    plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], 'gb_plot.png'))
    plt.close()

    # Check if request method is POST for prediction
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        # Extract features from form
        bmi = float(request.form.get('bmi'))
        age = int(request.form.get('age', 0))
        sex = request.form.get('sex', 'male')
        children = int(request.form.get('children', 0))
        smoker = request.form.get('smoker', 'no')

        # Map categorical variables manually
        sex_map = {'male': 1, 'female': 0}
        smoker_map = {'yes': 1, 'no': 0}
        sex_encoded = sex_map.get(sex, 0)
        smoker_encoded = smoker_map.get(smoker, 0)
        input_data = [[age, sex_encoded, bmi, children, smoker_encoded]]
        
        
            
            # Predict based on selected algorithm
        if algorithm == 'multiple':
            prediction = multiple_lr_model.predict(input_data)[0]
        elif algorithm == 'random_forest':
            prediction = rf_model.predict(input_data)[0]
        elif algorithm == 'gradient_boosting':
            prediction = gb_model.predict(input_data)[0]

        # Adjust prediction if necessary
        if prediction < 100:  # Example: if the prediction is in an unexpectedly low range
            prediction *= 1000 

        # Round the prediction for display
        prediction = round(prediction, 2)
    else:
        prediction = None  # No prediction if not submitted

    return render_template(
        'bo1_modeling.html',
        r2_multiple=r2_multiple,
        rmse_multiple=rmse_multiple,
        r2_rf=r2_rf,
        rmse_rf=rmse_rf,
        r2_gb=r2_gb,
        rmse_gb=rmse_gb,
        simple_lr_plot_url=os.path.join('/static/images/simple_lr_plot.png'),
        multiple_lr_plot_url=os.path.join('/static/images/multiple_lr_plot.png'),
        rf_plot_url=os.path.join('/static/images/rf_plot.png'),
        gb_plot_url=os.path.join('/static/images/gb_plot.png'),
        prediction=prediction,
        test_prediction=test_prediction
    )






@app.route('/bo2_modeling', methods=['GET', 'POST'])
def bo2_modeling():
    # Charger les données préparées
    prepared_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'prepared_dataAssurance.csv')
    data = pd.read_csv(prepared_filepath)

    X = data[['age', 'bmi', 'children', 'charges']]
    y = data['smoker'].apply(lambda x: 1 if x == 'yes' else 0)  # Convert smoker status to binary
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize models
    knn_model = KNeighborsClassifier(n_neighbors=3)
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    # Train and evaluate KNN model
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    knn_report = classification_report(y_test, y_pred_knn)
    knn_cm = confusion_matrix(y_test, y_pred_knn)
    
    # Save KNN confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("KNN Confusion Matrix")
    knn_plot_path = os.path.join(app.config['IMAGE_FOLDER'], 'knn_confusion_matrix.png')
    plt.savefig(knn_plot_path)
    plt.close()

    # Train and evaluate Decision Tree model
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_report = classification_report(y_test, y_pred_dt)
    dt_cm = confusion_matrix(y_test, y_pred_dt)
    
    # Save Decision Tree confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Greens", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Decision Tree Confusion Matrix")
    dt_plot_path = os.path.join(app.config['IMAGE_FOLDER'], 'dt_confusion_matrix.png')
    plt.savefig(dt_plot_path)
    plt.close()

    prediction_result = None

    # Handle user input if POST request is made
    if request.method == 'POST':
        # Get data from form input
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        charges = float(request.form['charges'])

        # Prepare input for prediction
        user_data = np.array([[age, bmi, children, charges]])

        # Predict using KNN and Decision Tree
        knn_prediction = knn_model.predict(user_data)[0]
        dt_prediction = dt_model.predict(user_data)[0]

        # Convert numeric results to 'Yes'/'No' for smoker
        prediction_result = {
            'KNN': 'Yes' if knn_prediction == 1 else 'No',
            'DecisionTree': 'Yes' if dt_prediction == 1 else 'No'
        }

    # Render HTML with the image paths for display
    return render_template('bo2_modeling.html', 
                           knn_report=knn_report, dt_report=dt_report, 
                           knn_plot_path=knn_plot_path, dt_plot_path=dt_plot_path,
                           prediction_result=prediction_result)

@app.route('/bo3_modeling', methods=['GET', 'POST'])
def bo3_modeling():
    # Charger les données préparées
    prepared_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'prepared_dataAssurance.csv')
    data = pd.read_csv(prepared_filepath)

    # Définir X et y pour la prédiction des coûts médicaux en fonction du nombre d’enfants
    X = data[['age', 'bmi', 'children', 'sex', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']]
    y = data['charges_per_child']

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1. Régression linéaire
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Calcul des métriques pour la régression linéaire
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # 2. Forêt aléatoire
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Calcul des métriques pour la forêt aléatoire
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    # Enregistrer le graphique de la relation entre prédictions et valeurs réelles pour la régression linéaire
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Prédictions")
    plt.title("Régression Linéaire - Prédictions vs Valeurs Réelles")
    lr_plot_path = os.path.join(app.config['IMAGE_FOLDER'], 'lr_predictions_vs_actual.png')
    plt.savefig(lr_plot_path)
    plt.close()
    
    # Enregistrer le graphique de la distribution des erreurs pour la régression linéaire
    plt.figure(figsize=(8, 6))
    sns.histplot(y_test - y_pred_lr, bins=20, kde=True, color='blue')
    plt.xlabel("Erreur")
    plt.ylabel("Fréquence")
    plt.title("Régression Linéaire - Distribution des Erreurs")
    lr_error_plot_path = os.path.join(app.config['IMAGE_FOLDER'], 'lr_error_distribution.png')
    plt.savefig(lr_error_plot_path)
    plt.close()

    # Enregistrer le graphique de la relation entre prédictions et valeurs réelles pour la forêt aléatoire
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, color='green', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Prédictions")
    plt.title("Forêt Aléatoire - Prédictions vs Valeurs Réelles")
    rf_plot_path = os.path.join(app.config['IMAGE_FOLDER'], 'rf_predictions_vs_actual.png')
    plt.savefig(rf_plot_path)
    plt.close()

    # Enregistrer le graphique de la distribution des erreurs pour la forêt aléatoire
    plt.figure(figsize=(8, 6))
    sns.histplot(y_test - y_pred_rf, bins=20, kde=True, color='green')
    plt.xlabel("Erreur")
    plt.ylabel("Fréquence")
    plt.title("Forêt Aléatoire - Distribution des Erreurs")
    rf_error_plot_path = os.path.join(app.config['IMAGE_FOLDER'], 'rf_error_distribution.png')
    plt.savefig(rf_error_plot_path)
    plt.close()

    # Initialize prediction results
    user_prediction_lr = None
    user_prediction_rf = None

    if request.method == 'POST':
        # Get form data for prediction
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        sex = request.form['sex']
        smoker = request.form['smoker']
        region = request.form['region']

        # Preprocess input features
        sex_male = 1 if sex == 'male' else 0
        smoker_yes = 1 if smoker == 'yes' else 0
        region_northwest = 1 if region == 'northwest' else 0
        region_southeast = 1 if region == 'southeast' else 0
        region_southwest = 1 if region == 'southwest' else 0

        # Create input vector for prediction
        user_input = [[age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest]]
        
        # Predict charges using both models
        user_prediction_lr = lr_model.predict(user_input)[0]
        user_prediction_rf = rf_model.predict(user_input)[0]

        user_prediction_lr *= 100
        user_prediction_rf *= 100

    # Calculate metrics for both models
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)


    return render_template(
        'bo3_modeling.html',
        mse_lr=mse_lr,
        r2_lr=r2_lr,
        mse_rf=mse_rf,
        r2_rf=r2_rf,
        lr_plot_path=lr_plot_path, lr_error_plot_path=lr_error_plot_path,
        rf_plot_path=rf_plot_path, rf_error_plot_path=rf_error_plot_path,
        user_prediction_lr=user_prediction_lr,
        user_prediction_rf=user_prediction_rf
    )


if __name__ == '__main__':
    app.run(debug=True)
