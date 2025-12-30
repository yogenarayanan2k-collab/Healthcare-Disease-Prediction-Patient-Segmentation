# HEALTHCARE DISEASE PREDICTION PROJECT
# Supervised + Unsupervised Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("HEALTHCARE DISEASE PREDICTION PROJECT")
print("(Supervised + Unsupervised Machine Learning)")

# 1. LOAD DATASET

df = pd.read_csv("Dataset/cardio_train.csv", sep=';')
print("Dataset Shape:", df.shape)
print(df.head())

# 2. BASIC EXPLORATION

print(df.info())
print(df.describe())

# 3. FEATURE ENGINEERING

df['age_years'] = (df['age'] / 365).astype(int)
df.drop('age', axis=1, inplace=True)

# 4. TARGET VISUALIZATION

sns.countplot(x='cardio', data=df)
plt.title("Heart Disease Distribution")
plt.show()

# 5. CORRELATION HEATMAP

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 6. OUTLIER DETECTION

plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Outlier Visualization")
plt.show()

# 7. PREPROCESSING

X = df.drop('cardio', axis=1)
y = df['cardio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 8. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#===================================
# SUPERVISED MODEL 1: RANDOM FOREST
#===================================

rf = RandomForestClassifier(random_state=42)

rf_params = {
    'n_estimators': [100],
    'max_depth': [10, 20]
}

rf_grid = GridSearchCV(rf, rf_params, cv=5)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

sns.heatmap(confusion_matrix(y_test, y_pred_rf),
            annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()

# ============================================
# SUPERVISED MODEL 2: SUPPORT VECTOR MACHINE
# ============================================

svc = SVC()

svc_params = {
    'C': [1, 10],
    'kernel': ['rbf']}

svc_grid = GridSearchCV(svc, svc_params, cv=3, n_jobs=-1)
svc_grid.fit(X_train, y_train)

svc_best = svc_grid.best_estimator_
y_pred_svc = svc_best.predict(X_test)

print("SVM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

sns.heatmap(confusion_matrix(y_test, y_pred_svc),
            annot=True, fmt='d', cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.show()

# ===============================
# UNSUPERVISED LEARNING: K-MEANS
# ===============================

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters
print(df[['Cluster']].head())

# PCA Visualization
# ==================

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=clusters, cmap='viridis')
plt.title("Patient Segmentation using K-Means (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()

print("Healthcare ML Project Completed Successfully!")
