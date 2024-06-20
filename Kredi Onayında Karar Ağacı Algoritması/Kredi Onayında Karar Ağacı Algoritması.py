import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Veriyi yükleme ve hazırlama
file_path = 'D:/Downloads/creditapproval/crx.data'  # Veri dosyasının yolu
columns = [
    'Başvuranın tipi', 'Başvuranın yaşı', 'Başvuranın işinin karakteri', 'Başvuranın gelir durumu',
    'Başvuranın varlık değeri', 'Birlikte başvurduğu kişinin tipi', 'Birlikte başvurduğu kişinin yaşı',
    'Birlikte başvurduğu kişinin işinin karakteri', 'Birlikte başvurduğu kişinin gelir durumu', 'Birlikte başvurduğu kişinin varlık değeri',
    'Kredi miktarı', 'Kredi süresi', 'Ödeme planı', 'Aylık gelir', 'Mevcut varlık değeri', 'Kredi başvurusunun onay durumu (+/-)'
]
data = pd.read_csv(file_path, header=None, names=columns, na_values='?')  # Veriyi yükleme

# Eksik verileri doldurma
data.fillna({
    'Başvuranın tipi': 'a', 'Başvuranın yaşı': 0, 'Başvuranın işinin karakteri': 'u', 'Başvuranın gelir durumu': 'g',
    'Başvuranın varlık değeri': 'p', 'Birlikte başvurduğu kişinin tipi': 'a', 'Birlikte başvurduğu kişinin yaşı': 0,
    'Birlikte başvurduğu kişinin işinin karakteri': 'u', 'Birlikte başvurduğu kişinin gelir durumu': 'g', 'Birlikte başvurduğu kişinin varlık değeri': 'p',
    'Kredi miktarı': 0, 'Kredi süresi': 0, 'Ödeme planı': 't', 'Aylık gelir': 0, 'Mevcut varlık değeri': 0
}, inplace=True)  # Eksik değerleri uygun varsayılan değerlerle doldurma

# Tüm kategorik verileri string'e dönüştürme
categorical_features = data.select_dtypes(include=['object']).columns
data[categorical_features] = data[categorical_features].astype(str)  # Kategorik verileri string'e dönüştürme

# Hedef değişkeni belirleme ve eksik değerleri kaldırma
data = data.dropna(subset=['Kredi başvurusunun onay durumu (+/-)'])  # Eksik hedef değişkenleri kaldırma
y = data['Kredi başvurusunun onay durumu (+/-)']  # Hedef değişken
X = data.drop('Kredi başvurusunun onay durumu (+/-)', axis=1)  # Özellikler

# Eksik verileri doldurma ve kategorik verileri one-hot encoding yapma
numeric_features = X.select_dtypes(include=['number']).columns.tolist()      # Sayısal özellikler
categorical_features = X.select_dtypes(include=['object']).columns.tolist()  # Kategorik özellikler

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Eksik sayısal verileri medyan ile doldurma
    ('scaler', StandardScaler())                    # Sayısal verileri ölçeklendirme
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Eksik kategorik verileri en sık görülen ile doldurma
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Kategorik verileri one-hot encoding ile dönüştürme
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),         # Sayısal veriler için dönüşümler
        ('cat', categorical_transformer, categorical_features)  # Kategorik veriler için dönüşümler
    ])

# Ön işlemeyi ve model oluşturmayı bir pipeline ile birleştirme
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', DecisionTreeClassifier(random_state=42))])  # Pipeline oluşturma

# Hiperparametre optimizasyonu için GridSearchCV kullanma
param_grid = {
    'classifier__max_depth': [3, 5, 7, 10],             # Maksimum derinlik parametreleri
    'classifier__min_samples_split': [2, 5, 10],        # Minimum örnek bölme parametreleri
    'classifier__min_samples_leaf': [1, 2, 5],          # Minimum yaprak örneği parametreleri
    'classifier__max_features': [None, 'sqrt', 'log2']  # Maksimum özellik parametreleri
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')  # GridSearchCV ile hiperparametre optimizasyonu
grid_search.fit(X, y)  # Modeli eğitme

# En iyi modeli seçme
best_model = grid_search.best_estimator_

# Başlangıç doğruluğunu hesapla
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Eğitim ve test verilerini ayırma
best_model.fit(X_train, y_train)  # Modeli eğitme
y_pred = best_model.predict(X_test)  # Test verileri üzerinde tahmin yapma
initial_accuracy = accuracy_score(y_test, y_pred)  # Başlangıç doğruluğunu hesaplama

# Doğruluk değerlerini saklama
accuracies = [initial_accuracy]

# Tkinter arayüzü oluşturma
def retrain_and_evaluate_model():
    try:
        # Kullanıcıdan alınan verileri al ve doğrula
        input_data = []
        for entry, column in zip(entries, X.columns):
            value = entry.get()
            if column in categorical_features:
                input_data.append(value)
            else:
                input_data.append(float(value))
        
        # Kullanıcı girişlerini dönüştürme
        input_df = pd.DataFrame([input_data], columns=X.columns)
        
        # Önceden fit edilmiş pipeline kullanarak kullanıcı girişlerini dönüştürme
        input_transformed = best_model.named_steps['preprocessor'].transform(input_df)
        
        # Model tahmini yapma
        new_prediction = best_model.named_steps['classifier'].predict(input_transformed)[0]
        
        # Yeni tahmin ile doğruluk hesaplama
        y_test_with_new = list(y_test) + [new_prediction]
        y_pred_with_new = list(y_pred) + [new_prediction]
        accuracy_with_new = accuracy_score(y_test_with_new, y_pred_with_new)
        accuracies.append(accuracy_with_new)
        
        # Doğruluk değerlerini pencereye yazdırma
        initial_accuracy_label.config(text=f"Başlangıç Model Doğruluğu: {initial_accuracy * 100:.2f}%")
        accuracy_label.config(text=f"Model Doğruluğu: {accuracy_with_new * 100:.2f}%")
        
        # Kredi başvurusunun sonucunu yazdırma
        if new_prediction == 1:  # 'Kabul Edildi' durumunu temsil eden değeri değiştirin
            result_label.config(text="Kredi Onayı Başvurunuz Kabul Edildi")
        else:
            result_label.config(text="Kredi Onayı Başvurunuz Reddedildi")
        
        # Grafik güncelleme
        update_plot()
    except ValueError as e:
        messagebox.showerror("Hata", str(e))

def update_plot():
    # Grafik verilerini güncelleme
    ax.clear()                       # Grafiği temizleme
    ax.plot(accuracies, marker='o')  # Doğruluk verilerini grafiğe ekleme
    ax.set_title("Model Doğruluğu")  # Grafik başlığı
    ax.set_xlabel("Yineleme")        # X ekseni etiketi
    ax.set_ylabel("Doğruluk")        # Y ekseni etiketi
    ax.grid(True)                    # Izgara çizgilerini gösterme
    canvas.draw()                    # Grafiği çizme

root = tk.Tk()                               # Ana pencereyi oluşturma
root.title("Kredi Başvurusu Değerlendirme")  # Pencere başlığını ayarlama

input_frame = tk.Frame(root)                                 # Girdi çerçevesi oluşturma
input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)  # Girdi çerçevesini sola yerleştirme

labels = X.columns  # Veri sütunlarını etiketler olarak alma
entries = []        # Girdi kutuları listesi

for label in labels:
    frame = tk.Frame(input_frame)                  # Her etiket için bir çerçeve oluşturma
    frame.pack(fill=tk.X, pady=2)                  # Çerçeveyi yatay olarak yerleştirme
    lbl = tk.Label(frame, text=label, width=30)    # Etiket oluşturma
    lbl.pack(side=tk.LEFT, padx=5, pady=5)         # Etiketi sola yerleştirme
    entry = tk.Entry(frame)                        # Girdi kutusu oluşturma
    entry.pack(side=tk.LEFT, padx=5, expand=True)  # Girdi kutusunu sola yerleştirme ve genişletme
    
    # Uygun değerleri gösteren etiket ekleme
    if label in categorical_features:
        allowed_values = ', '.join(map(str, X[label].dropna().unique()))  # Kategorik değerleri alma
        hint = tk.Label(frame, text=f"({allowed_values})", fg='grey')     # Kategorik değerleri gösteren etiket oluşturma
    else:
        hint = tk.Label(frame, text="(numerik)", fg='grey')  # Sayısal veri için etiket oluşturma
    
    hint.pack(side=tk.LEFT, padx=5, pady=5)  # Uygun değerler etiketini sola yerleştirme
    entries.append(entry)                    # Girdi kutusunu listeye ekleme

initial_accuracy_label = tk.Label(input_frame, text=f"Başlangıç Model Doğruluğu: {initial_accuracy * 100:.2f}%")
initial_accuracy_label.pack(pady=5)  # Başlangıç doğruluğunu gösteren etiketi yerleştirme

accuracy_label = tk.Label(input_frame, text="")  # Model doğruluğunu gösteren etiketi oluşturma
accuracy_label.pack(pady=5)                      # Model doğruluğu etiketini yerleştirme

result_label = tk.Label(input_frame, text="")  # Kredi başvuru sonucunu gösteren etiketi oluşturma
result_label.pack(pady=5)                      # Kredi başvuru sonucu etiketini yerleştirme

btn = tk.Button(input_frame, text="Modeli Kontrol Et", command=retrain_and_evaluate_model)  # Buton oluşturma
btn.pack(pady=10)                                                                           # Butonu yerleştirme

# Matplotlib grafiği oluşturma
fig = Figure(figsize=(6, 4), dpi=100)                                  # Grafik figürü oluşturma
ax = fig.add_subplot(111)                                              # Alt grafik oluşturma
canvas = FigureCanvasTkAgg(fig, master=root)                           # Grafik tuvalini ana pencereye bağlama
canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Grafik tuvalini sağa yerleştirme

# İlk grafiği çizme
update_plot()  # Grafiği çizme

root.mainloop()  # Ana pencere döngüsünü başlatma
