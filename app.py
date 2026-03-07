import tkinter as tk
from tkinter import messagebox
import joblib

try:
    model = joblib.load("model/decision_tree.pkl")
    le = joblib.load("model/label_encoder.pkl")
except Exception as e:
    print("Error loading model:", e)

def recommend_crop():
    try:
        N = float(entry_N.get())
        P = float(entry_P.get())
        K = float(entry_K.get())
        temp = float(entry_temperature.get())
        humidity = float(entry_humidity.get())
        ph = float(entry_ph.get())
        rainfall = float(entry_rainfall.get())
        input_data = [[N, P, K, temp, humidity, ph, rainfall]]
        prediction = le.inverse_transform(model.predict(input_data))
        messagebox.showinfo("Recommended Crop", f"Recommended Crop: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("AI Crop Recommendation System")
root.geometry("400x400")

labels = ["Nitrogen (N)", "Phosphorous (P)", "Potassium (K)", 
          "Temperature (°C)", "Humidity (%)", "Soil pH", "Rainfall (mm)"]
entries = []

for i, label_text in enumerate(labels):
    tk.Label(root, text=label_text).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

entry_N, entry_P, entry_K, entry_temperature, entry_humidity, entry_ph, entry_rainfall = entries

tk.Button(root, text="Recommend Crop", command=recommend_crop, bg="green", fg="white").grid(row=8, column=0, columnspan=2, pady=20)

root.mainloop()