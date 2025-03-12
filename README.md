# 🚗 Vehicle Detection using Background Subtraction

This project implements **vehicle detection** using the **background subtraction** technique in both **Matlab** and **Python**. It is based on processing a traffic image sequence to identify moving vehicles.

## 📌 Features

✅ Load and preprocess the dataset (traffic sequence)  
✅ Compute the **background model** (mean & standard deviation)  
✅ Detect vehicles by **subtracting the background**  
✅ Improve detection using **adaptive thresholding**  
✅ Generate **segmented binary masks**  
✅ Save results as **video output**

## 🛠 Technologies Used

- **Matlab** 🟡
- **Python** 🐍
- **OpenCV** 👁
- **NumPy** 🔢
- **Matplotlib** 📊

## 📂 Folder Structure

```
/vehicle-detection
│── /matlab-code        # Matlab implementation
│── /python-code        # Python implementation
│── /dataset            # Traffic sequence images (gitignore)
│── README.md           # This file
│── results/            # Processed images and videos
```

## 🚀 How to Run

### **1. Clone the repository**

```bash
git clone https://github.com/ChengjiePL/Background-Subtraction && cd Background-Subtraction
```

### **2. Get the Dataset**

```bash
curl -O http://jacarini.dinf.usherbrooke.ca/static/dataset/baseline/highway.zip

# or with wget

wget http://jacarini.dinf.usherbrooke.ca/static/dataset/baseline/highway.zip
```

### 🔹 **Matlab Version**

1️⃣ Open Matlab  
2️⃣ Navigate to the `/matlab-code/` folder  
3️⃣ Run:

```matlab
run main.m
```

### 🔹 **Python Version**

1️⃣ Install dependencies:

```bash
pip install opencv-python numpy matplotlib scikit-image
```

2️⃣ Run the main script:

```bash
python main.py
```

## 🎥 Example Output

Here's an example of the vehicle detection output:

[segmentacion_coches.webm](https://github.com/user-attachments/assets/165c51c2-e185-417b-861d-4a957243e364)

## 🛡 License

This project is open-source under the **MIT License**.
