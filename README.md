# 🚀 AI Demo Project (C++)

This project demonstrates core Machine Learning algorithms built **from scratch in C++** — without any external ML frameworks:

✅ Linear Regression  
✅ Multi Linear Regression (Gradient Descent)  
✅ Logistic Regression  
✅ K-Means Clustering (Unsupervised Learning)

Great for learning fundamentals of AI/ML through pure implementation!

---

## 📂 Project Structure

```plaintext
├── app
│   └── main.cpp
├── CMakeLists.txt
├── core
│   └── data_types.h
├── data
│   └── preprocessing
│       ├── csv_parser.cpp
│       ├── csv_parser.h
│       ├── data_preprocessor.cpp
│       └── data_preprocessor.h
├── models
│   ├── clustering
│   │   ├── k_means_clusterer.cpp
│   │   └── k_means_clusterer.h
│   ├── decision_tree
│   │   ├── decision_tree.cpp
│   │   └── decision_tree.h
│   └── linear
│   |    ├── logistic_regression.cpp
│   |    ├── logistic_regression.h
│   |    ├── multi_linear_regression.cpp
│   |    └── multi_linear_regression.h
│   └── neural
│       ├── neural_network.cpp
│       └── neural_network.h
└── README.md
```


## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/aleon120894/ai-lab.git
cd ai-lab
```

Make sure you have CMake and a C++17-compatible compiler installed:
```bash
cmake --version
g++ --version
```

Build the project:
```bash
mkdir build && cd build
cmake ..
make
```

▶️ Usage

Run the executable:
```bash
./ai_lab_demo
```

You will see an interactive menu:
```plaintext 
==============================
  AI Lab — Model Selection
==============================
1. K-Means Clustering
2. Multi-Linear Regression
3. Logistic Regression
4. Decision tree
5. Neural Network
6. Exit
```

Choose a model to run.

## 📊 Example Outputs

### 1️⃣ K-Means Clustering

```plaintext
Running K-Means clustering on demo dataset...

Cluster assignment results:
(1.0, 2.0) → Cluster 0
(5.0, 8.0) → Cluster 1
(9.0, 10.0) → Cluster 2
...

Final centroid positions:
Cluster 0: (1.25, 1.90)
Cluster 1: (6.20, 7.50)
Cluster 2: (9.10, 8.30)
```


### 2️⃣ Multi-Linear Regression

```plaintext
Training Multi-Linear Regression...
Epoch 0   | Loss: 106250.0 | w: 5.0 b: 1.0
Epoch 200 | Loss: 310.5    | w: 4.98 b: -50.0
...
Prediction for x = [55, 3] → ~225
```

### 3️⃣ Logistic Regression

```plaintext
Training Logistic Regression...
Epoch 0   | Loss: 0.69
Epoch 200 | Loss: 0.42
...
Prediction for x = 3.0 → Class: 1 (p = 0.83)
```

### 4️⃣ Decision Tree

```plaintext
Node: f0 < 1
  Node: f1 < 1
    Leaf: class=0
    Leaf: class=1
  Node: f1 < 1
    Leaf: class=1
    Leaf: class=0
Prediction [0,0] -> 0
Prediction [0,1] -> 1
Prediction [1,0] -> 1
Prediction [1,1] -> 0
```

### 5 Neural Network

```plaintext
==============================
   NEURAL NETWORK — XOR DEMO
==============================
Epoch 0 | Loss: 0.130228
Epoch 500 | Loss: 0.0354588
Epoch 1000 | Loss: 0.00383305
Epoch 1500 | Loss: 0.00137337
Epoch 2000 | Loss: 0.000754239
Epoch 2500 | Loss: 0.00049798
Epoch 3000 | Loss: 0.000363197
Epoch 3500 | Loss: 0.00028189
Epoch 4000 | Loss: 0.000228136
Epoch 4500 | Loss: 0.000190376
0 XOR 0 -> p=0.0159469 label=0
0 XOR 1 -> p=0.977828 label=1
1 XOR 0 -> p=0.98469 label=1
1 XOR 1 -> p=0.0179071 label=0
```

## 🧠 Algorithms Implemented

| Model                   | Technique                      | Problem Type     |
|-------------------------|--------------------------------|-------------------|
| Linear Regression       | Gradient Descent               | Regression        |
| Multi-Linear Regression | Gradient Descent + MSE         | Regression        |
| Logistic Regression     | Sigmoid + BCE Loss             | Classification    |
| K-Means                 | Euclidean Distance Clustering  | Unsupervised      |
| Decision Tree           | Gini/Entropy metrics           | Classification    |
| Neural Network          | Backpropagation                | Classification    |

All models are implemented using raw **C++** and **STL containers**, without external ML frameworks.

---

## 🔮 Roadmap / Planned Features

✅ CSV dataset loading support  
🔄 Command-line args for batch mode  
🔄 PCA dimensionality reduction  
🔄 Training metrics visualization  
🔄 CI/CD with GitHub Actions  

---

## 🤝 Contributing

Pull Requests are welcome!  
Feel free to extend or improve the learning demos 😊

📓 Notes

This project implements fundamental ML algorithms:

Linear Regression: Uses Gradient Descent with Mean Squared Error.

K-Means: Uses Euclidean distance for assignment and mean calculation for centroid update.

Datasets are currently small and hardcoded inside main.cpp.

The project is extendable for bigger datasets and file input.


🔮 Optional Extensions

✅ Add file-based dataset loading (CSV parser)
✅ Implement Logistic Regression
✅ Use Eigen library for matrix/vector math
✅ Wrap into a small C++ API for inference
✅ Add CI/CD with GitHub Actions to test compilation


📜 License

This project is released under the MIT License.
Feel free to use and modify it for learning or research purposes.
