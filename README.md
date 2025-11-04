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
│   └── main.cpp
├── CMakeLists.txt
├── core
│   └── data_types.h
├── data
│   └── preprocessing
│       ├── csv_parser.cpp
│       ├── csv_parser.h
│       ├── data_preprocessor.cpp
│       └── data_preprocessor.h
├── models
│   ├── clustering
│   │   ├── k_means_clusterer.cpp
│   │   └── k_means_clusterer.h
│   └── linear
│       ├── logistic_regression.cpp
│       ├── logistic_regression.h
│       └── multi_linear_regression.cpp
└── README.md


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


Expected training log
```plaintext
Epoch 0   | Loss: 106250.0 | w: 5.0 b: 1.0
Epoch 100 | Loss: 532.4    | w: 4.97 b: -49.8
Epoch 200 | Loss: 310.5    | w: 4.98 b: -50.0
...
Prediction for x=55: ~225
```

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
