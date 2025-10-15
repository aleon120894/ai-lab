# 🚀 AI Demo Project (C++)

This project demonstrates core Machine Learning algorithms, including **Linear Regression**, **Multi Linear Regression** (both implemented via **Gradient Descent**), and **K-Means Clustering** (an unsupervised iterative algorithm), **Logistic regression** all built from scratch in C++.
It is intended as an educational project for understanding the basics of AI/ML without external frameworks.

---

## 📂 Project Structure

```plaintext
ai-lab/
├── src/
│   ├── main.cpp                        # Main program and demo runner
│   ├── k_means_clusterer.cpp           # K-Means core implementation
│   └── multi_linear_regression.cpp     # Multi Linear Regression implementation (Original)
    └── logistic_regression.cpp         # Logistic regression implementation
├── include/
    ├── data_types.h                    # Common structures (e.g., DataPoint)
    ├── k_means_clusterer.h             # K-Means Clusterer class definition
    ├── logistic_regression.h           # Logistic Regression class definition
├── CMakeLists.txt                      # Build configuration
└── README.md                           # Documentation


---

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
./ai_cpp_demo
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