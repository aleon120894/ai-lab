# 🚀 AI Demo Project (C++)

This project demonstrates a simple **Linear Regression** and **Multi Linear Regression** implemented from scratch in C++ using **Gradient Descent**.  
It is intended as an educational project for understanding the basics of AI/ML without external frameworks.

---

## 📂 Project Structure

```plaintext
ai-cpp-demo/
├── src/
│   └── main.cpp                           # Main program with linear regression implementation
    └── multi_linear_regression.cpp        # Program with multi linear regression implementation
├── CMakeLists.txt                         # Build configuration
├── README.md                              # Documentation
└── .gitignore                             # Ignored files


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

The model is a simple Linear Regression (y = wx + b)

Optimization uses Gradient Descent with Mean Squared Error

Dataset is small and hardcoded inside main.cpp

Extendable for bigger datasets and file input

🔮 Optional Extensions

✅ Add file-based dataset loading (CSV parser)

✅ Implement Logistic Regression

✅ Use Eigen library for matrix/vector math

✅ Wrap into a small C++ API for inference

✅ Add CI/CD with GitHub Actions to test compilation

📜 License

This project is released under the MIT License.
Feel free to use and modify it for learning or research purposes.