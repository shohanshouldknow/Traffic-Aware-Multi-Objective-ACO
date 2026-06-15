# 🚀 TA-MOACO: Traffic-Aware Multi-Objective Ant Colony Optimization for Green Data Centers

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Research](https://img.shields.io/badge/Research-Academic-green)
![License](https://img.shields.io/badge/License-MIT-orange)

## 📖 Overview

This project presents a **faculty-ready implementation** of the research paper:

> **Traffic-Aware Multi-Objective Ant Colony Optimization (TA-MOACO) for Minimizing Data Center Network and Server Power Consumption**

The implementation demonstrates how **Virtual Machine Placement (VMP)** can simultaneously optimize:

- 🖥 Server Energy Consumption
- 🌐 Network Switch Energy Consumption
- 📦 VM Traffic Locality
- ⚡ Overall Data Center Energy Efficiency

Unlike traditional server consolidation approaches, TA-MOACO considers **inter-VM communication traffic**, reducing both **server power** and **network infrastructure power**.

---

# 🎯 Objectives

- Reduce total data center power consumption
- Minimize active physical machines
- Minimize active network switches
- Localize VM communication
- Reduce average hop count
- Improve energy efficiency without violating SLA constraints

---

# 🏗 System Architecture

```
                  +--------------------+
                  |   Real Workload    |
                  |     Datasets       |
                  +---------+----------+
                            |
                            v
                  +--------------------+
                  | Data Preprocessing |
                  +---------+----------+
                            |
                            v
                  +--------------------+
                  | Traffic Matrix      |
                  | Generation          |
                  +---------+----------+
                            |
                            v
                  +--------------------+
                  | TA-MOACO Engine     |
                  +---------+----------+
                            |
           +----------------+----------------+
           |                |                |
           v                v                v
      Server Power    Network Power    VM Placement
           |                |                |
           +----------------+----------------+
                            |
                            v
                  Energy Optimization Result
```

---

# ✨ Features

- ✅ TA-MOACO implementation
- ✅ First Fit Decreasing (FFD)
- ✅ ACS-VMP
- ✅ Traffic-Aware VMP
- ✅ Fat-Tree Data Center Topology
- ✅ Server Power Model
- ✅ Network Switch Power Model
- ✅ Real Cloud Workload Support
- ✅ Automatic Traffic Matrix Inference
- ✅ Interactive Dashboard
- ✅ HTML Report Generation
- ✅ High-Resolution Graphs
- ✅ Faculty Presentation Mode

---

# 📂 Project Structure

```
TA-MOACO/
│
├── src/
│   ├── algorithms/
│   ├── simulation.py
│   ├── topology.py
│   ├── power_model.py
│   ├── visualization.py
│   ├── real_data.py
│   └── main.py
│
├── local_dashboard.py
├── app.py
├── sample_real_trace.csv
├── results/
├── requirements.txt
└── README.md
```

---

# 📊 Implemented Algorithms

| Algorithm | Server Aware | Traffic Aware | Network Power |
|------------|-------------|--------------|---------------|
| FFD | ✅ | ❌ | ❌ |
| ACS-VMP | ✅ | ❌ | ❌ |
| TA-VMP | ✅ | ✅ | Partial |
| **TA-MOACO** | ✅ | ✅ | ✅ |

---

# ⚙ TA-MOACO Highlights

The proposed algorithm combines:

- Ant Colony Optimization
- Resource Heuristic
- Traffic Affinity Heuristic
- Multi-objective Optimization
- Pheromone Learning
- Network-aware VM Placement

Objective Function:

```
Minimize:

Total Power =
Server Power
+
Network Switch Power
```

---

# 📈 Evaluation Metrics

The system evaluates:

- Active Physical Machines
- Active Switches
- Server Power
- Network Power
- Total Data Center Power
- Average Hop Count
- Network Localization Score
- SLA Violations
- Energy Savings

---

# 📊 Generated Visualizations

- Energy Comparison
- Server vs Network Power
- Active PM Comparison
- Active Switch Comparison
- Hop Count Comparison
- Traffic Heatmap
- Switch Activation Matrix
- TA-MOACO Convergence Curve
- VM Placement Map
- Rack Utilization
- CPU Utilization
- RAM Utilization

---

# 🌍 Real Dataset Support

The framework supports:

- Bitbrains VM Traces
- Alibaba Cluster Trace
- Generic Cloud Workload CSV
- Custom VM Traces

Automatic preprocessing includes:

- Column Detection
- Missing Value Handling
- Traffic Matrix Inference
- Resource Normalization

---

# 🖥 Dashboard Features

The interactive dashboard provides:

- Dataset Preview
- Synthetic and Real Trace Modes
- Parameter Selection
- Progress Tracking
- Algorithm Comparison
- Energy Metrics
- Visualization Gallery
- HTML Report Export

---

# ▶️ Installation

```bash
git clone https://github.com/yourusername/TA-MOACO.git

cd TA-MOACO

pip install -r requirements.txt
```

---

# ▶️ Run CLI

```bash
python src/main.py
```

---

# ▶️ Run Dashboard

```bash
python local_dashboard.py
```

Open:

```
http://localhost:8501
```

---

# 📚 Research Contribution

This implementation demonstrates that incorporating **traffic locality into virtual machine placement** enables significant reductions in **network switch power consumption** while maintaining efficient server utilization.

The proposed TA-MOACO framework provides a practical approach for **Green Cloud Computing** and **Energy-Efficient Data Center Management**.

---

# 🔬 Future Work

- SDN Integration
- Live VM Migration
- Kubernetes Scheduler Integration
- Reinforcement Learning Optimization
- LSTM-Based Traffic Prediction
- Multi-Data Center Optimization
- Digital Twin Simulation

---

# 👨‍💻 Authors

**Kamruzzaman Shohan**  
Department of Computer Science & Engineering  
United International University (UIU)

Research Area:

- Green Computing
- Cloud Computing
- Artificial Intelligence
- Optimization Algorithms
- Energy-Efficient Data Centers

---

# ⭐ If you find this project useful, please consider giving it a star!
