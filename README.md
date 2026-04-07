# TECS Engine: Element-Slot Network (ESN)

TECS (Time Element Cycle Slot engine) is a high-performance simulation and training engine for **Element-Slot interaction**. It provides a hybrid architecture that bridges discrete dynamic system modeling with modern neuro-symbolic learning.

## 🚀 Core Philosophy: Element-Slot Interaction (ESI)

At its heart, TECS operates on the principle of **Elements** being dynamically routed into **Slots**. This is not a simple attention mechanism; it's a structural routing system where:
- **Elements** represent data units (e.g., image patches, data indices).
- **Slots** act as shared memory and aggregation units with finite capacity.
- **Routing** is governed by dynamic priorities, conflict resolution, and stochastic stalling.

## 🛠️ Architecture & Mechanics

### 1. Elements & Groups
Elements are dynamic entities defined by:
- **Weight (w)**: Significance in the slot aggregation.
- **Period (k)**: The rate of movement through allowed slots.
- **Aging Factor**: A mechanism that increases an element's priority if it's repeatedly blocked, preventing starvation and deadlocks.
- **Stall Probability**: A chance for the element to remain in its current slot instead of moving, introducing natural stochasticity.

### 2. Slots & Aggregation
Slots are the shared workspace. The engine supports:
- **Aggregation**: Elements in the same slot can be aggregated (summed, averaged, or hashed).
- **Conflict Resolution**: Preventing overlaps between disparate groups of elements based on priority levels.
- **Dynamic Shift**: A global temporal shift that rotates the available slots, simulating a moving reference system.

### 3. Routing Logic (Neuro-Symbolic)
The neural implementation (`ElementSlotLayer`) uses **Gumbel-Softmax** to learn the optimal routing from inputs to slots. It includes:
- **Load Balancing**: Routing loss terms that ensure slots are utilized evenly.
- **Memory Integration**: Recurrent updates to slot memory during inference.

---

## 🧠 Hybrid System: Simulation + Training

TECS provides two primary operational modes:

### **Mode A: Discrete Simulation (`llm_scripts`)**
A pure simulation environment to test the dynamics of the Element-Slot model. Useful for:
- Agent-based modeling.
- Studying complex traffic or resource allocation patterns.
- Testing deterministic vs. stochastic routing rules.

### **Mode B: Deep Learning Training (`ElementSlotNetwork`)**
Integration with PyTorch for end-to-end learning. 
- **Task Example**: Digit classification on **MNIST**.
- **Process**: The network partitions images into patches (elements), routes them to internal slots, aggregates information, and performs classification.
- **Performance**: High accuracy with GPU (CUDA) acceleration and optimized routing losses.

---

## 📂 Project Structure

- `llm_scripts/`: Core simulation logic, including hybrid models and CUDA-ready versions.
- `ElementSlotNetwork_LLM_training/`: Training scripts, weights (`best_element_slot.pt`), and MNIST dataset logic.
- `tests/`: Correctness tests for the simulation engine.
- `Discrete_Dynamic_System.pdf`: Theoretical background and mathematical foundations.

---

## ⚙️ Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jandos77/TECS_engine.git
   cd TECS_engine
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Running the Simulation:**
```bash
python llm_scripts/tecs_LLM_Hybrid.py
```

**Training/Fine-tuning the ESN:**
```bash
python ElementSlotNetwork_LLM_training/ElementSlotNetwork_LLM_Dataset_training.py
```

---

## 📜 License
TECS_engine is licensed under the **MIT License**.