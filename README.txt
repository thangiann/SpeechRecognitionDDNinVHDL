✔Speech Recognition DNN in VHDL
A hardware-optimized Deep Neural Network (DNN) pipeline designed for real-time 
speech phoneme recognition. This repository bridges the gap between PyTorch-based 
acoustic modeling and highly parallel, resource-efficient VHDL implementation.

The architecture features an optimized Diamond Hidden Layer Structure trained on the 
LibriSpeech corpus, designed specifically to yield high accuracy while maintaining clean 
parameterization for Coarse-Grain Sparsification (CGS) on FPGA hardware.

📂 Repository Structure
SpeechRecognitionDDNinVHDL/
├── Python/
│   └── src/
│       └── dnn.py                 # Core model architecture, training, and evaluation loop
├── speech_dnn_weights.pth         # Automatically saved PyTorch model weights (72.67% checkpoint)
└── README.md                      # Project documentation

🧠 Model Architecture
To handle complex phoneme transitional patterns without expanding the external input dimensions, 
the network utilizes a Diamond Hidden Layer Structure. It expands in the middle to extract 
abstract acoustic features and compresses them before the output stage, making it highly 
compatible with FPGA layer scheduling.
Network Configuration
Input Layer: 440 features (13 Mel-frequency Cepstral coefficients or filterbanks + context window of 5 left and 5 right frames)
Hidden Layer 1: 1024 Neurons (Linear -> LayerNorm -> ReLU -> Dropout 0.15)
Hidden Layer 2: 1024 Neurons (Linear -> LayerNorm -> ReLU -> Dropout 0.15)
Hidden Layer 3: 1024 Neurons (Linear -> LayerNorm -> ReLU -> Dropout 0.15)
Hidden Layer 4: 1024 Neurons (Linear -> LayerNorm -> ReLU -> Dropout 0.15)
Output Layer: 71 Output Channels (Phoneme target classification)
