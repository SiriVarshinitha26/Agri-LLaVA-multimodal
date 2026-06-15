# 🌾 Agri-LLaVA – Multilingual and Voice Assistant Crop Disease Diagnosis System

Agri-LLaVA is a Vision-Language AI system designed to help farmers identify crops, pests, and plant diseases using images and natural language queries in **English, Hindi, and Telugu**.

The system combines:
- Qwen 2.5 Vision-Language Model (7B)
- LoRA fine-tuning for agricultural disease knowledge
- Flask backend API
- Web-based multilingual UI with speech support

---

## 🚀 Features

- 📸 Image-based crop, pest, and disease identification
- 🌱 Disease symptoms, causes, and treatment suggestions
- 🗣️ Multilingual support (English | हिंदी | తెలుగు)
- 🔊 Text-to-Speech for farmer-friendly interaction
- 🧠 LoRA adapters for domain-specific fine-tuning
- ♻️ Context-aware follow-up questions
- ⚡ Optimized for low VRAM and faster inference

---

## 🧠 Model Architecture

- **Base Model:** Qwen2.5-VL (7B)
- **Fine-Tuning:** LoRA (Low-Rank Adaptation)
- **Input:** Image + Text
- **Output:** Natural language explanation

---

## 📊 Mathematical Foundations

- Vision Transformer (ViT)
- Multi-Head Self Attention
- Cross-Entropy Loss
- AdamW Optimization
- Softmax-based Confidence Estimation
- Precision, Recall, F1-score evaluation

(See `docs/Project_Summary.md` for formulas)

---

## 📂 Dataset Used

**Agri-LLaVA Dataset**
- Agricultural pests and disease images
- Instruction-tuned conversations

🔗 Dataset link:
https://huggingface.co/datasets/Agri-LLaVA-Anonymous/Agricultural_pests_and_diseases_instruction_tuning_data

---

## ⚙️ Installation & Setup

### 1️⃣ Install Ollama
```bash
ollama serve
ollama pull qwen2.5vl:7b
### 2️⃣ Install Python Dependencies
pip install -r requirements.txt
###3️⃣ Run the Application
python app1.py
###Open browser: 
http://localhost:5000
 ###    Technologies Used
 Python
Flask
Ollama
Qwen2.5-VL-7B-instruct
LoRA (PEFT)
HTML, CSS, JavaScript
Google Translate TTS (Telugu)


###User Interface
Upload leaf/crop images
Ask questions via text or voice
Receive spoken responses
Supports farmer-friendly language


