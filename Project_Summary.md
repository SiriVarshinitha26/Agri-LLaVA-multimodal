# Agri-LLaVA: Crop Disease Diagnosis using Vision-Language Models

## Objective
To assist farmers in identifying crop diseases, pests, and plant health issues using images and natural language in regional languages.
## System Workflow
1. User uploads an image via web interface
2. Image is resized and encoded (Base64)
3. Prompt constructed based on language and query type
4. Image + text sent to Qwen2.5-VL model
5. LoRA adapters inject agricultural knowledge
6. Model generates structured response
7. Confidence score estimated from token probabilities
## Key Mathematical Models

### Vision Transformer Attention
Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V

### LoRA Update
W' = W + αBA

### Cross Entropy Loss
L = −Σ y log(ŷ)
### Confidence Estimation
Confidence = mean(max(softmax(logits))) × 100
## Evaluation Metrics
- Accuracy
## Languages Supported
- English
- हिंदी
- తెలుగు
## Applications
- Farmer advisory systems
- Smart agriculture
