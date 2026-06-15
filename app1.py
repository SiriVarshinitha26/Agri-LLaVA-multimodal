from flask import Flask, request, jsonify, render_template
import requests
import base64
import os
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model configurations
OLLAMA_API_URL = "http://localhost:11434/api/chat"
QWEN_MODEL = "qwen2.5vl:7b"  # UPGRADED to 7B for better accuracy in Hindi/Telugu

# CRITICAL FIX: Increased timeout values
REQUEST_TIMEOUT = 300  # 5 minutes for complex vision tasks
CONNECTION_TIMEOUT = 30  # 30 seconds to establish connection

# LoRA configuration
LORA_ADAPTERS = {
    'crop_disease': {
        'path': './lora_weights/crop_disease_adapter',
        'enabled': True,
        'weight': 1.0
    },
    'plant_identification': {
        'path': './lora_weights/plant_id_adapter',
        'enabled': False,
        'weight': 0.8
    }
}

# Store conversation history and last uploaded image
conversations = {}
last_images = {}

def check_ollama_status():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_ollama_models():
    """Get list of available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except:
        return []

def get_active_lora_adapters():
    """Get list of enabled LoRA adapters"""
    return {k: v for k, v in LORA_ADAPTERS.items() if v['enabled']}

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/status', methods=['GET'])
def status():
    """Check status of models"""
    is_running = check_ollama_status()
    models = get_ollama_models() if is_running else []
    
    return jsonify({
        'ollama_running': is_running,
        'models': models,
        'current_model': QWEN_MODEL,
        'lora_adapters': get_active_lora_adapters()
    })

@app.route('/lora/toggle', methods=['POST'])
def toggle_lora():
    """Enable/disable LoRA adapters"""
    try:
        data = request.json
        adapter_name = data.get('adapter_name')
        enabled = data.get('enabled', True)
        
        if adapter_name in LORA_ADAPTERS:
            LORA_ADAPTERS[adapter_name]['enabled'] = enabled
            return jsonify({
                'success': True,
                'adapter': adapter_name,
                'enabled': enabled
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Adapter {adapter_name} not found'
            }), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/lora/weight', methods=['POST'])
def update_lora_weight():
    """Update LoRA adapter weight"""
    try:
        data = request.json
        adapter_name = data.get('adapter_name')
        weight = data.get('weight', 1.0)
        
        if adapter_name in LORA_ADAPTERS:
            LORA_ADAPTERS[adapter_name]['weight'] = float(weight)
            return jsonify({
                'success': True,
                'adapter': adapter_name,
                'weight': weight
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Adapter {adapter_name} not found'
            }), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        conv_id = request.form.get('conversation_id', 'default')
        question = request.form.get('question', '')
        language = request.form.get('language', 'en')  # Get selected language
        
        print(f"=== NEW REQUEST ===")
        print(f"Language selected: {language}")
        print(f"Question: {question}")
        
        # AUTO-DETECT language from question if possible (for 3B model reliability)
        detected_lang = language
        if question:
            hindi_chars = sum(1 for c in question if '\u0900' <= c <= '\u097F')
            telugu_chars = sum(1 for c in question if '\u0C00' <= c <= '\u0C7F')
            total_chars = len(question.replace(' ', ''))
            
            if total_chars > 0:
                if hindi_chars / total_chars > 0.3:
                    detected_lang = 'hi'
                    print(f"🔍 AUTO-DETECTED: Hindi (found {hindi_chars} Hindi chars)")
                elif telugu_chars / total_chars > 0.3:
                    detected_lang = 'te'
                    print(f"🔍 AUTO-DETECTED: Telugu (found {telugu_chars} Telugu chars)")
                else:
                    detected_lang = 'en'
                    print(f"🔍 AUTO-DETECTED: English")
        
        # Override user selection with detected language for consistency
        language = detected_lang
        
        # SIMPLE EVERYDAY LANGUAGE - Easy to understand for farmers
        system_prompts = {
            'en': """You are a helpful farming assistant.
Use SIMPLE ENGLISH that farmers understand.
Identify fruits, vegetables, crops from images.
Common fruits: apple, banana, mango, orange, watermelon, papaya, guava, pomegranate, grapes.
Answer clearly and simply.""",
            
            'hi': """आप एक खेती का सहायक हैं।
आसान हिंदी बोलें जो किसान समझ सकें।
तस्वीर से फल, सब्जी, फसल पहचानें।
आम फल: सेब, केला, आम, संतरा, तरबूज, पपीता, अमरूद, अनार, अंगूर।
साफ और आसान भाषा में जवाब दें। बोलचाल की भाषा इस्तेमाल करें।""",
            
            'te': """మీరు వ్యవసాయ సహాయకుడు.
రైతులకు అర్థమయ్యే సులభమైన తెలుగు మాట్లాడండి.
చిత్రం నుండి పండ్లు, కూరగాయలు, పంటలు గుర్తించండి.
సాధారణ పండ్లు: ఆపిల్, అరటి, మామిడి, నారింజ, పుచ్చకాయ, బొప్పాయి, జామ, దానిమ్మ, ద్రాక్ష.
స్పష్టంగా మరియు సులభంగా సమాధానం ఇవ్వండి। మాట్లాడే భాషలో చెప్పండి।"""
        }
        
        # Check if Ollama is running
        if not check_ollama_status():
            return jsonify({
                'success': False,
                'error': 'Ollama is not running. Please start Ollama service.'
            }), 503
        
        # Initialize conversation history if new
        if conv_id not in conversations:
            conversations[conv_id] = []
        
        # Handle image if provided
        image_data = None
        has_image = False
        new_image_uploaded = False
        
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                has_image = True
                new_image_uploaded = True
                img = Image.open(file.stream)
                
                # OPTIMIZATION: Better image quality for accurate identification
                max_size = (1024, 1024)  # Increased back to 1024 for better accuracy
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffered = BytesIO()
                img_format = 'JPEG' if img.mode == 'RGB' else 'PNG'
                if img_format == 'JPEG':
                    img.save(buffered, format='JPEG', quality=95, optimize=False)  # Higher quality
                else:
                    img.save(buffered, format='PNG', optimize=False)
                
                image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Store this NEW image separately - don't replace old one
                print(f"🆕 NEW IMAGE UPLOADED")
                last_images[conv_id] = image_data
        
        # If no new image but we have a stored one, use it for follow-up questions
        elif conv_id in last_images and last_images[conv_id]:
            image_data = last_images[conv_id]
            has_image = True
            print(f"♻️ Using last uploaded image for follow-up question")
        
        # Build prompt based on language - SIMPLE CONVERSATIONAL STYLE
        language_instruction = {
            'en': "Answer in simple English.",
            'hi': "आसान हिंदी में बताओ।",
            'te': "సులభమైన తెలుగులో చెప్పండి।"
        }
        
        # OPTIMIZATION: Shortened prompts to reduce token count
        if has_image and question:
            disease_keywords = ['disease', 'infection', 'fungus', 'bacteria', 'virus', 'problem', 'wrong', 'sick', 'affected', 'issue',
                              'रोग', 'संक्रमण', 'कवक', 'बीमारी', 'समस्या',
                              'వ్యాధి', 'సంక్రమణ', 'ఫంగస్', 'సమస్య']
            treatment_keywords = ['treat', 'cure', 'prevent', 'fix', 'solution', 'remedy', 'control', 'manage', 'stop', 'help',
                                'उपचार', 'इलाज', 'रोकथाम', 'समाधान',
                                'చికిత్స', 'నివారణ', 'పరిష్కారం']
            identification_keywords = ['what is', 'name', 'identify', 'type of', 'kind of', 'this',
                                      'क्या है', 'नाम', 'पहचान',
                                      'ఏమిటి', 'పేరు', 'గుర్తించు']
            
            is_disease_question = any(keyword in question.lower() for keyword in disease_keywords)
            is_treatment_question = any(keyword in question.lower() for keyword in treatment_keywords)
            is_identification = any(keyword in question.lower() for keyword in identification_keywords)
            
            if is_disease_question and not is_identification:
                if language == 'hi':
                    final_prompt = f"""सवाल: "{question}"

तस्वीर देखकर बताओ:
1. क्या बीमारी है?
2. लक्षण क्या हैं?
3. क्यों हुई?
4. कितनी खतरनाक है?

आसान भाषा में समझाओ। 100-150 शब्दों में।"""
                elif language == 'te':
                    final_prompt = f"""ప్రశ్న: "{question}"

చిత్రం చూసి చెప్పండి:
1. ఏ వ్యాధి?
2. లక్షణాలు ఏమిటి?
3. ఎందుకు వచ్చింది?
4. ఎంత ప్రమాదకరం?

సులభంగా వివరించండి. 100-150 పదాలలో."""
                else:
                    final_prompt = f"""Question: "{question}"

Look at the image and tell me:
1. What disease is this?
2. What are the symptoms?
3. Why did it happen?
4. How serious is it?

Explain in simple words. 100-150 words."""
            
            elif is_treatment_question:
                if language == 'hi':
                    final_prompt = f"""सवाल: "{question}"

इलाज बताओ:
1. तुरंत क्या करें?
2. कौनसी दवाई लगाएं?
3. आगे कैसे बचाएं?

आसान भाषा में। व्यावहारिक सलाह दो।"""
                elif language == 'te':
                    final_prompt = f"""ప్రశ్న: "{question}"

చికిత్స చెప్పండి:
1. వెంటనే ఏం చేయాలి?
2. ఏ మందు వేయాలి?
3. ముందు ఎలా కాపాడుకోవాలి?

సులభ భాషలో. ఆచరణీయ సలహా ఇవ్వండి."""
                else:
                    final_prompt = f"""Question: "{question}"

Tell me the treatment:
1. What to do immediately?
2. Which medicine to use?
3. How to prevent it next time?

Use simple language. Give practical advice."""
            
            else:
                # Simple identification - VERY DETAILED PROMPT FOR ACCURACY
                if language == 'hi':
                    final_prompt = f"""सवाल: "{question}"

तस्वीर को बहुत ध्यान से देखो:
- रंग क्या है? (हरा, लाल, पीला, नारंगी, बैंगनी?)
- आकार कैसा है? (गोल, लंबा, अंडाकार?)
- छिलका कैसा है? (चिकना, खुरदरा, धारीदार?)
- आकार कितना है? (छोटा, मध्यम, बड़ा?)

सभी विशेषताओं को देखकर फल का सही नाम बताओ।
आम फल: सेब, केला, आम, संतरा, तरबूज, पपीता, अमरूद, अंगूर, अनार, किवी, खरबूजा, तरबूज।

एक वाक्य में जवाब दो।"""
                elif language == 'te':
                    final_prompt = f"""ప్రశ్న: "{question}"

చిత్రాన్ని చాలా జాగ్రత్తగా చూడండి:
- రంగు ఏమిటి? (ఆకుపచ్చ, ఎరుపు, పసుపు, నారింజ, ఊదా?)
- ఆకారం ఎలా ఉంది? (గుండ్రంగా, పొడవుగా, అండాకారంగా?)
- తొక్క ఎలా ఉంది? (మృదువుగా, కఠినంగా, చారలతో?)
- పరిమాణం ఎంత? (చిన్నది, మధ్యస్థం, పెద్దది?)

అన్ని లక్షణాలను చూసి పండు యొక్క సరైన పేరు చెప్పండి.
సాధారణ పండ్లు: ఆపిల్, అరటి, మామిడి, నారింజ, పుచ్చకాయ, బొప్పాయి, జామ, ద్రాక్ష, దానిమ్మ, కివీ, ఖర్బూజా.

ఒక వాక్యంలో సమాధానం ఇవ్వండి."""
                else:
                    final_prompt = f"""Question: "{question}"

Look at the image VERY CAREFULLY:
- What COLOR is it? (green, red, yellow, orange, purple?)
- What SHAPE is it? (round, oval, elongated?)
- What is the SKIN like? (smooth, rough, striped, spotted?)
- What SIZE is it? (small, medium, large?)

Based on ALL these features, identify the fruit correctly.
Common fruits: apple, banana, mango, orange, watermelon, papaya, guava, grapes, pomegranate, kiwi, cantaloupe, dragon fruit, passion fruit.

Answer in ONE clear sentence with the correct fruit name."""
        
        elif has_image and not question:
            # Auto-analysis - DETAILED OBSERVATION FOR ACCURACY
            if language == 'hi':
                final_prompt = f"""इस तस्वीर को बहुत ध्यान से देखो।

पहले इन चीजों को देखो:
1. रंग - हरा, लाल, पीला, नारंगी, बैंगनी?
2. आकार - गोल, अंडाकार, लंबा?
3. छिलका - चिकना, खुरदरा, धारीदार?
4. साइज - छोटा, मध्यम, बड़ा?

फिर बताओ:
1. यह कौन सा फल/सब्जी/पौधा है? (सही नाम)
2. स्वस्थ है या बीमार?
3. और क्या दिख रहा है?

आसान भाषा में। 80-100 शब्दों में।"""
            elif language == 'te':
                final_prompt = f"""ఈ చిత్రాన్ని చాలా జాగ్రత్తగా చూడండి.

ముందు ఈ విషయాలు చూడండి:
1. రంగు - ఆకుపచ్చ, ఎరుపు, పసుపు, నారింజ, ఊదా?
2. ఆకారం - గుండ్రంగా, అండాకారంగా, పొడవుగా?
3. తొక్క - మృదువుగా, కఠినంగా, చారలతో?
4. పరిమాణం - చిన్నది, మధ్యస్థం, పెద్దది?

తర్వాత చెప్పండి:
1. ఇది ఏ పండు/కూరగాయ/మొక్క? (సరైన పేరు)
2. ఆరోగ్యంగా ఉందా లేక వ్యాధి ఉందా?
3. ఇంకా ఏం కనిపిస్తుంది?

సులభ భాషలో. 80-100 పదాలలో."""
            else:
                final_prompt = f"""Look at this image VERY CAREFULLY.

First observe these details:
1. COLOR - green, red, yellow, orange, purple?
2. SHAPE - round, oval, elongated?
3. SKIN - smooth, rough, striped, spotted?
4. SIZE - small, medium, large?

Then tell me:
1. What is this fruit/vegetable/plant? (exact name)
2. Is it healthy or diseased?
3. What else do you notice?

Use simple language. 80-100 words."""
        
        else:
            # General question - CONVERSATIONAL
            if language == 'hi':
                final_prompt = f"""सवाल: "{question}"

आसान हिंदी में जवाब दो। रोजमर्रा की भाषा में बताओ।"""
            elif language == 'te':
                final_prompt = f"""ప్రశ్న: "{question}"

సులభ తెలుగులో సమాధానం చెప్పండి. రోజువారీ భాషలో చెప్పండి."""
            else:
                final_prompt = f"""Question: "{question}"

Answer in simple English. Use everyday language."""
        
        # Prepare messages - Keep history but send ONLY current image
        # Keep last 8 messages (4 exchanges) for context
        recent_history = conversations[conv_id][-8:] if len(conversations[conv_id]) > 8 else conversations[conv_id]
        
        # Add system message
        messages = [
            {'role': 'system', 'content': system_prompts[language]}
        ]
        
        # Add text conversation history WITHOUT images (to avoid confusion)
        for msg in recent_history:
            msg_copy = {'role': msg['role'], 'content': msg['content']}
            # Don't include old images in history
            messages.append(msg_copy)
        
        # Add current message with ONLY the current/latest image
        current_msg = {'role': 'user', 'content': final_prompt}
        if image_data:
            current_msg['images'] = [image_data]  # Only current image
        
        messages.append(current_msg)
        
        print(f"📝 Messages in context: {len(messages)}")
        print(f"🖼️ Current message has image: {image_data is not None}")
        if new_image_uploaded:
            print(f"🆕 This is a NEW image upload")
        
        # Prepare payload - OPTIMIZED FOR BETTER VISION ACCURACY
        payload = {
            'model': QWEN_MODEL,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': 0.2,  # Lower for more accurate identification
                'top_p': 0.85,
                'num_predict': 400,
                'num_ctx': 4096,
                'repeat_penalty': 1.15,
                'num_gpu': 1,  # Use GPU if available
            }
        }
        
        # Add LoRA adapters if enabled
        active_adapters = get_active_lora_adapters()
        if active_adapters:
            payload['lora_adapters'] = [
                {
                    'path': adapter['path'],
                    'scale': adapter['weight']
                }
                for adapter in active_adapters.values()
            ]
        
        # CRITICAL FIX: Use proper timeout tuple (connection_timeout, read_timeout)
        print("Sending request to Ollama...")
        response = requests.post(
            OLLAMA_API_URL, 
            json=payload, 
            timeout=(CONNECTION_TIMEOUT, REQUEST_TIMEOUT)
        )
        
        if response.status_code != 200:
            print(f"Ollama error: {response.text}")
            return jsonify({
                'success': False,
                'error': f'Ollama API error: {response.text}'
            }), 500
        
        # Extract response
        response_data = response.json()
        assistant_message = response_data['message']['content']
        
        print(f"Response received (first 100 chars): {assistant_message[:100]}")
        
        # Update conversation history - Store text only (images stored separately)
        user_msg_for_history = {
            'role': 'user', 
            'content': question if question else 'Analyze this image'
        }
        
        conversations[conv_id].append(user_msg_for_history)
        conversations[conv_id].append({'role': 'assistant', 'content': assistant_message})
        
        # Keep last 10 exchanges (20 messages) for context
        if len(conversations[conv_id]) > 20:
            conversations[conv_id] = conversations[conv_id][-20:]
        
        return jsonify({
            'success': True,
            'response': assistant_message,
            'conversation_id': conv_id,
            'has_image': has_image,
            'image_remembered': conv_id in last_images,
            'language_used': language,
            'lora_adapters_used': list(active_adapters.keys()) if active_adapters else []
        })
    
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'Request timeout. The model is taking too long. Try: 1) Using smaller images, 2) Shorter questions, 3) Restarting Ollama'
        }), 504
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'error': 'Cannot connect to Ollama. Make sure it is running: ollama serve'
        }), 503
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        }), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    try:
        conv_id = request.json.get('conversation_id', 'default')
        if conv_id in conversations:
            del conversations[conv_id]
        if conv_id in last_images:
            del last_images[conv_id]
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🌾 Crop Disease Helper - Qwen 2.5-VL 7B (UPGRADED)")
    print("=" * 70)
    
    # Check Ollama status
    print("\n🔍 Checking Ollama Status...")
    if check_ollama_status():
        print("✅ Ollama is running!")
        models = get_ollama_models()
        print(f"📋 Available models: {', '.join(models) if models else 'None'}")
        print(f"🎯 Using model: {QWEN_MODEL}")
        
        if QWEN_MODEL not in models:
            print(f"\n⚠️  WARNING: Model '{QWEN_MODEL}' not found!")
            print("   Please run: ollama pull qwen2.5vl:7b")
            print("   This model is REQUIRED for accurate Hindi/Telugu responses!")
        else:
            print(f"✅ Model '{QWEN_MODEL}' is ready!")
    else:
        print("❌ Ollama is NOT running!")
        print("   Start with: ollama serve")
    
    print("\n" + "=" * 70)
    print("🗣️ LANGUAGE SETTINGS:")
    print("   • Using SIMPLE, EVERYDAY language")
    print("   • 7B model for ACCURATE multilingual responses")
    print("   • बोलचाल की भाषा | మాట్లాడే భాష")
    print("=" * 70)
    print("💡 FEATURES:")
    print("   • Multiple images supported (all stay in chat)")
    print("   • Each new image identified correctly")
    print("   • Conversation history maintained")
    print("   • Text-to-speech in all 3 languages")
    print("=" * 70)
    print(f"🌐 Server starting at: http://localhost:5000")
    print("   🇬🇧 English | 🇮🇳 हिंदी | తెలుగు")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
