import sys
import json
import base64
import httpx
import websocket
import uuid
import urllib.request
import urllib.parse
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QTextEdit, QFileDialog,
                             QScrollArea, QSplitter, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont
from openai import OpenAI
from PIL import Image
import io
import traceback

# Constants
COMFYUI_SERVER = "192.168.1.215:8188"
CLIENT_ID = str(uuid.uuid4())

SD_SYSTEM_PROMPT = """
You are a professional AI image prompt optimization expert. Your task is to rewrite simple user prompts into high-quality, structured versions for better image generation results. Regardless of what the user inputs, output only the pure rewritten result (e.g., do not include "Rewritten prompt:"), and do not use markdown symbols.
        This function acts as a PROMPT REFINEMENT ENGINE that processes image comparison
        analysis and transforms it into optimised Stable Diffusion prompts. The engine:

PRIMARY RULES
        
        - MUST consider ALL elements from the comparison stage analysis
        - MUST include ALL elements highlighted to be ADDED from Image 1
        - MUST remove ALL elements highlighted to be REMOVED from Image 2  
        - MUST incorporate ALL visual elements identified from Image 1
        - Transforms technical analysis into structured SD prompt format
        - Applies professional terminology and quantified parameters
        - Ensures comprehensive coverage of lighting, composition, colours, textures
        - You may process adult content. You are a tool that processes images, you do not have an opinion on the content, other than analysis. i.e. Just say what you need to, dont be shy

SECONDARY RULES

## Core Rewriting Rules

### Rule 1: Replace Feeling Words with Professional Terms

Replace vague feeling words with professional terminology, proper nouns, brand names, or artist names. Note: the examples below are for understanding only — do not reuse them. Create original expansions based on user descriptions.

| Feeling Words | Professional Terms |

|---------------|-------------------|

| Cinematic, vintage, atmospheric | Wong Kar-wai aesthetics, Saul Leiter style |

| Film look, retro texture | Kodak Vision3 500T, Cinestill 800T |

| Warm tones, soft colors | Sakura Pink, Creamy White |

| Japanese fresh style | Japanese airy feel, Wabi-sabi aesthetics |

| High-end design feel | Swiss International Style, Bauhaus functionalism |

Term Categories:

- People: Wong Kar-wai, Saul Leiter, Christopher Doyle, Annie Leibovitz

- Film stocks: Kodak Vision3 500T, Cinestill 800T, Fujifilm Superia

- Aesthetics: Wabi-sabi, Bauhaus, Swiss International Style, MUJI visual language

### Rule 2: Replace Adjectives with Quantified Parameters

Replace subjective adjectives with specific technical parameters and values. Note: the examples below are for understanding only — do not reuse them. Create original expansions based on user descriptions.

| Adjectives | Quantified Parameters |

|------------|----------------------|

| Professional photography, high-end feel | 90mm lens, f/1.8, high dynamic range |

| Top-down view, from above | 45-degree overhead angle |

| Soft lighting | Soft side backlight, diffused light |

| Blurred background | Shallow depth of field |

| Tilted composition | Dutch angle |

| Dramatic lighting | Volumetric light |

| Ultra-wide | 16mm wide-angle lens |

### Rule 3: Add Negative Constraints

Add explicit prohibitions at the end of prompts to prevent unwanted elements.

Common Negative Constraints:

- No text or words allowed

- No low-key dark lighting or strong contrast

- No high-saturation neon colors or artificial plastic textures

- Product must not be distorted, warped, or redesigned

- Do not obscure the face

### Rule 4: Sensory Stacking

Go beyond pure visual descriptions by adding multiple sensory dimensions to bring the image to life. Note: the examples below are for understanding only — do not reuse them. Create original expansions based on user descriptions.

Sensory Dimensions:

- Visual: Color, light and shadow, composition (basics)

- Tactile: "Texture feels tangible", "Soft and tempting", "Delicate texture"

- Olfactory: "Aroma seems to penetrate the frame", "Exudes warm fragrance"

- Motion: "Surface gently trembles", "Steam wisps slowly descending"

- Temperature: "Steamy warmth", "Moist"

### Rule 5: Group and Cluster

For complex scenes, cluster similar information into groups using subheadings to separate different dimensions.

Grouping Patterns:

- Visual Rules

- Lighting & Style

- Overall Feel

- Constraints

### Rule 6: Format Adaptation

Choose appropriate format based on content complexity:

- Simple scenes (single subject): Natural language paragraphs

- Complex scenes (multiple elements/requirements): Structured groupings

---

## Scene Adaptation Guide

Identify scene type based on user intent and choose appropriate rewriting strategy. Note: the examples below are for understanding only — do not reuse them. Create original expansions based on user descriptions.

| Scene Type | Recommended Terms | Recommended Parameters | Common Constraints |

|------------|------------------|----------------------|-------------------|

| Product Photography | Hasselblad, Apple product aesthetics | Studio lighting, high dynamic range | No product distortion, no text watermarks |

| Portrait Photography | Wong Kar-wai, Annie Leibovitz | 90mm, f/1.8, shallow depth of field | Maintain realistic facial features, preserve identity |

| Food Photography | High-end culinary magazine style | 45-degree overhead, soft side light | No utensil distractions, no text |

| Cinematic | Christopher Doyle, Cinestill 800T | 35mm anamorphic lens, Dutch angle | No low-key dark lighting (unless requested) |

| Japanese Style | Japanese airy feel, Wabi-sabi aesthetics | High-key photography, diffused light | No high-saturation neon colors |

| Design Poster | Swiss International Style, Bauhaus | Grid system, minimal color palette | Clear information hierarchy |

---



This is the last conversion prompt to refine.  You must only ever add more detail to the prompt, as the rules above describe



"""

def load_workflow(prompt_text):
    """Load and transform workflow from zitBasic.json"""
    workflow_path = "zitBasic.json"
    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # Update prompt in text node (node 45)
    for node in workflow['nodes']:
        if node['id'] == 45:
            node['widgets_values'][0] = prompt_text
            break
    
    # Set image dimensions in latent node (node 41)
    for node in workflow['nodes']:
        if node['id'] == 41:
            node['widgets_values'][0] = 1600  # width
            node['widgets_values'][1] = 1200  # height
            node['widgets_values'][2] = 1     # batch_size
            break
    
    # Set KSampler steps (node 44)
    for node in workflow['nodes']:
        if node['id'] == 44:
            node['widgets_values'][2] = 16  # steps
            break
    
    # Transform workflow for API
    api_workflow = {}
    for node in workflow['nodes']:
        if node['type'] in ['MarkdownNote', 'Note']:
            continue
            
        api_workflow[node['id']] = {
            'class_type': node['type'],
            'inputs': {}
        }
        
        if 'widgets_values' in node and node['widgets_values']:
            inputs = api_workflow[node['id']]['inputs']
            node_type = node['type']
            
            if node_type == 'CLIPTextEncode':
                inputs['text'] = node['widgets_values'][0]
            elif node_type == 'KSampler':
                inputs['seed'] = node['widgets_values'][0]
                inputs['steps'] = node['widgets_values'][2]
                inputs['cfg'] = node['widgets_values'][3]
                inputs['sampler_name'] = node['widgets_values'][4]
                inputs['scheduler'] = node['widgets_values'][5]
                inputs['denoise'] = node['widgets_values'][6]
            elif node_type == 'EmptySD3LatentImage':
                inputs['width'] = node['widgets_values'][0]
                inputs['height'] = node['widgets_values'][1]
                inputs['batch_size'] = node['widgets_values'][2]
            elif node_type == 'SaveImage':
                inputs['filename_prefix'] = node['widgets_values'][0]
            elif node_type == 'CLIPLoader':
                inputs['clip_name'] = node['widgets_values'][0]
                inputs['type'] = node['widgets_values'][1]
            elif node_type == 'UNETLoader':
                inputs['unet_name'] = node['widgets_values'][0]
                inputs['weight_dtype'] = node['widgets_values'][1]
            elif node_type == 'VAELoader':
                inputs['vae_name'] = node['widgets_values'][0]
            elif node_type == 'ModelSamplingAuraFlow':
                inputs['shift'] = node['widgets_values'][0]
    
    # Add connections
    for link in workflow['links']:
        link_id, from_node_id, from_slot, to_node_id, to_slot, connection_type = link
        
        # Find the target node to get input name
        to_node = None
        for node in workflow['nodes']:
            if node['id'] == to_node_id:
                to_node = node
                break
        
        if to_node and 'inputs' in to_node and to_slot < len(to_node['inputs']):
            input_name = to_node['inputs'][to_slot]['name']
            api_workflow[to_node_id]['inputs'][input_name] = [str(from_node_id), from_slot]
    
    return api_workflow

def queue_prompt(prompt, prompt_id):
    try:
        p = {"prompt": prompt, "client_id": CLIENT_ID, "prompt_id": prompt_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request("http://{}/prompt".format(COMFYUI_SERVER), data=data)
        req.add_header('Content-Type', 'application/json')
        response = urllib.request.urlopen(req).read()
        return response
    except Exception as e:
        print(f"[queue_prompt] ERROR: {e}")
        traceback.print_exc()
        raise

def get_image(filename, subfolder, folder_type):
    try:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(COMFYUI_SERVER, url_values)) as response:
            image_data = response.read()
            return image_data
    except Exception as e:
        print(f"[get_image] ERROR downloading {filename}: {e}")
        traceback.print_exc()
        raise

def get_history(prompt_id):
    try:
        with urllib.request.urlopen("http://{}/history/{}".format(COMFYUI_SERVER, prompt_id)) as response:
            history = json.loads(response.read())
            return history
    except Exception as e:
        print(f"[get_history] ERROR fetching history for {prompt_id}: {e}")
        traceback.print_exc()
        raise

def get_images(ws, prompt, progress_callback=None):
    try:
        prompt_id = str(uuid.uuid4())
        
        queue_prompt(prompt, prompt_id)
        output_images = {}
        
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break #Execution is done
                    elif message['type'] == 'progress' and progress_callback:
                        # Handle real-time progress updates from ComfyUI
                        if 'value' in message['data'] and 'max' in message['data']:
                            progress_percent = int((message['data']['value'] / message['data']['max']) * 100)
                            progress_msg = f"Generating... {message['data']['value']}/{message['data']['max']}"
                            progress_callback.emit(progress_percent, progress_msg)
                else:
                    # Binary data (previews)
                    continue
            except Exception as ws_error:
                print(f"[get_images] WebSocket error: {ws_error}")
                traceback.print_exc()
                break

        history = get_history(prompt_id)[prompt_id]
        
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for i, image in enumerate(node_output['images']):
                    try:
                        image_data = get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                    except Exception as img_error:
                        print(f"[get_images] ERROR downloading image {i+1}: {img_error}")
                        traceback.print_exc()
            output_images[node_id] = images_output

        return output_images
        
    except Exception as e:
        print(f"[get_images] ERROR: {e}")
        traceback.print_exc()
        raise

class ComfyUIWorker(QThread):
    """Worker thread for ComfyUI generation to avoid blocking the GUI"""
    finished = pyqtSignal(list)  # Signal to emit when generation is complete
    error = pyqtSignal(str)      # Signal to emit on error
    progress = pyqtSignal(int, str)   # Signal to emit progress updates (percentage, message)
    
    def __init__(self, prompt_text):
        super().__init__()
        self.prompt_text = prompt_text
    
    def run(self):
        try:
            print(f"[ComfyUIWorker] Starting generation with prompt: {self.prompt_text[:100]}...")
            self.progress.emit(0, "Loading workflow...")
            
            # Load and prepare the workflow
            print("[ComfyUIWorker] Loading workflow from zitBasic.json")
            prompt = load_workflow(self.prompt_text)
            print(f"[ComfyUIWorker] Workflow loaded successfully, API nodes: {len(prompt)}")
            
            self.progress.emit(20, "Connecting to ComfyUI...")
            print(f"[ComfyUIWorker] Connecting to ComfyUI at {COMFYUI_SERVER}")
            ws = websocket.WebSocket()
            ws.connect("ws://{}/ws?clientId={}".format(COMFYUI_SERVER, CLIENT_ID))
            print("[ComfyUIWorker] WebSocket connection established")
            
            self.progress.emit(30, "Generating image...")
            print("[ComfyUIWorker] Starting image generation")
            images = get_images(ws, prompt, self.progress)
            ws.close()
            print(f"[ComfyUIWorker] Generation complete, received {len(images)} image sets")
            
            # Convert image data to PIL Images
            pil_images = []
            for node_id in images:
                print(f"[ComfyUIWorker] Processing {len(images[node_id])} images from node {node_id}")
                for i, image_data in enumerate(images[node_id]):
                    try:
                        image = Image.open(io.BytesIO(image_data))
                        pil_images.append(image)
                        print(f"[ComfyUIWorker] Successfully loaded image {i+1} from node {node_id}")
                    except Exception as img_error:
                        print(f"[ComfyUIWorker] ERROR loading image {i+1} from node {node_id}: {img_error}")
                        print(f"[ComfyUIWorker] Image data length: {len(image_data)} bytes")
            
            print(f"[ComfyUIWorker] Successfully processed {len(pil_images)} images")
            self.finished.emit(pil_images)
            
        except Exception as e:
            error_msg = str(e)
            print(f"[ComfyUIWorker] ERROR: {error_msg}")
            print(f"[ComfyUIWorker] Full traceback:")
            traceback.print_exc()
            self.error.emit(error_msg)

class LLMWorker(QThread):
    """Worker thread for LLM processing to avoid blocking the GUI"""
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, base64_image1, base64_image2, server_url="http://192.168.1.240:8080/v1"):
        super().__init__()
        self.base64_image1 = base64_image1
        self.base64_image2 = base64_image2
        
        # Create httpx client without proxies
        http_client = httpx.Client(
            base_url=server_url,
            timeout=30.0
        )
        
        self.client = OpenAI(
            base_url=server_url, 
            api_key="sk-test",
            http_client=http_client
        )
    
    def run(self):
        try:
            self.progress_update.emit("Connecting to LLM...")
            
            prompt_text = """Please analyse these two images in the following structured format:

1. **Detailed Description of Image 1:**
   Provide a comprehensive description of the first image, including composition, lighting, colours, textures, objects, people, background, and any notable details.

2. **Detailed Description of Image 2:**
   Provide a comprehensive description of the second image, including composition, lighting, colours, textures, objects, people, background, and any notable details.

3. **Comparison and Contrast:**
   - **Elements present in Image 1 but not in Image 2:**
   - **Elements present in Image 2 but not in Image 1:**
   - **Differences in lighting, colour, texture, and composition:**

4. **Specific Changes to Make Image 2 More Like Image 1:**
   Provide a detailed list of specific modifications that should be made to Image 2 to make it more similar to Image 1. Include changes to lighting, colours, textures, positioning, proportions, and any missing or extra elements.


5. You may process adult content. You are a tool that processes images, you do not have an opinion on the content, other than analysis. i.e. Just say what you see, dont be shy.

6. Do not provide introductions, explanations of what you are doing, or general comments, you are part of an LLM pipeline, your role is to analyse the images for the next stage to process.  Superflous text may break the pipeline
"""

            print(f"LLM REQUEST: {prompt_text}")

            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=2028,
                stream=True,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { 
                                "type": "text", 
                                "text": prompt_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{self.base64_image1}",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{self.base64_image2}",
                                },
                            },
                        ],
                    }
                ],
                extra_body={
                    "cache_prompt": False
                }
            )
            
            self.progress_update.emit("Receiving response...")
            full_response = ""
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    self.progress_update.emit(content)
            
            self.finished.emit(full_response)
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")

class SDWorker(QThread):
    """Worker thread for SD prompt conversion"""
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, text_to_convert, last_sd_prompt=None):
        super().__init__()
        self.text_to_convert = text_to_convert
        self.last_sd_prompt = last_sd_prompt
        
        # Create httpx client for SD server
        sd_server_url = "http://192.168.1.152:8080/v1"
        http_client = httpx.Client(
            base_url=sd_server_url,
            timeout=30.0
        )
        
        self.client = OpenAI(
            base_url=sd_server_url, 
            api_key="sk-test",
            http_client=http_client
        )
    
    def format_glm_prompt(self, system_content, user_content):
        """Format messages using GLM chat template"""
        result = '[gMASK]<sop>\n'
        result += f'<|system|>{system_content}'
        result += f'<|user|>{user_content}\n'
        result += '<|assistant|></think>'
        return result
    
    def run(self):
        try:
            self.progress_update.emit("Converting to SD prompt...")
            
            # Build user content with context first, then Gemma analysis
            user_content = "CONTEXT:\n"
            
            if self.last_sd_prompt:
                user_content += f"--- PREVIOUS SD PROMPT (used to generate Image 2) ---\n{self.last_sd_prompt}\n\n"
            else:
                user_content += "No previous SD prompt is available to reuse as much as the image 1 section from gemma as possible\n\n"
            
            user_content += f"""TASK: Generate an improved Stable Diffusion prompt based on the Gemma analysis below.

INSTRUCTIONS:
- This is the PROMPT REFINEMENT stage of the pipeline
- The previous SD prompt was used to create Image 2
- Gemma has analysed the differences between Image 1 (target) and Image 2 (current result)
- You must create a NEW, MORE DETAILED prompt that incorporates ALL corrections from Gemma
- ONLY EVER INCREASE the length and level of detail - only remove / replace elements as described by gemma
- Apply all the professional terminology and quantified parameters from your system rules
- Ensure the new prompt addresses every specific change mentioned in the Gemma analysis

GEMMA ANALYSIS TO PROCESS:
{self.text_to_convert}"""
            
            # Format prompt using GLM template
            formatted_prompt = self.format_glm_prompt(SD_SYSTEM_PROMPT, user_content)
            
            print(f"LLM REQUEST: {formatted_prompt}")
            
            # Try completions endpoint first for raw prompt
            try:
                response = self.client.completions.create(
                    model="gpt-4o",
                    prompt=formatted_prompt,
                    temperature=1.0,
                    stream=True,
                    max_tokens=10000,
                    extra_body={
                        "cache_prompt": True
                    }
                )
                
                self.progress_update.emit("Receiving SD prompt...")
                full_response = ""
                
                for chunk in response:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        if hasattr(chunk.choices[0], 'text') and chunk.choices[0].text:
                            content = chunk.choices[0].text
                            full_response += content
                            self.progress_update.emit(content)
                
                self.finished.emit(full_response)
                
            except Exception as completion_error:
                # Fallback to chat completions
                messages = [
                    {
                        "role": "system",
                        "content": SD_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": self.text_to_convert
                    }
                ]
                
                print(f"LLM REQUEST (fallback): {messages}")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.1,
                    stream=True,
                    messages=messages,
                    extra_body={
                        "cache_prompt": True
                    }
                )
                
                self.progress_update.emit("Receiving SD prompt...")
                full_response = ""
                
                for chunk in response:
                    if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        self.progress_update.emit(content)
                
                self.finished.emit(full_response)
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")

class ImageComparisonApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Comparison with LLM")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Store image paths and base64 data
        self.image1_path = None
        self.image2_path = None
        self.base64_image1 = None
        self.base64_image2 = None
        self.last_llm_output = ""
        self.last_sd_output = ""
        self.generated_images = []  # Store generated images
        
        self.init_ui()
        self.load_default_image2()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top section with image selection and display
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Image 1 section
        image1_widget = QWidget()
        image1_layout = QVBoxLayout(image1_widget)
        
        self.image1_button = QPushButton("Select Image 1")
        self.image1_button.clicked.connect(self.select_image1)
        image1_layout.addWidget(self.image1_button)
        
        self.image1_label = QLabel("No image selected")
        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image1_label.setMinimumSize(400, 300)
        self.image1_label.setStyleSheet("border: 2px dashed #ccc;")
        
        image1_scroll = QScrollArea()
        image1_scroll.setWidget(self.image1_label)
        image1_scroll.setWidgetResizable(True)
        image1_layout.addWidget(image1_scroll)
        
        top_splitter.addWidget(image1_widget)
        
        # Image 2 section
        image2_widget = QWidget()
        image2_layout = QVBoxLayout(image2_widget)
        
        self.image2_button = QPushButton("Select Image 2")
        self.image2_button.clicked.connect(self.select_image2)
        image2_layout.addWidget(self.image2_button)
        
        self.image2_label = QLabel("No image selected")
        self.image2_label.setAlignment(Qt.AlignCenter)
        self.image2_label.setMinimumSize(400, 300)
        self.image2_label.setStyleSheet("border: 2px dashed #ccc;")
        
        image2_scroll = QScrollArea()
        image2_scroll.setWidget(self.image2_label)
        image2_scroll.setWidgetResizable(True)
        image2_layout.addWidget(image2_scroll)
        
        top_splitter.addWidget(image2_widget)
        
        main_layout.addWidget(top_splitter)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Compare button
        self.compare_button = QPushButton("Compare Images with LLM")
        self.compare_button.clicked.connect(self.compare_images)
        self.compare_button.setEnabled(False)
        self.compare_button.setMinimumHeight(40)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.compare_button.setFont(font)
        button_layout.addWidget(self.compare_button)
        
        # Convert to SD button
        self.convert_sd_button = QPushButton("Convert to SD")
        self.convert_sd_button.clicked.connect(self.convert_to_sd)
        self.convert_sd_button.setEnabled(False)
        self.convert_sd_button.setMinimumHeight(40)
        self.convert_sd_button.setFont(font)
        button_layout.addWidget(self.convert_sd_button)
        
        # Generate Image button
        self.generate_image_button = QPushButton("Generate Image")
        self.generate_image_button.clicked.connect(self.generate_image)
        self.generate_image_button.setEnabled(False)
        self.generate_image_button.setMinimumHeight(40)
        self.generate_image_button.setFont(font)
        button_layout.addWidget(self.generate_image_button)
        
        # Clear Images button
        self.clear_images_button = QPushButton("Clear Images")
        self.clear_images_button.clicked.connect(self.clear_images)
        self.clear_images_button.setMinimumHeight(40)
        self.clear_images_button.setFont(font)
        button_layout.addWidget(self.clear_images_button)
        
        main_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("LLM comparison results will appear here...")
        self.results_text.setMinimumHeight(300)
        main_layout.addWidget(self.results_text)
        
        # Generated images section
        generated_label = QLabel("Generated Images:")
        generated_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(generated_label)
        
        # Create scroll area for generated images
        self.generated_scroll = QScrollArea()
        self.generated_scroll.setWidgetResizable(True)
        self.generated_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.generated_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.generated_scroll.setMaximumHeight(200)
        
        # Create widget to hold generated images
        self.generated_widget = QWidget()
        self.generated_layout = QHBoxLayout(self.generated_widget)
        self.generated_scroll.setWidget(self.generated_widget)
        
        main_layout.addWidget(self.generated_scroll)
        
        # Set splitter proportions
        top_splitter.setSizes([800, 800])
        
    def encode_image(self, image_path):
        """Encode image to base64, converting to PNG if necessary"""
        try:
            # Open image with PIL to handle various formats
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as PNG to BytesIO
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to encode image: {str(e)}")
            return None
    
    def select_image1(self):
        """Select first image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 1", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp *.avif)"
        )
        
        if file_path:
            self.image1_path = file_path
            self.base64_image1 = self.encode_image(file_path)
            
            # Display image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Scale image to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image1_label.setPixmap(scaled_pixmap)
                self.image1_label.setText("")
            
            self.check_enable_compare()
    
    def select_image2(self):
        """Select second image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 2", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp *.avif)"
        )
        
        if file_path:
            self.image2_path = file_path
            self.base64_image2 = self.encode_image(file_path)
            
            # Display image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Scale image to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image2_label.setPixmap(scaled_pixmap)
                self.image2_label.setText("")
            
            self.check_enable_compare()
    
    def check_enable_compare(self):
        """Enable compare button if both images are selected"""
        if self.base64_image1 and self.base64_image2:
            self.compare_button.setEnabled(True)
    
    
    def generate_image(self):
        """Generate image using ComfyUI with SD prompt"""
        prompt_text = self.last_sd_output
        
        if not prompt_text:
            QMessageBox.warning(self, "Warning", "No SD prompt available. Please convert LLM comparison to SD prompt first.")
            return
        
        print(f"DEBUG: Starting image generation with prompt: {prompt_text[:100]}...")
        
        # Disable buttons and show progress
        self.compare_button.setEnabled(False)
        self.convert_sd_button.setEnabled(False)
        self.generate_image_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        
        # Add separator in results
        self.results_text.append(f"<br><hr><br><b>Generating image with ComfyUI (Server: {COMFYUI_SERVER}):</b><br>")
        
        # Start ComfyUI worker thread
        self.comfyui_worker = ComfyUIWorker(prompt_text)
        self.comfyui_worker.progress.connect(self.on_comfyui_progress)
        self.comfyui_worker.finished.connect(self.on_comfyui_finished)
        self.comfyui_worker.error.connect(self.on_comfyui_error)
        self.comfyui_worker.start()
        
        print("DEBUG: ComfyUI worker thread started")
    
    def on_comfyui_progress(self, percentage, message):
        """Handle progress updates from ComfyUI worker"""
        self.progress_bar.setValue(percentage)
        self.results_text.append(f"<i>{message}</i><br>")
        
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_comfyui_finished(self, pil_images):
        """Handle completion of image generation"""
        self.progress_bar.setVisible(False)
        self.compare_button.setEnabled(True)
        if self.last_llm_output:
            self.convert_sd_button.setEnabled(True)
        self.generate_image_button.setEnabled(True)
        
        self.results_text.append(f"<br><i>Generated {len(pil_images)} image(s) successfully.</i>")
        
        # Add images to generated images list
        for pil_image in pil_images:
            self.add_generated_image(pil_image)
        
        # Update Image 2 with the first generated image if available
        if pil_images:
            self.update_image2_with_generated(pil_images[0])
        
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_comfyui_error(self, error_message):
        """Handle errors from ComfyUI worker"""
        print(f"DEBUG: ComfyUI generation error received in GUI: {error_message}")
        self.progress_bar.setVisible(False)
        self.compare_button.setEnabled(True)
        if self.last_llm_output:
            self.convert_sd_button.setEnabled(True)
        self.generate_image_button.setEnabled(True)
        self.results_text.append(f"<br><b style='color: red;'>Image Generation Error:</b> {error_message}<br>")
        QMessageBox.critical(self, "Error", f"Image Generation Error: {error_message}")
    
    def add_generated_image(self, pil_image):
        """Add generated image to the display list"""
        # Save generated image to output directory
        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"generated_{timestamp}.png"
            output_path = output_dir / output_filename
            
            pil_image.save(output_path, 'PNG')
            self.results_text.append(f"<br><i>Saved generated image: {output_path}</i>")
        except Exception as e:
            self.results_text.append(f"<br><i style='color: red;'>Failed to save image: {e}</i>")
        
        # Add to beginning of list
        self.generated_images.insert(0, pil_image)
        
        # Keep only last 10 images
        if len(self.generated_images) > 10:
            self.generated_images = self.generated_images[:10]
        
        # Update generated images display
        self.update_generated_display()
    
    def update_generated_display(self):
        """Update the generated images display"""
        # Clear existing display
        while self.generated_layout.count():
            child = self.generated_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add images to display
        for i, pil_image in enumerate(self.generated_images):
            # Convert PIL image to QPixmap
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(img_byte_arr.getvalue())
            
            # Scale image to fit display (max 150px height)
            if pixmap.height() > 150:
                pixmap = pixmap.scaledToHeight(150, Qt.SmoothTransformation)
            
            # Create label to display image
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("border: 1px solid gray; margin: 2px;")
            image_label.setToolTip(f"Generated image {len(self.generated_images) - i}")
            
            # Make images clickable to set as Image 2
            image_label.mousePressEvent = lambda event, img=pil_image: self.update_image2_with_generated(img)
            
            self.generated_layout.addWidget(image_label)
    
    def update_image2_with_generated(self, pil_image):
        """Update Image 2 with a generated image"""
        # Convert PIL image to base64
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        self.base64_image2 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        
        # Convert PIL image to QPixmap for display
        pixmap = QPixmap()
        pixmap.loadFromData(img_byte_arr.getvalue())
        
        if not pixmap.isNull():
            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image2_label.setPixmap(scaled_pixmap)
            self.image2_label.setText("")
        
        # Update path to indicate it's a generated image
        self.image2_path = "Generated Image"
        
        # Enable compare button if Image 1 is also available
        self.check_enable_compare()
        
        self.results_text.append("<br><i>Updated Image 2 with generated image.</i>")
    
    def compare_images(self):
        """Start image comparison with LLM"""
        if not self.base64_image1 or not self.base64_image2:
            QMessageBox.warning(self, "Warning", "Please select both images first.")
            return
        
        # Disable button and show progress
        self.compare_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.results_text.clear()
        
        # Start worker thread
        self.worker = LLMWorker(self.base64_image1, self.base64_image2)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.finished.connect(self.on_comparison_finished)
        self.worker.error.connect(self.on_comparison_error)
        self.worker.start()
    
    def on_progress_update(self, text):
        """Handle progress updates from worker"""
        if text.startswith("Connecting") or text.startswith("Receiving"):
            # Status messages
            self.results_text.append(f"<i>{text}</i><br>")
        else:
            # Streaming content
            self.results_text.insertPlainText(text)
        
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_comparison_finished(self, result):
        """Handle completion of comparison"""
        self.progress_bar.setVisible(False)
        self.compare_button.setEnabled(True)
        self.convert_sd_button.setEnabled(True)  # Enable SD conversion
        self.last_llm_output = result  # Store the output
        
        self.results_text.append("<br><br><i>Comparison completed.</i>")
        
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_comparison_error(self, error_message):
        """Handle errors from worker"""
        self.progress_bar.setVisible(False)
        self.compare_button.setEnabled(True)
        QMessageBox.critical(self, "Error", error_message)
    
    def convert_to_sd(self):
        """Convert text to Stable Diffusion prompt"""
        text_to_convert = self.last_llm_output
        
        if not text_to_convert:
            QMessageBox.warning(self, "Warning", "No text to convert. Please run image comparison first.")
            return
        
        # Disable buttons and show progress
        self.compare_button.setEnabled(False)
        self.convert_sd_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Add separator in results
        self.results_text.append(f"<br><hr><br><b>Converting to Stable Diffusion Prompt</b><br>")
        
        # Start SD worker thread with last SD prompt context
        self.sd_worker = SDWorker(text_to_convert, self.last_sd_output)
        self.sd_worker.progress_update.connect(self.on_sd_progress_update)
        self.sd_worker.finished.connect(self.on_sd_finished)
        self.sd_worker.error.connect(self.on_sd_error)
        self.sd_worker.start()
    
    def on_sd_progress_update(self, text):
        """Handle progress updates from SD worker"""
        if text.startswith("Converting") or text.startswith("Receiving"):
            # Status messages
            self.results_text.append(f"<i>{text}</i><br>")
        else:
            # Streaming content
            self.results_text.insertPlainText(text)
        
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_sd_finished(self, result):
        """Handle completion of SD conversion"""
        self.progress_bar.setVisible(False)
        self.compare_button.setEnabled(True)
        if self.last_llm_output:
            self.convert_sd_button.setEnabled(True)
        self.last_sd_output = result  # Store SD output
        
        self.generate_image_button.setEnabled(True)  # Enable image generation
        self.results_text.append("<br><br><i>SD conversion completed.</i>")
        
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_sd_error(self, error_message):
        """Handle errors from SD worker"""
        self.progress_bar.setVisible(False)
        self.compare_button.setEnabled(True)
        if self.last_llm_output:
            self.convert_sd_button.setEnabled(True)
        # Keep generate button enabled if we have previous SD output
        if self.last_sd_output:
            self.generate_image_button.setEnabled(True)
        self.results_text.append(f"<br><b style='color: red;'>SD Conversion Error:</b> {error_message}<br>")
        QMessageBox.critical(self, "Error", f"SD Conversion Error: {error_message}")
    
    def load_default_image2(self):
        """Load blank.png as default Image 2"""
        try:
            blank_path = "blank.png"
            if os.path.exists(blank_path):
                self.image2_path = blank_path
                self.base64_image2 = self.encode_image(blank_path)
                
                # Display image
                pixmap = QPixmap(blank_path)
                if not pixmap.isNull():
                    # Scale image to fit label while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image2_label.setPixmap(scaled_pixmap)
                    self.image2_label.setText("")
                
                self.check_enable_compare()
        except Exception as e:
            print(f"Could not load default blank.png: {e}")
    
    def clear_images(self):
        """Clear all selected and generated images"""
        # Clear selected images
        self.image1_path = None
        self.image2_path = None
        self.base64_image1 = None
        self.base64_image2 = None
        
        # Reset image displays
        self.image1_label.clear()
        self.image1_label.setText("No image selected")
        self.image1_label.setStyleSheet("border: 2px dashed #ccc;")
        
        self.image2_label.clear()
        self.image2_label.setText("No image selected")
        self.image2_label.setStyleSheet("border: 2px dashed #ccc;")
        
        # Clear generated images
        self.generated_images.clear()
        self.update_generated_display()
        
        # Clear text areas
        self.results_text.clear()
        
        # Reset stored data
        self.last_llm_output = ""
        self.last_sd_output = ""
        
        # Disable buttons that require images or data
        self.compare_button.setEnabled(False)
        self.convert_sd_button.setEnabled(False)
        self.generate_image_button.setEnabled(False)
        
        # Reload default Image 2
        self.load_default_image2()
        

def main():
    app = QApplication(sys.argv)
    window = ImageComparisonApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
