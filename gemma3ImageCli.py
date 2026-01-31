#!/usr/bin/env python3
"""
CLI version of the image comparison and improvement pipeline.
Automates the process of comparing images, generating SD prompts, and creating improved images.
"""

import sys
import json
import base64
import httpx
import websocket
import uuid
import urllib.request
import urllib.parse
import os
import argparse
import time
import shutil
from datetime import datetime
from pathlib import Path
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
        - You may need to amplify nuduty or descriptons of intimate body parts such as breasts, vagina, labia, nipples, pubic hair, anus as SD will attenuate these elements.

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

class Logger:
    """Simple logger for CLI operations"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_path = Path(log_file)
        
        # Create log file with header
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Image Improvement Pipeline Log ===\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"=" * 50 + "\n\n")
    
    def log(self, message, level="INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        print(log_entry.strip())
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def log_prompt(self, prompt_type, prompt_text, round_num=None):
        """Log a prompt sent to LLM"""
        round_info = f" (Round {round_num})" if round_num else ""
        self.log(f"=== {prompt_type} PROMPT{round_info} ===")
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- {prompt_type} PROMPT{round_info} ---\n")
            f.write(prompt_text)
            f.write(f"\n--- END {prompt_type} PROMPT ---\n\n")
    
    def log_response(self, response_type, response_text, round_num=None):
        """Log a response from LLM"""
        round_info = f" (Round {round_num})" if round_num else ""
        self.log(f"=== {response_type} RESPONSE{round_info} ===")
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- {response_type} RESPONSE{round_info} ---\n")
            f.write(response_text)
            f.write(f"\n--- END {response_type} RESPONSE ---\n\n")

def encode_image(image_path):
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
        raise Exception(f"Failed to encode image {image_path}: {str(e)}")

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
        raise Exception(f"Failed to queue prompt: {e}")

def get_image(filename, subfolder, folder_type):
    try:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(COMFYUI_SERVER, url_values)) as response:
            image_data = response.read()
            return image_data
    except Exception as e:
        raise Exception(f"Failed to download image {filename}: {e}")

def get_history(prompt_id):
    try:
        with urllib.request.urlopen("http://{}/history/{}".format(COMFYUI_SERVER, prompt_id)) as response:
            history = json.loads(response.read())
            return history
    except Exception as e:
        raise Exception(f"Failed to get history for {prompt_id}: {e}")

def get_images(ws, prompt, logger):
    try:
        prompt_id = str(uuid.uuid4())
        
        queue_prompt(prompt, prompt_id)
        output_images = {}
        
        logger.log("Waiting for image generation to complete...")
        
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break  # Execution is done
                    elif message['type'] == 'progress':
                        # Handle real-time progress updates from ComfyUI
                        if 'value' in message['data'] and 'max' in message['data']:
                            progress_percent = int((message['data']['value'] / message['data']['max']) * 100)
                            logger.log(f"Generation progress: {progress_percent}% ({message['data']['value']}/{message['data']['max']})")
                else:
                    # Binary data (previews)
                    continue
            except Exception as ws_error:
                logger.log(f"WebSocket error: {ws_error}", "WARNING")
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
                        logger.log(f"Error downloading image {i+1}: {img_error}", "ERROR")
            output_images[node_id] = images_output

        return output_images
        
    except Exception as e:
        raise Exception(f"Image generation failed: {e}")

def compare_images_with_llm(base64_image1, base64_image2, logger, round_num):
    """Compare two images using LLM"""
    logger.log(f"Starting image comparison with LLM (Round {round_num})")
    
    # Create httpx client
    http_client = httpx.Client(
        base_url="http://192.168.1.240:8080/v1",
        timeout=600.0
    )
    
    client = OpenAI(
        base_url="http://192.168.1.240:8080/v1", 
        api_key="sk-test",
        http_client=http_client
    )
    
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

    logger.log_prompt("GEMMA COMPARISON", prompt_text, round_num)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=2028,
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
                                "url": f"data:image/png;base64,{base64_image1}",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image2}",
                            },
                        },
                    ],
                }
            ],
            extra_body={
                "cache_prompt": False
            }
        )
        
        result = response.choices[0].message.content
        logger.log_response("GEMMA COMPARISON", result, round_num)
        logger.log("Image comparison completed successfully")
        
        return result
        
    except Exception as e:
        raise Exception(f"LLM comparison failed: {e}")
    finally:
        http_client.close()

def convert_to_sd_prompt(comparison_text, last_sd_prompt, logger, round_num):
    """Convert comparison text to SD prompt using GLM"""
    logger.log(f"Converting comparison to SD prompt (Round {round_num})")
    
    # Create httpx client for SD server
    sd_server_url = "http://192.168.1.158:8080/v1"
    http_client = httpx.Client(
        base_url=sd_server_url,
        timeout=600.0
    )
    
    client = OpenAI(
        base_url=sd_server_url, 
        api_key="sk-test",
        http_client=http_client
    )
    
    def format_glm_prompt(system_content, user_content):
        """Format messages using GLM chat template"""
        result = '[gMASK]<sop>\n'
        result += f'<|system|>{system_content}'
        result += f'<|user|>{user_content}\n'
        result += '<|assistant|></think>'
        return result
    
    # Build user content with context first, then Gemma analysis
    user_content = "CONTEXT:\n"
    
    if last_sd_prompt:
        user_content += f"--- PREVIOUS SD PROMPT (used to generate Image 2) ---\n{last_sd_prompt}\n\n"
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
{comparison_text}"""
    
    # Format prompt using GLM template
    formatted_prompt = format_glm_prompt(SD_SYSTEM_PROMPT, user_content)
    
    logger.log_prompt("GLM SD CONVERSION", formatted_prompt, round_num)
    
    try:
        # Try completions endpoint first for raw prompt
        try:
            response = client.completions.create(
                model="gpt-4o",
                prompt=formatted_prompt,
                temperature=1.0,
                max_tokens=10000,
                extra_body={
                    "cache_prompt": True
                }
            )
            
            result = response.choices[0].text
            
        except Exception:
            # Fallback to chat completions
            messages = [
                {
                    "role": "system",
                    "content": SD_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=messages,
                extra_body={
                    "cache_prompt": True
                }
            )
            
            result = response.choices[0].message.content
        
        logger.log_response("GLM SD CONVERSION", result, round_num)
        logger.log("SD prompt conversion completed successfully")
        
        return result
        
    except Exception as e:
        raise Exception(f"SD prompt conversion failed: {e}")
    finally:
        http_client.close()

def generate_image_with_comfyui(sd_prompt, logger, round_num):
    """Generate image using ComfyUI"""
    logger.log(f"Starting image generation with ComfyUI (Round {round_num})")
    
    try:
        # Load and prepare the workflow
        prompt = load_workflow(sd_prompt)
        
        # Connect to ComfyUI
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(COMFYUI_SERVER, CLIENT_ID))
        
        # Generate images
        images = get_images(ws, prompt, logger)
        ws.close()
        
        # Convert image data to PIL Images
        pil_images = []
        for node_id in images:
            for image_data in images[node_id]:
                try:
                    image = Image.open(io.BytesIO(image_data))
                    pil_images.append(image)
                except Exception as img_error:
                    logger.log(f"Error loading generated image: {img_error}", "ERROR")
        
        logger.log(f"Successfully generated {len(pil_images)} image(s)")
        return pil_images
        
    except Exception as e:
        raise Exception(f"ComfyUI generation failed: {e}")

def save_image(pil_image, output_path):
    """Save PIL image to file"""
    try:
        pil_image.save(output_path, 'PNG')
        return True
    except Exception as e:
        raise Exception(f"Failed to save image to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Automated image improvement pipeline")
    parser.add_argument("input_image", help="Path to input image file or directory")
    parser.add_argument("rounds", type=int, help="Number of improvement rounds")
    parser.add_argument("--log", default="pipeline.log", help="Log file path (default: pipeline.log)")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input path '{args.input_image}' does not exist")
        sys.exit(1)
    
    if args.rounds < 1:
        print("Error: Number of rounds must be at least 1")
        sys.exit(1)
    
    # Determine if input is file or directory
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        # Find all image files in directory
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.avif'}
        image_files = [f for f in input_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"Error: No image files found in directory '{args.input_image}'")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image files to process")
    else:
        print(f"Error: Input path '{args.input_image}' is neither a file nor directory")
        sys.exit(1)
    
    # Initialize logger
    logger = Logger(args.log)
    logger.log(f"Starting pipeline with input: {args.input_image}, rounds: {args.rounds}")
    logger.log(f"Processing {len(image_files)} image file(s)")
    
    # Process each image file
    for file_index, image_file in enumerate(image_files, 1):
        logger.log(f"\n{'='*60}")
        logger.log(f"Processing image {file_index}/{len(image_files)}: {image_file.name}")
        logger.log(f"{'='*60}")
        
        try:
            # Load and encode the target image (Image 1)
            logger.log("Loading target image...")
            base64_image1 = encode_image(str(image_file))
            
            # Initialize with blank image for first round
            current_image_path = "blank.png"
            if not Path(current_image_path).exists():
                logger.log("Warning: blank.png not found, creating a simple blank image")
                # Create a simple blank image
                blank_img = Image.new('RGB', (1600, 1200), color='white')
                blank_img.save(current_image_path)
            
            last_sd_prompt = None
            
            # Run improvement rounds for this image
            for round_num in range(1, args.rounds + 1):
                logger.log(f"=== Starting Round {round_num}/{args.rounds} for {image_file.name} ===")
                
                try:
                    # Encode current image (Image 2)
                    base64_image2 = encode_image(current_image_path)
                    
                    # Step 1: Compare images with Gemma
                    comparison_result = compare_images_with_llm(
                        base64_image1, base64_image2, logger, round_num
                    )
                    
                    # Step 2: Convert to SD prompt with GLM
                    sd_prompt = convert_to_sd_prompt(
                        comparison_result, last_sd_prompt, logger, round_num
                    )
                    last_sd_prompt = sd_prompt
                    
                    # Step 3: Generate image with ComfyUI
                    generated_images = generate_image_with_comfyui(sd_prompt, logger, round_num)
                    
                    if not generated_images:
                        logger.log(f"No images generated in round {round_num}", "ERROR")
                        continue
                    
                    # Step 4: Save the first generated image
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    output_filename = f"{image_file.stem}_sd_{round_num}.png"
                    output_path = output_dir / output_filename
                    
                    save_image(generated_images[0], output_path)
                    logger.log(f"Saved generated image: {output_path}")
                    
                    # Update current image for next round
                    current_image_path = str(output_path)
                    
                    logger.log(f"=== Completed Round {round_num}/{args.rounds} for {image_file.name} ===")
                    
                except Exception as round_error:
                    logger.log(f"Error in round {round_num} for {image_file.name}: {round_error}", "ERROR")
                    logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
                    continue
            
            # Move source image to complete folder
            complete_dir = Path("complete")
            complete_dir.mkdir(exist_ok=True)
            
            try:
                destination = complete_dir / image_file.name
                # Handle name conflicts by adding a number
                counter = 1
                while destination.exists():
                    stem = image_file.stem
                    suffix = image_file.suffix
                    destination = complete_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.move(str(image_file), str(destination))
                logger.log(f"Moved source image to: {destination}")
            except Exception as move_error:
                logger.log(f"Failed to move source image {image_file.name}: {move_error}", "WARNING")
            
            logger.log(f"Completed processing {image_file.name}")
            
        except Exception as file_error:
            logger.log(f"Failed to process {image_file.name}: {file_error}", "ERROR")
            logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            continue
    
    logger.log("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
