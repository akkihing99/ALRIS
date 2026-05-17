import sys
import os
import torch
import torch.nn.functional as F


DEVICE_ID = 0 
try:
    torch.cuda.set_device(DEVICE_ID)
    physical_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    print(f"[STARTUP] Process sees logical device cuda:{DEVICE_ID} (mapped from physical cuda:{physical_gpu})")
except Exception as e:
    print(f"[FATAL] Could not set device to cuda:{DEVICE_ID}. Error: {e}")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'vip_llava')))

from flask import Flask, render_template, send_from_directory, jsonify, request, session, redirect, url_for
from vip_llava.llava.model.builder import load_pretrained_model
from vip_llava.llava.mm_utils import tokenizer_image_token
from vip_llava.llava.conversation import conv_templates
from vip_llava.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from PIL import Image, ImageDraw
import cv2
import numpy as np
import base64
from io import BytesIO
import json
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MASK_FOLDER = os.path.join(BASE_DIR, 'masks')
ANNOTATED_FOLDER = os.path.join(BASE_DIR, 'annotated_masks_ours')
CAPTION_FOLDER = os.path.join(BASE_DIR, 'annotated_captions_ours')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
os.makedirs(CAPTION_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = 'a-much-more-secure-secret-key-for-sessions'
model_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'vip_llava', 'vip-llava-7b'))
print(f"[INFO] Attempting to load model from: {model_path}")

DEVICE = f"cuda:{DEVICE_ID}"

print(f"[INFO] Loading ViP-LLaVA model onto {DEVICE}...")

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path,
    model_name="vip-llava-7b",
    model_base=None,
    load_8bit=True,
    load_4bit=False,
    device="cuda"
)
model.eval()
model_device = next(model.parameters()).device
print(f"[INFO] Model loaded successfully! Actual device: {model_device}")

@app.before_request
def assign_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        print(f"[SESSION] New user detected. Assigned ID: {session['user_id']}")


def get_all_images():
    return sorted([
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

def get_image_stem(filename):
    return os.path.splitext(filename)[0]

def get_highlighted_image_base64(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    green = np.zeros_like(img)
    green[..., 0] = 20
    green[..., 1] = 255
    green[..., 2] = 57
    
    overlay = img.copy()
    overlay[mask > 127] = green[mask > 127]
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(blended, contours, -1, (47, 255, 173), thickness=5, lineType=cv2.LINE_AA)
    
    _, buf = cv2.imencode(".png", blended)
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def load_image_tensor_with_box(img_path, mask_path, image_processor):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        x1, y1, x2, y2 = 0, 0, img.shape[1] - 1, img.shape[0] - 1
    else:
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for w in range(4):
        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=(255, 0, 0))
    return image_processor.preprocess(pil, return_tensors='pt')['pixel_values'].half().to(DEVICE)

@app.route('/')
def index():
    print("\n" + "="*50)
    user_id = session['user_id']
    print(f"[INDEX] Route '/' accessed by User: {user_id}")
    
    images = get_all_images()
    if not images:
        return "No images found in uploads folder", 404
    
    for img in images:
        stem = get_image_stem(img)
        
        user_mask_path = os.path.join(ANNOTATED_FOLDER, user_id, f"{stem}.png")
        user_caption_path = os.path.join(CAPTION_FOLDER, user_id, f"{stem}.json")
        
        has_mask = os.path.exists(user_mask_path)
        has_caption = os.path.exists(user_caption_path)
        
        if not has_caption:
            if has_mask:
                print(f"[INDEX] User '{user_id}' → TEXT annotation for {img}")
                return redirect(url_for('text_annotate', image_name=img))
            else:
                print(f"[INDEX] User '{user_id}' → MASK annotation for {img}")
                return render_template('mask_annotate.html', start_image=img)
    
    print(f" -> User '{user_id}' has completed all annotations!")
    return f"<h1>All annotations are complete! Thank you.</h1><p>Your completion code is: <strong>{user_id}</strong></p>"


@app.route('/api/save_annotation/<image_stem>', methods=['POST'])
def save_mask_annotation(image_stem):
    user_id = session['user_id']
    data = request.get_json()

    if not data or 'image_data' not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        user_annotated_dir = os.path.join(ANNOTATED_FOLDER, user_id)
        os.makedirs(user_annotated_dir, exist_ok=True)
        save_path = os.path.join(user_annotated_dir, f"{image_stem}.png")

        header, encoded = data['image_data'].split(",", 1)
        image_data = base64.b64decode(encoded)
        img = Image.open(BytesIO(image_data))
        img.save(save_path, 'PNG')

        print(f"[SAVE_MASK] Saved mask for user '{user_id}'.")

        original_image = next((f for f in get_all_images() if get_image_stem(f) == image_stem), None)
        redirect_url = url_for('text_annotate', image_name=original_image)
        return jsonify({"success": True, "redirect_url": redirect_url}), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/text_annotate/<image_name>', methods=['GET', 'POST'])
def text_annotate(image_name):
    user_id = session['user_id']
    img_path = os.path.join(UPLOAD_FOLDER, image_name)
    image_stem = get_image_stem(image_name)
    mask_path = os.path.join(ANNOTATED_FOLDER, user_id, f"{image_stem}.png")
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return f"[ERROR] Image or user's mask not found for {image_name}", 404

    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "reset":
            print(f"[TEXT] User '{user_id}' requested RESET for {image_name}")
            session.pop(f"{user_id}_chosen_ids", None)
            session.pop(f"{user_id}_prompt_ids", None)
            session.pop(f"{user_id}_current_image", None)
            return redirect(url_for('text_annotate', image_name=image_name))

        if action == "finish":
            chosen_ids = session.get(f"{user_id}_chosen_ids", [])
            caption_text = tokenizer.decode(chosen_ids) if chosen_ids else ""

            user_caption_dir = os.path.join(CAPTION_FOLDER, user_id)
            os.makedirs(user_caption_dir, exist_ok=True)
            caption_path = os.path.join(user_caption_dir, f"{image_stem}.json")

            final_data = {
                "image": image_name,
                "caption": caption_text,
            }

            with open(caption_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            print(f"[TEXT] Caption saved for {image_name}.")

            session.pop(f"{user_id}_chosen_ids", None)
            session.pop(f"{user_id}_prompt_ids", None)
            session.pop(f"{user_id}_current_image", None)

            return jsonify({"redirect": "/"})

        selected = request.form.getlist("tokens")
        custom = request.form.get("custom_token", "").strip()
        if custom: selected.append(custom)

        chosen_ids_key = f"{user_id}_chosen_ids"
        chosen_ids = session.get(chosen_ids_key, [])
        for tok in selected:
            if tok:
                tid = tokenizer(tok, add_special_tokens=False).input_ids[0]
                chosen_ids.append(tid)
        session[chosen_ids_key] = chosen_ids
        
        return redirect(url_for('text_annotate', image_name=image_name))

    prompt_ids_key = f"{user_id}_prompt_ids"
    chosen_ids_key = f"{user_id}_chosen_ids"
    current_image_key = f"{user_id}_current_image"
    
    if prompt_ids_key not in session or session.get(current_image_key) != image_name:
        print(f"[TEXT] Initializing session for user '{user_id}' on image: {image_name}")
        session[current_image_key] = image_name
        
        
        prompt = "Describe the object in the red box."
        prompt_ids = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + "\n" + prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze(0).tolist()
        session[prompt_ids_key] = prompt_ids
        
        prefix_text = "the"
        prefix_token_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        session[chosen_ids_key] = prefix_token_ids
        print(f"[TEXT] Initialized with prefix: '{prefix_text}'")

    image_base64 = get_highlighted_image_base64(img_path, mask_path)
    image_tensor = load_image_tensor_with_box(img_path, mask_path, image_processor)
    
    chosen_ids = session.get(chosen_ids_key, [])
    input_ids = torch.tensor(session[prompt_ids_key] + chosen_ids).unsqueeze(0).cuda()
    
    special_token_ids = {tid for tid in {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id} if tid is not None}
    with torch.no_grad():
        out = model(input_ids=input_ids, images=image_tensor, use_cache=True)
        probs = F.softmax(out.logits[:, -1], dim=-1).squeeze(0)
        topk = torch.topk(probs, k=20)
    
    top_tokens = []
    for token_id in topk.indices:
        if len(top_tokens) >= 5: break
        if token_id.item() not in special_token_ids:
            decoded_token = tokenizer.decode([token_id.item()])
            cleaned_token = decoded_token.strip()
            if cleaned_token and cleaned_token != '.':
                top_tokens.append(decoded_token)

    return render_template(
        "text_annotate.html",
        image_base64=image_base64,
        image_name=image_name,
        caption_text="Describe the object in the green mask.",
        top_tokens=top_tokens,
        prev_input=tokenizer.decode(chosen_ids) if chosen_ids else "",
        done=False
    )


@app.route('/api/images')
def get_image_list():
    try:
        return jsonify(get_all_images())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/masks/<image_stem>')
def get_mask_list(image_stem):
    try:
        mask_dir = os.path.join(MASK_FOLDER, image_stem)
        if not os.path.isdir(mask_dir):
            return jsonify([])

        all_mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        pseudo_masks = [f for f in all_mask_files if f == 'pseudo_mask.png']
        other_masks = sorted([f for f in all_mask_files if f != 'pseudo_mask.png'])
        mask_files = pseudo_masks + other_masks
        
        print(f"[MASK_API] {image_stem} mask order: {mask_files}")
        
        masks_data = []
        for filename in mask_files:
            file_path = os.path.join(mask_dir, filename)
            with open(file_path, "rb") as image_file:
                file_bytes = image_file.read()
                encoded_string = base64.b64encode(file_bytes).decode('utf-8')
                
                np_arr = np.frombuffer(file_bytes, np.uint8)
                mask_img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                contours_list = []
                if contours:
                    contours_list = [c.squeeze().tolist() for c in contours]

                masks_data.append({
                    "name": filename,
                    "data": encoded_string,
                    "contour": contours_list
                })
        
        return jsonify(masks_data)
        
    except Exception as e:
        print(f"Error in get_mask_list for {image_stem}: {e}")
        return jsonify({"error": str(e)}), 500

        
@app.route('/complete')
def complete():
    total_images = len(get_all_images())
    return render_template('complete.html', total=total_images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/masks/<image_stem>/<filename>')
def mask_file(image_stem, filename):
    return send_from_directory(os.path.join(MASK_FOLDER, image_stem), filename)

if __name__ == '__main__':
    print(f"📁 UPLOAD_FOLDER: {UPLOAD_FOLDER}")
    print(f"📁 MASK_FOLDER: {MASK_FOLDER}")
    print(f"📁 ANNOTATED_FOLDER: {ANNOTATED_FOLDER}")
    print(f"📁 CAPTION_FOLDER: {CAPTION_FOLDER}")
    app.run(debug=True, use_reloader=True, port=3000)