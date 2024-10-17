import os
import logging
from flask import Flask, request, send_file, jsonify, redirect
from werkzeug.utils import secure_filename
from image_processor_final import process_image, detect_objects, process_image_for_inpainting

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Configure folders
MEDIA_FOLDER = 'media'

# Ensure the media folder exists
if not os.path.exists(MEDIA_FOLDER):
    os.makedirs(MEDIA_FOLDER)

app.config['MEDIA_FOLDER'] = MEDIA_FOLDER

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_user_folder_structure(user_id):
    """Create user folder structure if it doesn't exist"""
    user_folder = os.path.join(MEDIA_FOLDER, user_id)
    input_folder = os.path.join(user_folder, 'input')
    output_folder = os.path.join(user_folder, 'output')

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    return input_folder, output_folder

def get_last_saved_image(output_folder):
    """Retrieve the most recently saved image from the output folder"""
    files = [f for f in os.listdir(output_folder) if allowed_file(f)]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)), reverse=True)
    return os.path.join(output_folder, files[0])

def get_next_output_number(output_folder):
    """Get the next available output number"""
    existing_files = [f for f in os.listdir(output_folder) if f.startswith('output') and f.endswith('.png')]
    if not existing_files:
        return 1
    numbers = [int(f[6:-4]) for f in existing_files if f[6:-4].isdigit()]
    return max(numbers) + 1 if numbers else 1

@app.route('/wall_color_change', methods=['POST'])
def wall_color_change():
    # Get user_id from form-data
    user_id = request.form.get('user_id')
    if not user_id:
        return "Error: Missing user_id", 400

    # Create folders for the user
    input_folder, output_folder = create_user_folder_structure(user_id)
    
    try:
        if 'image' not in request.files:
            app.logger.error("No file part in the request")
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            app.logger.error("No selected file")
            return redirect(request.url)
        
        text_prompt = request.form.get('text_prompt')
        color_name = request.form.get('color_name')
        
        if not text_prompt or not color_name:
            app.logger.error("Missing text_prompt or color_name")
            return "Error: Missing text prompt or color name", 400
        
        filename = secure_filename(image.filename)
        image_path = os.path.join(input_folder, filename)
        image.save(image_path)
        app.logger.info(f"Image saved to {image_path}")
        
        next_number = get_next_output_number(output_folder)
        output_filename = f'output{next_number}.png'
        output_path = os.path.join(output_folder, output_filename)
        app.logger.info(f"Processing image with text_prompt: {text_prompt}, color_name: {color_name}")
        process_image(image_path, text_prompt, color_name=color_name, output_path=output_path)
        
        app.logger.info(f"Image processed, output saved to {output_path}")
        
        return send_file(output_path, mimetype='image/png', as_attachment=True, download_name=output_filename)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@app.route('/detect_objects', methods=['POST'])
def detect_objects_route():
    # Get user_id from form-data
    user_id = request.form.get('user_id')
    if not user_id:
        return "Error: Missing user_id", 400

    # Create folders for the user
    input_folder, output_folder = create_user_folder_structure(user_id)

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(input_folder, filename)
        file.save(file_path)

        detected_objects = detect_objects(file_path, return_objects=True)
        app.logger.info(f"Detected objects: {detected_objects}")
        return jsonify({"detected_objects": detected_objects})
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/inpaint', methods=['POST'])
def inpaint_image():
    # Get user_id from form-data
    user_id = request.form.get('user_id')
    if not user_id:
        return "Error: Missing user_id", 400

    # Create folders for the user
    input_folder, output_folder = create_user_folder_structure(user_id)

    if 'image' not in request.files:
        return 'No image file uploaded', 400
    
    image_file = request.files['image']
    object_to_detect = request.form.get('object_to_detect')
    inpaint_prompt = request.form.get('inpaint_prompt')
    
    if not object_to_detect or not inpaint_prompt:
        return 'Missing object_to_detect or inpaint_prompt', 400

    image_path = os.path.join(input_folder, secure_filename(image_file.filename))
    image_file.save(image_path)

    inpainted_image = process_image_for_inpainting(image_path, object_to_detect, inpaint_prompt)

    next_number = get_next_output_number(output_folder)
    output_filename = f"output{next_number}.png"
    output_path = os.path.join(output_folder, output_filename)
    inpainted_image.save(output_path)

    return send_file(output_path, mimetype='image/png', as_attachment=True, download_name=output_filename)

@app.route('/inpaint_using_last_output', methods=['POST'])
def inpaint_using_last_output():
    # Try to retrieve user_id from form data or JSON data
    user_id = request.form.get('user_id') or request.json.get('user_id')
    if not user_id:
        return "Error: Missing user_id", 400

    # Create folders for the user
    _, output_folder = create_user_folder_structure(user_id)

    # Get the last saved image in the output folder
    last_image_path = get_last_saved_image(output_folder)
    if not last_image_path:
        return "Error: No previously processed image found for this user", 404

    # Retrieve the inpainting parameters from either form or JSON data
    object_to_detect = request.form.get('object_to_detect') or request.json.get('object_to_detect')
    inpaint_prompt = request.form.get('inpaint_prompt') or request.json.get('inpaint_prompt')
    
    if not object_to_detect or not inpaint_prompt:
        return 'Missing object_to_detect or inpaint_prompt', 400

    # Process the image for inpainting
    inpainted_image = process_image_for_inpainting(last_image_path, object_to_detect, inpaint_prompt)

    # Get the next output number and create the new filename
    next_number = get_next_output_number(output_folder)
    output_filename = f"output{next_number}.png"
    output_path = os.path.join(output_folder, output_filename)
    inpainted_image.save(output_path)

    return send_file(output_path, mimetype='image/png', as_attachment=True, download_name=output_filename)

if __name__ == '__main__':
    app.run(debug=True)