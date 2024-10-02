from flask import Flask, request, render_template
from style_transfer import run_style_transfer
import os

app = Flask(__name__)
IMAGE_PATH = 'static/generated_output.png'
UPLOAD_FOLDER = 'static/uploads/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route(rule="/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    else:
        num_steps = request.form.get("num_steps")
        display_after_step = request.form.get("display_after_step")
        content_image = request.files.get('content_image')
        style_image = request.files.get('style_image')

        if content_image:
            content_image_path = os.path.join(UPLOAD_FOLDER, content_image.filename)
            content_image.save(content_image_path)

        if style_image:
            style_image_path = os.path.join(UPLOAD_FOLDER, style_image.filename)
            style_image.save(style_image_path)
        
        # Run your style transfer function with parameters
        generated_image = run_style_transfer(int(num_steps), int(display_after_step), content_path=content_image_path, style_path=style_image_path)
        generated_image.save(IMAGE_PATH)
        # Assuming run_style_transfer saves the image to IMAGE_PATH
        return render_template("index.html", content_image = content_image, style_image=style_image, output_image=IMAGE_PATH)
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8888, use_reloader=False)