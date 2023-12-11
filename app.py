import re
import webcolors
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import PIL
from PIL import Image, ImageColor
from main import *

UPLOAD_FOLDER = 'C:/Users/W10/PycharmProjects/stable_diffusion/uploaded_images'
ALLOWED_EXTENSIONS = {'.png', '.jpg'}

app = Flask(__name__, template_folder='template')
app.config['UPLOAD FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def show_template():

    if request.method == 'POST':

        file = request.files['image']
        if os.path.splitext(file.filename)[1] not in ALLOWED_EXTENSIONS:
            return redirect(url_for('get_file_error'))

        file.save(secure_filename(file.filename))

        prompt = request.form.get('prompt')
        colour = request.form.get('colour_hex_code_in_image')
        colour_in_template = request.form.get('colour_hex_code_in_ad_template')
        button = request.form.get('button_text')
        punchline = request.form.get('punchline_text')

        logo = request.files['logo']
        if os.path.splitext(logo.filename)[1] not in ALLOWED_EXTENSIONS:
            return redirect(url_for('get_file_error'))
        logo.save(secure_filename(logo.filename))

        image = get_image_file(file)
        logo = get_logo(logo)

        if not re.search(r'^#(?:[0-9a-fA-F]{1,2}){3}$', colour):
            return redirect(url_for('get_error'))
        elif not re.search(r'^#(?:[0-9a-fA-F]{1,2}){3}$', colour_in_template):
            return redirect(url_for('get_error'))
        elif webcolors.hex_to_name(colour) not in list(PIL.ImageColor.colormap.keys()):
            return redirect(url_for('get_error'))

        prompt = create_prompt_with_colour(prompt, colour)
        generated_image = generate_image(prompt, image)
        create_ad_template(generated_image, logo, button, punchline, colour_in_template)

        return redirect(url_for('show_ad'))

    return render_template('index.html')


@app.route('/generated_image', methods=['GET', 'POST'])
def show_image():
    image = '/generated_image.png'
    return render_template('result.html', image=image)


@app.route('/ad_template', methods=['GET'])
def show_ad():
    return render_template('result_ad_template.html')


@app.route('/error', methods=['GET'])
def get_error():
    error = "The hex code you have stated does not exist in the colour list"
    return render_template('500.html', error=error)


@app.route('/file_error', methods=['GET'])
def get_file_error():
    file_error = "Extension of the file has not been permitted"
    return render_template('404.html', file_error=file_error)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(host='0.0.0.0', port=3000, debug=True)