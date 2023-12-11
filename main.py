import gc
import os
import random
import tomesd
import torch
from io import BytesIO
from torch import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPTextModel, CLIPTokenizer
import webcolors
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from PIL import Image, ImageColor, ImageDraw, ImageFont
import textwrap


def get_image_file(file):
    return Image.open(file)


def get_logo(logo_image):
    return Image.open(logo_image).convert('RGBA')


def create_prompt_with_colour(input_prompt, hex_code):
    # we want to generate an image including the colour we set in the app
    input_prompt += ", including the colour of {}".format(webcolors.hex_to_name(hex_code))
    return input_prompt


def create_negative_prompt():
    negative_prompt_list = ['low quality, ', 'duplicate, ',
                            '(worst quality:1.4), ',
                            'bad anatomy, ',
                            'nudity, ',
                            '(inaccurate limb:1.2), ',
                            'bad composition, ',
                            'inaccurate eyes, ',
                            'extra digit, ',
                            'fewer digits, ',
                            '(extra arms:1.2)']

    negative_prompt = ""
    for item in negative_prompt_list:
        negative_prompt += item

    return negative_prompt


def generate_image(prompt, image):
    torch.cuda.empty_cache()

    #model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    # model_id = "helenai/stabilityai-stable-diffusion-2-1-ov"
    # model_id = "CompVis/stable-diffusion-v1-4"
    #pipe.save_pretrained("openvino-sd-xl-refiner-1.0")
    #a little home on the roadside, a car going on the road, snowy weather

    # we use a stable diffusion model form the Stability AI
    # Stability AI model is the fastest model comparing to CompVis and RunwayML.
    # also, this model generates images so close to the input text and input image
    # however, another two models could not generate images close to the inputs as the Stability AI do
    #model_id = "stabilityai/stable-diffusion-2-1-base"

    # set the pipeline
    #model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, use_safetensors=True)
    #pipe.save_pretrained("./stabilityai_cpu")

    # we use the DPM Solver as a scheduler for accelerating CPU process
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # we work on cpu instead of cuda
    pipe = pipe.to('cpu')

    # token merging
    tomesd.apply_patch(pipe, ratio=0.5)

    # set the generator with manual seed in order to generate different images at each try
    generator = torch.Generator('cpu').manual_seed(random.randint(1, 9999999999999))

    # create negative prompt
    negative_prompt = create_negative_prompt()

    # the optimal values of strength and guidance scale are 0.7 and 7.5, respectively.
    # at the 20 steps we generate images with high quality, but it takes long time
    new_image = pipe(prompt=prompt, image=image, strength=0.7,
                     guidance_scale=7.5, generator=generator,
                     num_inference_steps=20, negative_prompt=negative_prompt).images[0]
    new_image.save(f"static/results/generated_image.png")

    return new_image


# set a function for obtaining an image with rounded corners
def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2 - 1, rad * 2 - 1), fill=256)
    alpha = Image.new('L', im.size, 256)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im


def create_ad_template(generated_image, input_logo, button_text, punchline_text, colour_in_template):
    # create the ad template
    ad_template = Image.new('RGBA', (1024, 1024), ImageColor.getrgb("#FFFFFF"))
    generated_image = generated_image.resize((int(ad_template.size[0]/2), int(ad_template.size[1]/2)))
    input_logo = input_logo.resize((int(ad_template.size[0]/8), int(ad_template.size[1]/8)))
    ad_template.paste(input_logo, (int(7*(ad_template.size[0]/16)), int(ad_template.size[0]/32)), mask=input_logo)

    # create a template for the image with rounded corners
    rounded_image = add_corners(generated_image, 50)
    ad_template.paste(rounded_image, (int(ad_template.size[0]/4), int(3*(ad_template.size[0]/16))), rounded_image)

    # set the fonts on the button and punchline
    sans16 = ImageFont.truetype("fonts/helvetica-rounded-bold-5871d05ead8de.otf", int(3*(min(ad_template.size[0],
                                                                                             ad_template.size[1])/64)))
    times = ImageFont.truetype("fonts/Helvetica-Bold.ttf", int(3*(min(ad_template.size[0], ad_template.size[1])/128)))

    # create the punchline
    para = textwrap.wrap(punchline_text, width=40)
    para_button = button_text.split(' ')

    current_height, padding = int(3*(ad_template.size[1]/4)), 5
    for punchline in para:
        width, height = ImageDraw.Draw(ad_template).textsize(punchline, font=sans16)
        ImageDraw.Draw(ad_template).text((int(ad_template.size[0] - width) / 2, current_height), punchline,
                                         fill=ImageColor.getrgb(colour_in_template), font=sans16)
        current_height += height + padding

    # create the button
    current_height_button, pad_button = 50, 10
    current_height_text, pad_button_text = 32, 10
    current_width_button = 100

    # arrange position and size of the button so that they will be resized at each running the application
    for button_line in para_button:
        w, h = ImageDraw.Draw(ad_template).textsize(button_line, font=times)
        text_size = times.getsize(button_text)
        line_size = times.getsize(button_line)

        # arrange width of the button so that it will be resized based on the text on the button
        current_width_button += int(line_size[0]) + pad_button

        button_image = Image.new('RGBA', (current_width_button, current_height_button), ImageColor.getrgb("#FFFFFF"))

        # create a sub-template for the button with rounded corners
        ImageDraw.Draw(button_image).rounded_rectangle(((0, 0), (current_width_button, current_height_button)), 10,
                                                       fill=ImageColor.getrgb(colour_in_template))

        # set the position of text on the button
        position_x_text_on_button = int((ad_template.size[0] / current_width_button) + int((current_width_button - text_size[0]) / 2))
        position_y_text_on_button = current_height_text / 2

        # centering the text on the button
        ImageDraw.Draw(button_image).text((position_x_text_on_button, position_y_text_on_button),
                                          button_text, fill=ImageColor.getrgb("#FFFFFF"), font=times)

        # centering the button
        ad_template.paste(button_image, (int((ad_template.size[0]) / 2) - int(current_width_button / 2)+pad_button,
                                         int(11 * (ad_template.size[1] / 12))))

    ad_template.save(f"static/results/ad_template.png")
