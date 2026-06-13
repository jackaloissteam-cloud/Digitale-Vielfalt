import gradio as gr
import spaces
from diffusers import DiffusionPipeline
import torch

# Wir laden das aktuell beste kostenlose Modell: FLUX.1-schnell
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16).to(device)

@spaces.GPU # Diese Zeile erlaubt uns die Nutzung der starken Hugging Face Grafikkarte
def generate_image(style, gender, body, pose, setting, light):
    # Hier wird Ihre Formel zusammengebaut
    prompt = f"{style} of a {body} {gender}, {pose}, {setting}, {light}, photorealistic, highly detailed skin texture"
    
    # Bildgenerierung
    image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    return image

# Das Design der App (Gradio)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Diversity & Anatomy Explorer")
    gr.Markdown("Wählen Sie die Parameter nach Ihrer Formel aus, um ein Bild zu generieren.")
    
    with gr.Row():
        with gr.Column():
            # Die Komponenten Ihrer Formel als Auswahlmenüs
            style = gr.Dropdown(["Raw Photo", "Oil Painting", "Charcoal Sketch", "Cinematic"], label="Stil/Medium", value="Raw Photo")
            gender = gr.Dropdown(["Female", "Male", "Non-binary", "Trans"], label="Gender/Ethnie", value="Female")
            body = gr.Textbox(placeholder="z.B. curvy, muscular, plus-size, petite", label="Körperform/Gewicht")
            pose = gr.Textbox(placeholder="z.B. reclining, sitting, dynamic dancing", label="Stellung/Pose")
            setting = gr.Textbox(placeholder="z.B. minimalist studio, nature, rustic barn", label="Hintergrund")
            light = gr.Dropdown(["Natural Light", "Chiaroscuro", "Neon", "Golden Hour"], label="Beleuchtung", value="Natural Light")
            
            btn = gr.Button("Bild generieren", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(label="Ergebnis")
            gr.Markdown("Rechtsklick auf das Bild zum Speichern.")

    btn.click(fn=generate_image, inputs=[style, gender, body, pose, setting, light], outputs=output_img)

demo.launch()
