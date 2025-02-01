from flask import Flask, request, jsonify
import replicate
import os
import asyncio
import nest_asyncio

app = Flask(__name__)

@app.route('/generate_image', methods=['POST'])
async def generate_image():
    data = request.get_json()
    prompt = data.get('prompt', 'Une créature chibi mignonne avec de grands yeux et des ailes de fée, style anime')
    
    # Récupération de la clé API Replicate depuis une variable d'environnement
    replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_api_token:
        return jsonify({"error": "Clé API Replicate manquante"}), 500
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_token

    # Modèle SDXL avec le commit hash spécifié
    model = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"

    # Paramètres de génération
    params = {
        "prompt": prompt,
        "negative_prompt": "flou, peu détaillé, réaliste",
        "width": 512,
        "height": 512,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "scheduler": "K_EULER_ANCESTRAL",
    }

    # Exécution de la génération dans un thread séparé
    output = await asyncio.to_thread(replicate.run, model, input=params)

    # Récupérer l'URL de l'image générée
    if isinstance(output, list):
        image_url = output[0]
    else:
        image_url = output.get()
    print(f"Image URL : {image_url}")
    
    return jsonify({'image_url': image_url})

if __name__ == '__main__':
    nest_asyncio.apply()
    # Pour le développement local (pas pour Render)
    app.run(host="0.0.0.0", port=5000)
