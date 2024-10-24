from flask import Flask, request, jsonify, render_template
import replicate
import os
from dotenv import load_dotenv
import logging
import json
import sys

# Sæt op logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Indlæs miljøvariabler
load_dotenv()

app = Flask(__name__)

# Konfigurer Replicate API-token
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    logger.error("Ingen Replicate API-token fundet!")
    sys.exit("Indstil venligst REPLICATE_API_TOKEN i .env-filen")

os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

def load_prompt():
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        return f.read().strip()

def generate_recipe_prompt(ingredients, servings, meal_type):
    prompt = load_prompt()
    return (f"{prompt} Brug nogle eller alle af disse ingredienser: {ingredients}. "
            f"Inkluder følgende oplysninger i opskriften: "
            f"- Tidsforbrug: [Hvor lang tid det tager at forberede og tilberede retten] "
            f"- Sværhedsgrad: [Nem, Mellem, Svær] "
            f"- Ernæringsoplysninger: [Kalorier, protein, fedt osv.] "
            f"Returnér svaret i dette nøjagtige JSON-format og sørg for, at det er på dansk: "
            f"{{\"title\": \"Opskriftens Titel\", \"description\": \"Kort beskrivelse af opskriften\", "
            f"\"ingredients\": [\"ingrediens 1\", \"ingrediens 2\"], "
            f"\"instructions\": [\"trin 1\", \"trin 2\"], "
            f"\"servings\": \"{servings}\", \"meal_type\": \"{meal_type}\", "
            f"\"prep_time\": \"Tidsforbrug\", \"difficulty\": \"Sværhedsgrad\", "
            f"\"nutrition\": {{\"calories\": \"Kalorier\", \"protein\": \"Protein\", \"fat\": \"Fedt\"}}, "
            f"\"image_url\": \"URL til opskrift billede\"}}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-recipes', methods=['POST'])
def generate_recipes():
    try:
        data = request.json
        ingredients = data.get('ingredients', '').strip()
        servings = data.get('servings', '1')  # Default to 1 if not provided
        meal_type = data.get('mealType', 'morgenmad')  # Default to breakfast
        
        logger.info(f"Modtagne ingredienser: {ingredients}, Antal personer: {servings}, Type af måltid: {meal_type}")

        if not ingredients:
            logger.error("Ingen ingredienser angivet")
            return jsonify({'error': 'Ingen ingredienser angivet'}), 400

        logger.info("Kalder Replicate API for at generere opskrift")

        try:
            # Generate the prompt using the specified format
            prompt = generate_recipe_prompt(ingredients, servings, meal_type)

            output = replicate.run(
                "meta/llama-2-70b-chat",
                input={
                    "prompt": prompt,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_new_tokens": 500,
                    "min_new_tokens": 1
                }
            )
            logger.debug(f"Raw API response: {output}")

            if not output:
                logger.error("Modtaget tomt output fra Replicate API")
                return jsonify({'error': 'Modtaget tomt output fra API'}), 500

            # Flet output til en enkelt streng
            raw_response = ''.join(output).strip()
            logger.debug(f"Raw opskrift output: {raw_response}")

            # Parse JSON-respons
            try:
                recipe_data = json.loads(raw_response)
                logger.info("Succesfuldt parsed opskrift data")
            except json.JSONDecodeError as e:
                logger.error(f"JSON-dekodningsfejl: {str(e)} - Raw respons: {raw_response}")
                return jsonify({'error': 'Der opstod en fejl under parsing af opskriften'}), 500

            # Tilføj billede-URL (hvis der er en)
            image_url = recipe_data.get('image_url', 'default_image.jpg')  # Brug et standardbillede hvis ingen er angivet
            recipe_data['image_url'] = image_url

            return jsonify([recipe_data])

        except replicate.exceptions.ReplicateError as api_error:
            logger.error(f"Replicate API-fejl: {api_error}")
            return jsonify({'error': f'API-fejl: {str(api_error)}. Tjek venligst din anmodning og API-token.'}), 500

    except Exception as e:
        logger.error(f"Uventet fejl: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
