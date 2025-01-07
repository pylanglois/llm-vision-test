import base64

from dotenv import load_dotenv

import ac35
import aoai
import mistral

load_dotenv()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    b64_images = [encode_image(i) for i in ["./IMG_1148.JPG", "./IMG_1149.JPG", "./IMG_1150.JPG"]]
    system_prompt = """
    Analyse le contenu des images suivantes. 
    Réponds au format json suivant: {"description":"lait", "montant":"12.99$"} ou {"clé":"valeur"}.
    Si tu ne peux pas répondre, retourne {"erreur":"raison"}
    """
    questions = [
        "Quel est le prix des chaussettes?",
        "Quel est le prix pour les tuyaux?",
        "Combien coûte les chipits?",
        "qui m'a servi au IGA?",
        "quel est le prix de la veste?"
    ]
    models = [aoai, ac35, mistral]
    # models = [ac35]
    # models = [aoai]
    #models = [mistral]
    for model in models:
        print(f"\nUsing model: {model.__name__}")
        for question in questions:
            response = model.text_completion(system_prompt, question, b64_images)
            print(question, response)


if __name__ == '__main__':
    main()
