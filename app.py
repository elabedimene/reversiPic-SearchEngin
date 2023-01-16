# Import des bibliothèques nécessaires
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import base64

image_dir = 'data'
# Liste des noms de fichiers des images
image_filenames = os.listdir(image_dir)
images_names = []
# Chargement des images en mémoire
images = []
original_images = []
for filename in image_filenames:
    # Chargement de l'image en mémoire
    image = plt.imread(os.path.join(image_dir, filename))
    images_names.append(filename)
    # Conversion de l'image en tableau NumPy et ajout à la liste
    images.append(image)

# Initialiser la structure de données pour stocker les signatures
signatures = []
# Pour chaque image, extraire les descripteurs de couleur et de texture
for image in images:
    # Redimensionner l'image à une taille fixe
    image = cv2.resize(image, (256, 256))
    # Extraire le descripteur de couleur (histogramme de couleur)
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_hist = cv2.normalize(color_hist, None).flatten()

    # Extraire le descripteur de texture (histogramme de gradient orienté)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    texture_hist = cv2.calcHist([gray_image], [0], None, [16], [0, 256])
    texture_hist = cv2.normalize(texture_hist, None).flatten()

    # Ajouter les descripteurs de couleur et de texture à la structure de données
    signatures.append({
        'color': color_hist,
        'texture': texture_hist
    })

# Afficher les signatures extraites
# print(signatures)

def compute_image_signature(reference_image):
  reference_image = cv2.resize(reference_image, (256, 256))
  color_hist = cv2.calcHist([reference_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
  color_hist = cv2.normalize(color_hist, None).flatten()
  gray_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
  texture_hist = cv2.calcHist([gray_image], [0], None, [16], [0, 256])
  texture_hist = cv2.normalize(texture_hist, None).flatten()
  signature = {
        'color': color_hist,
        'texture': texture_hist
    }
  return signature


def get_signatures_from_database():
  return signatures

def compute_similarities(reference_signature, signatures):
  similarities = []
  for signature in signatures:
      color_similarity = np.dot(reference_signature['color'], signature['color']) / (np.linalg.norm(color_hist) * np.linalg.norm(signature['color']))
      texture_similarity = np.dot(reference_signature['texture'], signature['texture']) / (np.linalg.norm(texture_hist) * np.linalg.norm(signature['texture']))
      similarity = (color_similarity + texture_similarity) / 2
      similarities.append(similarity)
  return similarities

def get_most_similar_indices(similarities):
  sorted_indices = np.argsort(similarities)[::-1]
  return sorted_indices


def get_most_similar_images(most_similar_indices,similarity):
  most_similarity = []
  most_similar_images = []
  for i in most_similar_indices[:5]:  # afficher les 5 images les plus similaires
    most_similarity.append(round(similarity[i],2))
    image = cv2.imread('data/'+images_names[i])
    most_similar_images.append(image)
  return most_similar_images,most_similarity


from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/search-similar-images', methods=['POST'])
def search_similar_images():
    # Récupérer l'image de référence à partir de la requête HTTP
    reference_image = request.files['referenceImage']

    # Charger l'image de référence en utilisant OpenCV
    reference_image_np = cv2.imcode(reference_image.read(), np.unit8)
    reference_image_cv = cv2.imdecode(reference_image_np, cv2.IMREAD_UNCHANGED)

    # Calculer la signature de l'image de référence
    reference_signature = compute_image_signature(reference_image_cv)

    # Récupérer les signatures de toutes les images de la base de données
    signatures = get_signatures_from_database()

    # Calculer les similarités entre l'image de référence et toutes les images de la base de données
    similarities = compute_similarities(reference_signature, signatures)

    # Récupérer les indices des images les plus similaires
    most_similar_indices = get_most_similar_indices(similarities)

    # Récupérer les images les plus similaires à partir de la base de données
    most_similar_images,most_similarity = get_most_similar_images(most_similar_indices,similarities)

    # Encoder l'image de référence et les images les plus similaires en tant qu'objets JSON
    most_similar_images_base64 = []

    for img in most_similar_images:
        result , encoded_image = cv2.imencode('.JPG' , img)
        if result :
            encoded_image_base64 = base64.b64encode(encoded_image.tobytes()).decode()
            most_similar_images_base64.append(encoded_image_base64)
        else:
            print("Error : Failed o encode image .")
    # Renvoyer l'image de référence et les images les plus similaires à l'application Angular
    return jsonify({
        'mostSimilarity': most_similarity,
        'mostSimilarImages': most_similar_images_base64
    })

if __name__ == '__main__':
    app.run(debug=True)