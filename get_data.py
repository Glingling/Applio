 import requests
import json
from typing import Optional, List, Dict
import time

class NotionTextExtractor:
    def __init__(self, token: str):
        """
        Initialise le client Notion avec le token d'intégration.

        Args:
            token (str): Token d'intégration Notion
        """
        self.token = token
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

    def search_pages(self, query: str) -> List[Dict]:
        """
        Recherche des pages Notion par leur titre.

        Args:
            query (str): Texte à rechercher dans les titres des pages

        Returns:
            List[Dict]: Liste des pages trouvées
        """
        url = f"{self.base_url}/search"
        payload = {"query": query, "filter": {"property": "object", "value": "page"}}

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()["results"]

    def get_child_pages(self, page_id: str) -> List[Dict]:
        """
        Récupère la liste des pages enfants d'une page donnée.

        Args:
            page_id (str): ID de la page parent

        Returns:
            List[Dict]: Liste des pages enfants
        """
        url = f"{self.base_url}/blocks/{page_id}/children"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        blocks = response.json()["results"]
        child_pages = []

        for block in blocks:
            if block["type"] == "child_page":
                child_pages.append(
                    {"id": block["id"], "title": block["child_page"]["title"]}
                )

        return child_pages

    def get_page_content(self, page_id: str) -> str:
        """
        Récupère le contenu textuel d'une page Notion.

        Args:
            page_id (str): ID de la page Notion

        Returns:
            str: Contenu textuel de la page
        """
        url = f"{self.base_url}/blocks/{page_id}/children"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        blocks = response.json()["results"]
        text_content = []

        for block in blocks:
            if block["type"] in ["paragraph", "heading_1", "heading_2", "heading_3"]:
                rich_text = block[block["type"]]["rich_text"]
                if rich_text:
                    text_content.append(rich_text[0]["plain_text"])

        return "\n".join(text_content)


# Configuration de base
API_BASE_URL = "http://localhost:8000"

def start_tts_task(text: str):
    """
    Démarre une tâche TTS et retourne son ID
    """
    url = f"{API_BASE_URL}/tts/"

    # Données pour la requête TTS
    payload = {
    "tts_text": text,
    "tts_voice": "fr-FR-HenriNeural",
    "tts_rate": 0,
    "pitch": 0,
       "filter_radius": 3,
    "index_rate": 0.75,
    "volume_envelope": 1,
    "protect": 0.5,
    "hop_length": 128,
    "f0_method": "rmvpe",
    "pth_path:": "logs/model/model.pth",
    "index_path": "logs/metadata/metadata.index",
    "split_audio": False,
    "f0_autotune": False,
    "f0_autotune_strength": 1,
    "clean_audio": True,
    "clean_strength": 0.5,
    "export_format": "wav",
    "f0_file": "None",
    "embedder_model": "contentvec",
    "embedder_model_custom": None,
}

    # Envoi de la requête POST
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["task_id"]
    else:
        raise Exception(f"Erreur lors du démarrage de la tâche: {response.text}")

def check_task_status(task_id: str):
    """
    Vérifie le statut d'une tâche TTS
    """
    url = f"{API_BASE_URL}/tts/{task_id}/status"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur lors de la vérification du statut: {response.text}")

def get_task_result(task_id: str):
    """
    Récupère le résultat d'une tâche TTS terminée
    """
    url = f"{API_BASE_URL}/tts/{task_id}/result"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur lors de la récupération du résultat: {response.text}")

def delete_task(task_id: str):
    """
    Supprime une tâche TTS et ses fichiers associés
    """
    url = f"{API_BASE_URL}/tts/{task_id}"
    response = requests.delete(url)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur lors de la suppression de la tâche: {response.text}")

def wait_for_task_completion(task_id: str, check_interval: int = 5, timeout: int = 300):
    """
    Attend que la tâche soit terminée avec un timeout
    """
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"La tâche n'a pas été terminée après {timeout} secondes")

        status = check_task_status(task_id)
        if status["status"] == "completed":
            return get_task_result(task_id)
        elif status["status"] == "failed":
            raise Exception(f"La tâche a échoué: {status.get('error')}")

        time.sleep(check_interval)

def get_child_pages(pages, extractor, search_query):
    for i, page in enumerate(pages, 1):
        print(
            page.get("properties", {}).get("title", [{"plain_text": "Sans titre"}])[
                "title"
            ][0]
        )
        print(
            f"{i}. {page.get('properties', {}).get('title', [{'plain_text': 'Sans titre'}])['title'][0]['plain_text']}"
        )

    for page in pages:
        if (
            page.get("properties", {}).get("title", [{"plain_text": "Sans titre"}])[
                "title"
            ][0]["plain_text"]
            == search_query
        ):
            selected_page = page

    # Récupérer et afficher les pages enfants
    child_pages = extractor.get_child_pages(selected_page["id"])

    all_content = []

    if child_pages:
        for child in child_pages:
            content = extractor.get_page_content(child["id"])
            lignes = content.split("\n")
            lignes.pop()
            content = ""
            for i in range(len(lignes)):
                if len(content) > 0:
                    content = content + "\n" + str(lignes[i])
                else:
                    content = lignes[i]
            all_content.append({
                'titre': f"{search_query}_{child['title']}",
                'content': content
            })
        print(all_content)
    else:
        print("\nAucune page enfant trouvée.")

    return all_content

def use_appolo(child_content):
    # Utilisation simple
    task_id = start_tts_task(child_content["content"])
    result = wait_for_task_completion(task_id)
    print(result)
    result = get_task_result(task_id)
    open(f"{child_content['titre']}.wav", 'wb').write(result.content)
    print(result)
    delete_task(task_id)

def main():
    # Exemple d'utilisation
    token = "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"
    extractor = NotionTextExtractor(token)

    # Demander le nom de la page à rechercher
    search_query = input("\nEntrez le nom de la page à rechercher : ")
    pages = extractor.search_pages(search_query)

    if not pages:
        print(f"\nAucune page trouvée pour la recherche : {search_query}")
        return

    all_child_content = get_child_pages(pages=pages, extractor=extractor, search_query=search_query)

    for child_content in all_child_content:
        if (child_content["content"] != ""):
            use_appolo(child_content)
        else :
            print("\nPas de texte à convertir.")

if __name__ == "__main__":
    main()