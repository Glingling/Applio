from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import sys
from functools import lru_cache
import shutil
import asyncio
import uuid
from datetime import datetime

# Import des fonctions du fichier original
from core import (
    run_infer_script,
    run_batch_infer_script,
    run_tts_script,
    run_preprocess_script,
    run_extract_script,
    run_train_script,
    run_index_script,
    run_model_information_script,
    run_model_blender_script,
    run_tensorboard_script,
    run_download_script,
    run_prerequisites_script,
    run_audio_analyzer_script,
    load_voices_data
)

app = FastAPI(title="Voice Converter API")

# Stockage en mémoire des tâches TTS
tts_tasks: Dict[str, Dict] = {}

class TTSRequest(BaseModel):
    tts_text: str
    tts_voice: str = "fr-FR-HenriNeural"
    tts_rate: int = 0
    pitch: int = 0
    filter_radius: int = 3
    index_rate: float = 0.75
    volume_envelope: int = 1
    protect: float = 0.5
    hop_length: int = 128
    f0_method: str = "rmvpe"
    pth_path: str = "logs/model/model.pth"
    index_path: str = "logs/metadata/metadata.index"
    split_audio: bool = False
    f0_autotune: bool =  False
    f0_autotune_strength: int = 1
    clean_audio: bool =  True
    clean_strength: float = 0.5
    export_format: str = "WAV"
    f0_file: str =  None
    export_format: str = "wav"
    embedder_model: str = "contentvec"

class TTSResponse(BaseModel):
    task_id: str
    status: str
    created_at: str

class TTSStatus(BaseModel):
    task_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    output_file: Optional[str] = None
    error: Optional[str] = None

async def process_tts_task(task_id: str, request: TTSRequest):
    """
    Fonction asynchrone pour traiter une tâche TTS en arrière-plan
    """
    try:
        temp_dir = f"temp_tts_{task_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        tts_output = os.path.join(temp_dir, "tts_output.wav")
        rvc_output = os.path.join(temp_dir, f"rvc_output.{request.export_format}")
        
        # Mise à jour du statut
        tts_tasks[task_id]["status"] = "processing"
        
        # Exécution du traitement TTS dans un thread séparé pour ne pas bloquer
        loop = asyncio.get_event_loop()
        message, output_file = await loop.run_in_executor(
            None,
            lambda: run_tts_script(
                tts_file="",
                tts_text=request.tts_text,
                tts_voice=request.tts_voice,
                tts_rate=request.tts_rate,
                pitch=request.pitch,
                filter_radius=request.filter_radius,
                index_rate=request.index_rate,
                volume_envelope=request.volume_envelope,
                protect=request.protect,
                hop_length=request.hop_length,
                f0_method=request.f0_method,
                output_tts_path=tts_output,
                output_rvc_path=rvc_output,
                pth_path=request.pth_path,
                index_path=request.index_path,
                split_audio=request.split_audio,
                f0_autotune=request.f0_autotune,
                f0_autotune_strength=request.f0_autotune_strength,
                clean_audio=request.clean_audio,
                clean_strength=request.clean_strength,
                export_format=request.export_format,
                f0_file=None,
                embedder_model=request.embedder_model,
            )
        )
        
        # Mise à jour du statut final
        tts_tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "output_file": output_file
        })
        
    except Exception as e:
        # En cas d'erreur, mise à jour du statut
        tts_tasks[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })
        
        # Nettoyage des fichiers temporaires en cas d'erreur
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        raise e

@app.post("/tts/", response_model=TTSResponse)
async def text_to_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    """
    Endpoint pour démarrer une tâche TTS asynchrone
    """
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    # Création de la tâche
    tts_tasks[task_id] = {
        "status": "pending",
        "created_at": created_at,
        "request": request.dict()
    }
    
    # Ajout de la tâche en arrière-plan
    background_tasks.add_task(process_tts_task, task_id, request)
    
    return TTSResponse(
        task_id=task_id,
        status="pending",
        created_at=created_at
    )

@app.get("/tts/{task_id}/status", response_model=TTSStatus)
async def get_tts_status(task_id: str):
    """
    Endpoint pour vérifier le statut d'une tâche TTS
    """
    if task_id not in tts_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tts_tasks[task_id]
    return TTSStatus(
        task_id=task_id,
        status=task["status"],
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
        output_file=task.get("output_file"),
        error=task.get("error")
    )

@app.get("/tts/{task_id}/result")
async def get_tts_result(task_id: str):
    """
    Endpoint pour récupérer le résultat d'une tâche TTS
    """
    if task_id not in tts_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tts_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {task['status']}"
        )
    
    if not task.get("output_file"):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(task["output_file"])

@app.delete("/tts/{task_id}")
async def delete_tts_task(task_id: str):
    """
    Endpoint pour supprimer une tâche TTS et ses fichiers associés
    """
    if task_id not in tts_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tts_tasks[task_id]
    temp_dir = f"temp_tts_{task_id}"
    
    # Suppression des fichiers temporaires
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Suppression de la tâche
    del tts_tasks[task_id]
    
    return {"message": f"Task {task_id} deleted successfully"}

# Tâche périodique pour nettoyer les anciennes tâches
@app.on_event("startup")
async def setup_periodic_cleanup():
    async def cleanup_old_tasks():
        while True:
            current_time = datetime.now()
            tasks_to_delete = []
            
            for task_id, task in tts_tasks.items():
                created_at = datetime.fromisoformat(task["created_at"])
                # Suppression des tâches de plus de 24 heures
                if (current_time - created_at).days >= 1:
                    tasks_to_delete.append(task_id)
            
            for task_id in tasks_to_delete:
                temp_dir = f"temp_tts_{task_id}"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                del tts_tasks[task_id]
            
            await asyncio.sleep(3600)  # Vérification toutes les heures
    
    asyncio.create_task(cleanup_old_tasks())

def run_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_api()