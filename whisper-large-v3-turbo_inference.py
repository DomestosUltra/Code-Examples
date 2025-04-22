from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import os
import tempfile
from datetime import timedelta

app = FastAPI(title="Whisper Long-Form ASR API")

# Конфигурация для длинных аудио
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

# Инициализация модели с оптимизациями
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Специальный пайплайн для длинных записей
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
    chunk_length_s=60,          # Увеличиваем размер чанка
    stride_length_s=(4, 2),     # Перекрытие чанков для точности
    batch_size=4,               # Оптимальный размер батча для памяти
    return_timestamps="word",   # Детальные временные метки
    max_new_tokens=128,
    generate_kwargs={
        "task": "transcribe",
        "language": None,       # Автоопределение языка
        "without_timestamps": False
    }
)

@app.post("/transcribe")
async def transcribe_long_audio(
    audio_file: UploadFile,
    language: Optional[str] = None,
    precision: str = "balanced"
):
    """
    Оптимизированная обработка длинных аудио (30-60+ минут)
    
    Параметры precision:
    - "fast": ускоренная обработка (меньше проверок)
    - "balanced": оптимальный баланс (по умолчанию)
    - "accurate": максимальная точность
    """
    try:
        # Динамическая настройка под выбранный режим
        if precision == "fast":
            pipe.chunk_length_s = 90
            pipe.stride_length_s = (2, 1)
            pipe.generate_kwargs["compression_ratio_threshold"] = 2.4
        elif precision == "accurate":
            pipe.chunk_length_s = 30
            pipe.stride_length_s = (6, 3)
            pipe.generate_kwargs["compression_ratio_threshold"] = 2.2

        # Сохранение и конвертация аудио
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio_file.read()
            tmp.write(content)
            audio_path = tmp.name

        # Потоковая обработка больших файлов
        def process_chunks():
            for item in pipe(audio_path, chunk_length_s=pipe.chunk_length_s):
                yield item

        # Сбор и объединение результатов
        full_result = {"text": "", "chunks": []}
        for chunk_result in process_chunks():
            full_result["text"] += chunk_result["text"] + " "
            full_result["chunks"].extend([
                {
                    "text": seg["text"],
                    "start": seg["timestamp"][0],
                    "end": seg["timestamp"][1]
                }
                for seg in chunk_result["chunks"]
            ])

        # Постобработка
        full_result["text"] = full_result["text"].strip()
        os.unlink(audio_path)

        return {
            "text": full_result["text"],
            "duration": f"{timedelta(seconds=full_result['chunks'][-1]['end'])}",
            "segments": full_result["chunks"],
            "language": pipe.generate_kwargs["language"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)