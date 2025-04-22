import torch
import logging

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mistral 7B v0.3 API",
    description="API для инференса модели Mistral-7B-v0.3, совместимый с официальным API Mistral AI",
    version="0.1.0"
)

# Загрузка модели и токенизатора
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    logger.info("Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3").to(device)
    logger.info("Модель и токенизатор успешно загружены")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {e}")
    raise


# Pydantic модели для запроса и ответа
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "mistral-7b-v0.3"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False
    random_seed: Optional[int] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    try:
        # Подготовка промпта в формате Mistral
        chat_history = []
        for message in request.messages:
            chat_history.append(f"<|{message.role}|>\n{message.content}</s>")
        
        prompt = "\n".join(chat_history) + "\n<|assistant|>\n"
        
        # Токенизация
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Генерация
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
        
        # Декодирование
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлечение только сгенерированного ответа (после последнего промпта)
        response_text = generated_text[len(prompt):]
        
        # Подсчет токенов
        prompt_tokens = len(inputs.input_ids[0])
        completion_tokens = len(outputs[0]) - prompt_tokens
        
        # Формирование ответа
        return ChatCompletionResponse(
            id="cmpl-" + str(hash(prompt)),
            created=int(torch.tensor(time.time())),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except Exception as e:
        logger.error(f"Ошибка при генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)