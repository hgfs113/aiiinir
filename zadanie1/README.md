## Индустриальные исследования в искусственном интеллекте

### Задание 1

#### Попов Дмитрий Николаевич, 617 группа

- Перед запуском необходимо положить модель LLama2-7B в `app/llama-2-7b-chat.Q4_K_M.gguf`. Это можно сделать например так: `wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O app/llama-2-7b-chat.Q4_K_M.gguf`

- `docker build . --network host -t llm_username:v1`
- `docker run -p 8080:8080 llm_username:v1`

*Пример запроса*:
    - `curl -X POST -H "Content-Type: application/json" -d '{"message": "Сколько мне будут стоить смски с оповещениями об операциях", "user_id": "1232"}' http://localhost:8080/message`

*Пример ответа*:
`{"message":"Сколько мне будут стоить смски с оповещениями об операциях","result":"59 рублей в месячном tariffa 59 рублей в месячном плате за услугу \"Оповещение об операциях\" для абонентов Тинькофф Мобайл.","user_id":"1232"}`
