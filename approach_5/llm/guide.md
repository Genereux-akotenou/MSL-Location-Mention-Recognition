# OLlama: Offline Large Language model Meta AI - Docker runner

1. Run docker
```bash
docker compose up -d
```

2. Run the model locally (llama3):
```bash
docker exec -it ollama ollama run llama3.1
```

You can now chat with the model on the terminal. If you have GPU, go to the official  [ollama docker image](https://hub.docker.com/r/ollama/ollama) for configuration.
3. Execute python file on your host terminal
```bash
python req_ollama.py
```

4. REST API

- Generate a response
```sh
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt":"Why is the sky blue?"
}'
```

- Chat with a model
```sh
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

5. Full API docs

[https://github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md)
