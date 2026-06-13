# Crop Suitability Map

Local map-based crop prediction app using Flask, Leaflet, and Ollama. No API keys are required.

## Features

- Click a point on the map to send latitude and longitude to the backend.
- Rejects sea, lake-like, island, and desert-like zones before prediction.
- Uses local Ollama for crop recommendations.
- Shows a clean result panel and compact popup with the top crop.

## Run

1. Install Python packages:

```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running locally.

3. Pull a model if needed:

```bash
ollama pull llama3.1:8b
```

4. Start the app:

```bash
python app.py
```

5. Open `http://127.0.0.1:5000`

## Optional environment variables

- `OLLAMA_MODEL` default: `llama3.1:8b`
- `OLLAMA_URL` default: `http://127.0.0.1:11434/api/generate`

## Notes

- The location filter is India-focused and intentionally conservative.
- If Ollama is unavailable, the app falls back to a local heuristic crop list so the UI still works.
