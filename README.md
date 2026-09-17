# Asistente de obra · agente con Google ADK + RAG

Agente de IA que responde consultas técnicas de iluminación LED para proyectos de construcción
(arquitectos, constructoras, instaladores) buscando en la documentación real de un fabricante y
citando de dónde sale cada dato. Si la información no está en los documentos, lo dice en lugar
de inventarla.

**Demo:** _link de Cloud Run_ · **Autor:** Bruno Nuncibello · [portfolio](https://nuncibello.vercel.app)

## Stack

| Pieza | Uso |
|---|---|
| **Python 3.11 + FastAPI** | API asíncrona de punta a punta (`async/await`, `asyncio.gather`, semáforos, timeouts) |
| **Google ADK** | `LlmAgent` con una herramienta de búsqueda, `Runner` y sesiones para conversaciones de varios turnos |
| **Gemini 2.5 Flash** | modelo del agente |
| **gemini-embedding-001** | embeddings (768 dimensiones, `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`) |
| **RAG** | fragmentación con solapamiento, índice en memoria con similitud coseno (numpy) y caché por hash de contenido |
| **Docker + Cloud Run** | contenedor sin privilegios, escala a cero |
| **pytest** | pruebas sin red: embeddings falsos deterministas y agente simulado |

## Cómo funciona

```
POST /preguntar ──> ADK Runner ──> LlmAgent (Gemini)
                                     │  decide buscar, puede hacerlo varias veces
                                     ▼
                         buscar_documentacion(consulta)
                                     │
                        embedding de la consulta (async)
                                     │
                    similitud coseno contra el índice (numpy)
                                     │
                         top 4 fragmentos con su fuente
                                     ▼
                  respuesta final + fuentes + búsquedas + ms
```

1. **Al arrancar**, la API levanta enseguida y en paralelo indexa `docs/`: fragmenta cada documento
   (900 caracteres, 150 de solapamiento, cortando en espacios), genera los embeddings en lotes
   concurrentes con un semáforo y guarda el resultado en caché. Un reinicio no vuelve a pagar
   embeddings de lo que no cambió.
2. **En cada pregunta**, el agente de ADK decide qué buscar. La herramienta devuelve fragmentos con
   su fuente y la instrucción obliga a responder solo con esos datos y cerrar con las fuentes.
3. **Sesiones:** el campo `sesion` mantiene el contexto entre preguntas de la misma conversación.
4. **Documentos propios:** `POST /documentos` acepta PDF, Markdown o texto y los suma al índice en caliente.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | interfaz web para probarlo |
| `POST` | `/preguntar` | `{"pregunta": "...", "sesion": "opcional"}` → respuesta, fuentes, búsquedas, ms |
| `POST` | `/documentos` | subida de PDF, `.md` o `.txt` (máx. 5 MB) |
| `GET` | `/documentos` | documentos indexados y cantidad de fragmentos |
| `GET` | `/salud` | estado del servicio y del índice |

## Correrlo local

```bash
python -m venv .venv && .venv/Scripts/activate      # en Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt
python cargar_clave.py                              # guarda GOOGLE_API_KEY en .env
uvicorn app.main:app --reload --port 8080
pytest
```

## Docker y Cloud Run

```bash
docker build -t agente-obra .
docker run -p 8080:8080 -e GOOGLE_API_KEY=... agente-obra
```

En Cloud Run se despliega desde este repositorio (Cloud Build usa el `Dockerfile`) con la variable
`GOOGLE_API_KEY` cargada como secreto.

## Decisiones

- **Índice en memoria y no una base vectorial:** para decenas de documentos es más simple, rápido y
  gratis. El reemplazo natural a escala es pgvector o Vertex AI Vector Search, sin tocar el agente.
- **La herramienta devuelve la fuente con cada fragmento,** así las citas salen de datos reales y no
  de lo que el modelo recuerda.
- **Los fallos del modelo no tiran el servicio:** timeouts explícitos y errores 502/503/504 claros.
- **La documentación de ejemplo** son fichas técnicas públicas de [LED Premium](https://www.ledpremium.com.ar),
  plataforma que desarrollé y mantengo.
