---
title: Free Embedding API
emoji: E
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Free Embedding API

FastAPI service for generating normalized sentence embeddings with sentence-transformers.

Default model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Override it with the `MODEL_NAME` environment variable if you rebuild the Qdrant index with a different model.
