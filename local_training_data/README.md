# Local training data

Place `.txt`, `.md`, and `.json` files in this folder to train the local transformer with next-token prediction. The trainer also reads the locally saved web ingestion export at `data/web_knowledge.json` when it exists, so newly ingested internet content can be included in future background training cycles.

Local weight training may be slow. Output quality depends on the amount of data, the model size, training time, and CPU/GPU hardware. This prototype does not create a ChatGPT-quality model.
