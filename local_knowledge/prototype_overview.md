# Local Conversation Prototype Overview

This project is a local educational LLM prototype. The browser app is designed to run with `npm start` and a dependency-free Node server. The interface must not contact OpenAI, hosted AI vendors, scraping services, or any external model endpoint. It does not require API keys.

The default conversation experience is intentionally local and transparent. It combines chat-style message history, short-term conversation memory, keyword retrieval over local files, and simple response rules. It should be described as a retrieval-and-rules assistant rather than a fully trained large language model.

Useful conversation topics include the local-only design, what files are indexed, how fallback responses work, how the optional untrained transformer can be invoked, and how to add more local text files.

The application keeps memory only for the current browser conversation unless the user saves an interface checkpoint. Checkpoints are saved as local JSON files under the checkpoints directory.
