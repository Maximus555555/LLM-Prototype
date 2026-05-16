# Local Content Guide

To feed more local content into the conversation assistant, add plain text, Markdown, or JSON files to the `local_knowledge/` folder. The server reads files with .txt, .md, and .json extensions recursively, splits them into chunks, tokenizes the chunks, and ranks them by overlap with the user's message.

Good local content files are short notes, project documentation, FAQ entries, design constraints, personal notes intended for this app, or generated examples. The app should not scrape the internet. If users want more knowledge, they should manually place generated or local files in the folder.

When the retrieval system finds matching chunks, the assistant summarizes those chunks and lists local source paths. When it does not find enough evidence, it gives a fallback response that explains it does not know and suggests adding more local files.
