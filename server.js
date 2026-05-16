const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const CHECKPOINT_DIR = path.join(ROOT, 'checkpoints');
const KNOWLEDGE_DIR = path.join(ROOT, 'local_knowledge');
const TRAINING_DATA_DIR = path.join(ROOT, 'local_training_data');
const DATA_DIR = path.join(ROOT, 'data');
const WEB_KNOWLEDGE_PATH = path.join(DATA_DIR, 'web_knowledge.json');
const PORT = Number(process.env.PORT || 3000);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml; charset=utf-8',
};

const KNOWLEDGE_EXTENSIONS = new Set(['.txt', '.md', '.json']);
const CHECKPOINT_EXTENSIONS = new Set(['.json', '.pt']);
const STOP_WORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'can', 'do', 'for', 'from', 'has', 'have', 'how', 'i', 'if', 'in',
  'is', 'it', 'me', 'my', 'of', 'on', 'or', 'our', 'so', 'that', 'the', 'their', 'this', 'to', 'was', 'we', 'what', 'when',
  'where', 'which', 'who', 'why', 'with', 'you', 'your', 'about', 'into', 'does', 'not', 'no', 'yes', 'than', 'then', 'there',
]);

fs.mkdirSync(CHECKPOINT_DIR, { recursive: true });
fs.mkdirSync(KNOWLEDGE_DIR, { recursive: true });
fs.mkdirSync(TRAINING_DATA_DIR, { recursive: true });
fs.mkdirSync(DATA_DIR, { recursive: true });

const WEB_SEARCH_MAX_RESULTS = 5;
const WEB_FETCH_TIMEOUT_MS = 12_000;
const BACKGROUND_INGEST_INTERVAL_MS = Number(process.env.BACKGROUND_INGEST_INTERVAL_MS || 90_000);
const BACKGROUND_INGESTION_ENABLED = process.env.DISABLE_BACKGROUND_INGESTION !== '1';
const BACKGROUND_TOPICS = [
  'computer science education programming tutorial',
  'open educational resources science technology',
  'software engineering best practices guide',
  'public domain science education article',
  'technical writing programming concepts explained',
  'mathematics education algorithm explanation',
];

const webKnowledgeState = {
  items: [],
  loadedAt: null,
  savedAt: null,
  background: {
    enabled: BACKGROUND_INGESTION_ENABLED,
    running: false,
    lastRunAt: null,
    lastSavedUrl: null,
    lastError: null,
    searchesRun: 0,
    itemsSaved: 0,
  },
};

let activeChatResponses = 0;
let backgroundTimer = null;

const trainingState = {
  process: null,
  startedAt: null,
  stoppedAt: null,
  exitCode: null,
  status: 'idle',
  latestTrainLoss: null,
  latestValidationLoss: null,
  latestStep: null,
  latestCheckpoint: null,
  console: '',
};

function appendTrainingConsole(message) {
  const text = String(message || '');
  if (!text) {
    return;
  }
  trainingState.console = `${trainingState.console}${text}`.slice(-20_000);
  const lossMatch = text.match(/step\s+(\d+):.*?train_loss=([0-9.]+)(?:\s+validation_loss=([0-9.]+))?/);
  if (lossMatch) {
    trainingState.latestStep = Number(lossMatch[1]);
    trainingState.latestTrainLoss = Number(lossMatch[2]);
    if (lossMatch[3] !== undefined) {
      trainingState.latestValidationLoss = Number(lossMatch[3]);
    }
  }
  const checkpointMatch = text.match(/latest_checkpoint=([^\s]+)/) || text.match(/saved checkpoint .* and ([^\s]+)/);
  if (checkpointMatch) {
    trainingState.latestCheckpoint = checkpointMatch[1];
  }
}

function listCheckpoints() {
  return fs
    .readdirSync(CHECKPOINT_DIR, { withFileTypes: true })
    .filter((entry) => entry.isFile() && CHECKPOINT_EXTENSIONS.has(path.extname(entry.name).toLowerCase()))
    .map((entry) => {
      const fullPath = path.join(CHECKPOINT_DIR, entry.name);
      const stats = fs.statSync(fullPath);
      return {
        name: entry.name,
        path: path.relative(ROOT, fullPath).split(path.sep).join('/'),
        type: entry.name.endsWith('.pt') ? 'model' : 'session',
        bytes: stats.size,
        modifiedAt: stats.mtime.toISOString(),
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}

function getTrainingStatus() {
  return {
    status: trainingState.status,
    running: Boolean(trainingState.process),
    startedAt: trainingState.startedAt,
    stoppedAt: trainingState.stoppedAt,
    exitCode: trainingState.exitCode,
    latestStep: trainingState.latestStep,
    latestTrainLoss: trainingState.latestTrainLoss,
    latestValidationLoss: trainingState.latestValidationLoss,
    latestCheckpoint: trainingState.latestCheckpoint,
    console: trainingState.console,
    dataDirectory: 'local_training_data/',
    checkpointDirectory: 'checkpoints/',
    hasLatestCheckpoint: fs.existsSync(path.join(CHECKPOINT_DIR, 'latest.pt')),
    checkpoints: listCheckpoints(),
  };
}

function sendJson(response, statusCode, payload) {
  const body = JSON.stringify(payload, null, 2);
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  response.end(body);
}

function sendText(response, statusCode, message) {
  response.writeHead(statusCode, { 'Content-Type': 'text/plain; charset=utf-8' });
  response.end(message);
}

function readRequestJson(request) {
  return new Promise((resolve, reject) => {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        request.destroy(new Error('Request body is too large.'));
      }
    });
    request.on('end', () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(new Error(`Invalid JSON: ${error.message}`));
      }
    });
    request.on('error', reject);
  });
}

function safeCheckpointPath(name) {
  const rawName = String(name || '').trim();
  if (!rawName) {
    throw new Error('Checkpoint name is required.');
  }
  const fileName = rawName.endsWith('.json') ? rawName : `${rawName}.json`;
  if (!/^[a-zA-Z0-9._-]+\.json$/.test(fileName)) {
    throw new Error('Use letters, numbers, dots, underscores, or hyphens for checkpoint names.');
  }
  const fullPath = path.join(CHECKPOINT_DIR, fileName);
  if (!fullPath.startsWith(CHECKPOINT_DIR + path.sep)) {
    throw new Error('Invalid checkpoint path.');
  }
  return fullPath;
}

function clampNumber(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(Math.max(number, min), max);
}

function tokenize(text) {
  return String(text || '')
    .toLowerCase()
    .match(/[a-z0-9][a-z0-9'-]*/g)
    ?.filter((token) => token.length > 1 && !STOP_WORDS.has(token)) || [];
}

function uniqueTokens(text) {
  return new Set(tokenize(text));
}


function stableHash(text) {
  let hash = 2166136261;
  for (const char of String(text || '')) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

function normalizeUrl(url) {
  try {
    const parsed = new URL(String(url || '').trim());
    parsed.hash = '';
    for (const key of ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']) {
      parsed.searchParams.delete(key);
    }
    return parsed.toString();
  } catch (_error) {
    return '';
  }
}

function loadWebKnowledge() {
  if (!fs.existsSync(WEB_KNOWLEDGE_PATH)) {
    webKnowledgeState.items = [];
    webKnowledgeState.loadedAt = new Date().toISOString();
    return webKnowledgeState;
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(WEB_KNOWLEDGE_PATH, 'utf-8'));
    webKnowledgeState.items = Array.isArray(parsed.items) ? parsed.items : [];
    webKnowledgeState.loadedAt = parsed.loadedAt || new Date().toISOString();
    webKnowledgeState.savedAt = parsed.savedAt || null;
  } catch (error) {
    webKnowledgeState.items = [];
    webKnowledgeState.loadedAt = new Date().toISOString();
    webKnowledgeState.background.lastError = `Could not load web knowledge: ${error.message}`;
  }
  return webKnowledgeState;
}

function saveWebKnowledge() {
  webKnowledgeState.savedAt = new Date().toISOString();
  fs.writeFileSync(WEB_KNOWLEDGE_PATH, `${JSON.stringify({
    version: 1,
    loadedAt: webKnowledgeState.loadedAt,
    savedAt: webKnowledgeState.savedAt,
    items: webKnowledgeState.items,
  }, null, 2)}\n`, 'utf-8');
}

function exportWebKnowledge() {
  return {
    version: 1,
    exportedAt: new Date().toISOString(),
    source: 'LLM-Prototype local web knowledge export',
    items: webKnowledgeState.items,
  };
}

function importWebKnowledge(payload) {
  const imported = Array.isArray(payload?.items) ? payload.items : Array.isArray(payload) ? payload : [];
  let added = 0;
  for (const rawItem of imported) {
    const item = {
      url: normalizeUrl(rawItem.url),
      title: String(rawItem.title || 'Imported knowledge').slice(0, 240),
      dateSaved: rawItem.dateSaved || new Date().toISOString(),
      summary: String(rawItem.summary || '').slice(0, 3000),
      keywords: Array.isArray(rawItem.keywords) ? rawItem.keywords.slice(0, 20).map(String) : extractKeywords(`${rawItem.title || ''} ${rawItem.summary || ''}`),
      text: String(rawItem.text || rawItem.content || '').slice(0, 12_000),
      sourceType: rawItem.sourceType || 'import',
    };
    if (!item.url && !item.summary && !item.text) {
      continue;
    }
    if (storeWebKnowledgeItem(item).saved) {
      added += 1;
    }
  }
  saveWebKnowledge();
  return { added, total: webKnowledgeState.items.length };
}

function keywordSimilarity(a, b) {
  const left = new Set(a || []);
  const right = new Set(b || []);
  if (!left.size || !right.size) {
    return 0;
  }
  let overlap = 0;
  for (const token of left) {
    if (right.has(token)) {
      overlap += 1;
    }
  }
  return overlap / (left.size + right.size - overlap);
}

function extractKeywords(text, limit = 12) {
  const counts = new Map();
  for (const token of tokenize(text)) {
    if (token.length < 3) {
      continue;
    }
    counts.set(token, (counts.get(token) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([token]) => token);
}

function summarizeText(text, title = '') {
  const cleaned = String(text || '')
    .replace(/\s+/g, ' ')
    .replace(/[\u0000-\u001f]+/g, ' ')
    .trim();
  if (!cleaned) {
    return '';
  }
  const sentences = cleaned.match(/[^.!?]+[.!?]+|[^.!?]+$/g)?.map((sentence) => sentence.trim()).filter((sentence) => sentence.length > 40) || [];
  const keywords = extractKeywords(`${title} ${cleaned}`, 16);
  const scored = sentences.slice(0, 80).map((sentence, index) => {
    const tokens = uniqueTokens(sentence);
    let score = Math.max(0, 8 - index * 0.12);
    for (const keyword of keywords) {
      if (tokens.has(keyword)) {
        score += 2;
      }
    }
    return { sentence, score };
  });
  const selected = scored
    .sort((a, b) => b.score - a.score)
    .slice(0, 4)
    .sort((a, b) => cleaned.indexOf(a.sentence) - cleaned.indexOf(b.sentence))
    .map((item) => item.sentence);
  return (selected.length ? selected.join(' ') : cleaned.slice(0, 700)).slice(0, 1400);
}

function stripHtmlToText(html) {
  return String(html || '')
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<nav[\s\S]*?<\/nav>/gi, ' ')
    .replace(/<footer[\s\S]*?<\/footer>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractHtmlTitle(html, fallback = '') {
  const title = String(html || '').match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] || fallback;
  return stripHtmlToText(title).slice(0, 240) || fallback || 'Untitled webpage';
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs || WEB_FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'LLM-Prototype/0.1 local educational knowledge retriever',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,text/plain;q=0.7,*/*;q=0.5',
        ...(options.headers || {}),
      },
    });
  } finally {
    clearTimeout(timeout);
  }
}

async function searchDuckDuckGo(query, limit = WEB_SEARCH_MAX_RESULTS) {
  const endpoint = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
  const response = await fetchWithTimeout(endpoint);
  if (!response.ok) {
    throw new Error(`DuckDuckGo search failed with HTTP ${response.status}`);
  }
  const html = await response.text();
  const results = [];
  const resultRegex = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/gi;
  let match;
  while ((match = resultRegex.exec(html)) && results.length < limit) {
    let url = match[1].replace(/&amp;/g, '&');
    try {
      const parsed = new URL(url, 'https://duckduckgo.com');
      if (parsed.searchParams.has('uddg')) {
        url = parsed.searchParams.get('uddg');
      }
    } catch (_error) {
      continue;
    }
    const normalized = normalizeUrl(url);
    if (!normalized || !/^https?:\/\//i.test(normalized)) {
      continue;
    }
    results.push({
      title: stripHtmlToText(match[2]),
      url: normalized,
      snippet: stripHtmlToText(match[3]),
      engine: 'duckduckgo-html',
    });
  }
  return results;
}

async function searchWikipedia(query, limit = WEB_SEARCH_MAX_RESULTS) {
  const endpoint = `https://en.wikipedia.org/w/api.php?action=opensearch&namespace=0&limit=${limit}&format=json&search=${encodeURIComponent(query)}`;
  const response = await fetchWithTimeout(endpoint, { headers: { Accept: 'application/json' } });
  if (!response.ok) {
    throw new Error(`Wikipedia search failed with HTTP ${response.status}`);
  }
  const [term, titles, descriptions, urls] = await response.json();
  return (titles || []).map((title, index) => ({
    title,
    url: normalizeUrl(urls[index]),
    snippet: descriptions[index] || `Wikipedia result for ${term}.`,
    engine: 'wikipedia-opensearch',
  })).filter((result) => result.url).slice(0, limit);
}

async function webSearch(query, limit = WEB_SEARCH_MAX_RESULTS) {
  const normalizedQuery = String(query || '').trim();
  if (!normalizedQuery) {
    return [];
  }
  try {
    const results = await searchDuckDuckGo(normalizedQuery, limit);
    if (results.length) {
      return results;
    }
  } catch (error) {
    webKnowledgeState.background.lastError = error.message;
  }
  try {
    return await searchWikipedia(normalizedQuery, limit);
  } catch (error) {
    webKnowledgeState.background.lastError = error.message;
    return [];
  }
}

async function retrieveWebpageText(url) {
  const normalizedUrl = normalizeUrl(url);
  if (!normalizedUrl) {
    throw new Error('A valid http(s) URL is required.');
  }
  const response = await fetchWithTimeout(normalizedUrl);
  if (!response.ok) {
    throw new Error(`Could not retrieve webpage: HTTP ${response.status}`);
  }
  const contentType = response.headers.get('content-type') || '';
  const body = await response.text();
  const title = contentType.includes('html') ? extractHtmlTitle(body, normalizedUrl) : normalizedUrl;
  const text = contentType.includes('html') ? stripHtmlToText(body) : body.replace(/\s+/g, ' ').trim();
  return { url: normalizedUrl, title, text: text.slice(0, 60_000), contentType };
}

function storeWebKnowledgeItem(rawItem) {
  const text = String(rawItem.text || rawItem.summary || '').trim();
  const summary = String(rawItem.summary || summarizeText(text, rawItem.title)).trim();
  const keywords = Array.isArray(rawItem.keywords) && rawItem.keywords.length ? rawItem.keywords.slice(0, 20) : extractKeywords(`${rawItem.title || ''} ${summary} ${text}`, 12);
  const normalizedUrl = normalizeUrl(rawItem.url) || `local-import:${stableHash(`${rawItem.title || ''}${summary}${text}`)}`;
  const contentHash = stableHash(`${normalizedUrl}\n${summary}\n${text.slice(0, 5000)}`);
  const duplicate = webKnowledgeState.items.find((item) => item.url === normalizedUrl || item.contentHash === contentHash || keywordSimilarity(item.keywords, keywords) > 0.92 && item.title === rawItem.title);
  if (duplicate) {
    return { saved: false, reason: 'duplicate', item: duplicate };
  }
  const item = {
    id: contentHash,
    url: normalizedUrl,
    title: String(rawItem.title || 'Untitled webpage').slice(0, 240),
    dateSaved: rawItem.dateSaved || new Date().toISOString(),
    summary: summary.slice(0, 1800),
    keywords,
    text: text.slice(0, 12_000),
    sourceType: rawItem.sourceType || 'web',
    contentHash,
  };
  webKnowledgeState.items.unshift(item);
  webKnowledgeState.items = webKnowledgeState.items.slice(0, 500);
  saveWebKnowledge();
  webKnowledgeState.background.itemsSaved += 1;
  webKnowledgeState.background.lastSavedUrl = item.url;
  return { saved: true, item };
}

async function retrieveSummarizeAndStore(url, fallback = {}) {
  const page = await retrieveWebpageText(url);
  const summary = summarizeText(page.text, page.title);
  const keywords = extractKeywords(`${page.title} ${summary} ${page.text}`, 12);
  return storeWebKnowledgeItem({
    url: page.url,
    title: page.title || fallback.title,
    text: page.text,
    summary,
    keywords,
    sourceType: fallback.sourceType || 'webpage',
  });
}

function searchStoredWebKnowledge(query, limit = 5) {
  const queryTokens = uniqueTokens(query);
  if (!queryTokens.size) {
    return [];
  }
  return webKnowledgeState.items
    .map((item) => {
      const itemTokens = uniqueTokens(`${item.title} ${item.summary} ${(item.keywords || []).join(' ')} ${item.text || ''}`);
      let overlap = 0;
      for (const token of queryTokens) {
        if (itemTokens.has(token)) {
          overlap += 1;
        }
      }
      const score = overlap + overlap / Math.max(1, Math.sqrt(itemTokens.size));
      return {
        source: item.url,
        title: item.title,
        chunk: 1,
        content: item.summary || String(item.text || '').slice(0, 1200),
        score: Number(score.toFixed(3)),
        dateSaved: item.dateSaved,
        keywords: item.keywords || [],
        type: 'web-knowledge',
      };
    })
    .filter((match) => match.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function extractSearchQuery(message) {
  const text = String(message || '').trim();
  const explicit = /\b(search|look up|browse|internet|web|online|google|duckduckgo|find current|latest)\b/i.test(text);
  if (!explicit) {
    return '';
  }
  return text
    .replace(/^\s*(please\s+)?(search|look up|browse|find)\s+(the\s+)?(internet|web|online)?\s*(for|about)?\s*/i, '')
    .replace(/\b(on the internet|online|on the web)\b/ig, '')
    .trim() || text;
}

async function answerWithInternetSearch(message, memory) {
  const query = extractSearchQuery(message);
  const results = await webSearch(query, WEB_SEARCH_MAX_RESULTS);
  const stored = [];
  for (const result of results.slice(0, 3)) {
    try {
      const saved = await retrieveSummarizeAndStore(result.url, { title: result.title, sourceType: 'web-search' });
      stored.push(saved.item);
    } catch (_error) {
      const saved = storeWebKnowledgeItem({
        url: result.url,
        title: result.title,
        text: result.snippet,
        summary: result.snippet,
        keywords: extractKeywords(`${result.title} ${result.snippet}`),
        sourceType: 'search-result-snippet',
      });
      stored.push(saved.item);
    }
  }
  const retrieved = searchStoredWebKnowledge(`${query} ${message}`, 5);
  const snippets = retrieved.slice(0, 4).map((match, index) => `${index + 1}. ${match.title || match.source}: ${match.content}`);
  return {
    answer: [
      `I searched the web for “${query}” using no-key, free sources, retrieved/summarized available pages, and saved useful knowledge locally in data/web_knowledge.json.`,
      snippets.length ? snippets.join('\n\n') : 'I found search results but could not extract enough readable text to summarize confidently.',
      `Sources: ${retrieved.map((match) => match.source).join(', ') || results.map((result) => result.url).join(', ') || 'none'}.`,
      'This is retrieval-based local memory, not model retraining.',
    ].join('\n\n'),
    memory,
    sources: retrieved,
    documents: loadKnowledgeBase().documents,
    webResults: results,
    saved: stored.length,
  };
}

async function runBackgroundIngestion() {
  if (!BACKGROUND_INGESTION_ENABLED || webKnowledgeState.background.running || activeChatResponses > 0) {
    return;
  }
  webKnowledgeState.background.running = true;
  webKnowledgeState.background.lastRunAt = new Date().toISOString();
  webKnowledgeState.background.lastError = null;
  try {
    const topic = BACKGROUND_TOPICS[webKnowledgeState.background.searchesRun % BACKGROUND_TOPICS.length];
    webKnowledgeState.background.searchesRun += 1;
    const results = await webSearch(topic, 3);
    const candidate = results.find((result) => !webKnowledgeState.items.some((item) => item.url === result.url));
    if (candidate) {
      await retrieveSummarizeAndStore(candidate.url, { title: candidate.title, sourceType: 'background-web' });
    }
  } catch (error) {
    webKnowledgeState.background.lastError = error.message;
  } finally {
    webKnowledgeState.background.running = false;
  }
}

function startBackgroundIngestion() {
  if (!BACKGROUND_INGESTION_ENABLED || backgroundTimer) {
    return;
  }
  backgroundTimer = setInterval(() => {
    runBackgroundIngestion().catch((error) => {
      webKnowledgeState.background.lastError = error.message;
      webKnowledgeState.background.running = false;
    });
  }, BACKGROUND_INGEST_INTERVAL_MS);
  backgroundTimer.unref?.();
  setTimeout(() => runBackgroundIngestion().catch(() => {}), 5_000).unref?.();
}

function walkKnowledgeFiles(directory = KNOWLEDGE_DIR) {
  if (!fs.existsSync(directory)) {
    return [];
  }
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkKnowledgeFiles(fullPath));
    } else if (entry.isFile() && KNOWLEDGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      files.push(fullPath);
    }
  }
  return files.sort();
}

function toRelativeKnowledgePath(filePath) {
  return path.relative(ROOT, filePath).split(path.sep).join('/');
}

function chunkText(text, source) {
  const normalized = String(text || '')
    .replace(/\r\n/g, '\n')
    .replace(/[\t ]+/g, ' ')
    .trim();
  const paragraphs = normalized.split(/\n{2,}/).map((part) => part.trim()).filter(Boolean);
  const chunks = [];
  let current = '';
  for (const paragraph of paragraphs.length ? paragraphs : [normalized]) {
    const next = current ? `${current}\n\n${paragraph}` : paragraph;
    if (next.length > 900 && current) {
      chunks.push(current);
      current = paragraph;
    } else {
      current = next;
    }
  }
  if (current) {
    chunks.push(current);
  }
  return chunks.map((content, index) => ({
    source,
    chunk: index + 1,
    content: content.length > 1400 ? `${content.slice(0, 1400)}…` : content,
    tokens: uniqueTokens(content),
  }));
}

function loadKnowledgeBase() {
  const files = walkKnowledgeFiles();
  const documents = [];
  const chunks = [];
  for (const filePath of files) {
    const source = toRelativeKnowledgePath(filePath);
    const content = fs.readFileSync(filePath, 'utf-8');
    const fileChunks = chunkText(content, source);
    documents.push({ source, characters: content.length, chunks: fileChunks.length, type: 'local-file' });
    chunks.push(...fileChunks);
  }

  for (const item of webKnowledgeState.items) {
    const source = item.url;
    const content = [item.title, item.summary, item.text].filter(Boolean).join('\n\n');
    const itemChunks = chunkText(content, source).map((chunk) => ({
      ...chunk,
      title: item.title,
      dateSaved: item.dateSaved,
      keywords: item.keywords || [],
      type: 'web-knowledge',
    }));
    documents.push({
      source,
      title: item.title,
      characters: content.length,
      chunks: itemChunks.length,
      dateSaved: item.dateSaved,
      keywords: item.keywords || [],
      type: 'web-knowledge',
    });
    chunks.push(...itemChunks);
  }
  return { documents, chunks, loadedAt: new Date().toISOString() };
}

loadWebKnowledge();

function searchKnowledge(query, limit = 5) {
  const knowledgeBase = loadKnowledgeBase();
  const queryTokens = uniqueTokens(query);
  if (!queryTokens.size) {
    return { ...knowledgeBase, matches: [] };
  }

  const matches = knowledgeBase.chunks
    .map((chunk) => {
      let overlap = 0;
      for (const token of queryTokens) {
        if (chunk.tokens.has(token)) {
          overlap += 1;
        }
      }
      const density = overlap / Math.max(1, Math.sqrt(chunk.tokens.size));
      return { source: chunk.source, chunk: chunk.chunk, content: chunk.content, score: overlap + density };
    })
    .filter((match) => match.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((match) => ({ ...match, score: Number(match.score.toFixed(3)) }));

  return { ...knowledgeBase, matches };
}

function summarizeMemory(history) {
  const recent = Array.isArray(history) ? history.slice(-8) : [];
  const userTurns = recent.filter((message) => message && message.role === 'user').map((message) => String(message.content || '').trim()).filter(Boolean);
  const assistantTurns = recent.filter((message) => message && message.role === 'assistant').map((message) => String(message.content || '').trim()).filter(Boolean);
  return {
    turns: recent.length,
    recentUserTopics: userTurns.slice(-3),
    lastUserMessage: userTurns.at(-1) || '',
    lastAssistantNote: assistantTurns.at(-1) || '',
  };
}

function formatNumber(value) {
  if (!Number.isFinite(value)) {
    return 'undefined';
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return String(Number(value.toFixed(10))).replace(/\.0+$/, '');
}

function extractMathExpression(message) {
  const normalized = String(message || '')
    .toLowerCase()
    .replace(/[×x]/g, '*')
    .replace(/[÷]/g, '/')
    .replace(/plus/g, '+')
    .replace(/minus/g, '-')
    .replace(/times|multiplied by/g, '*')
    .replace(/divided by|over/g, '/')
    .replace(/to the power of/g, '^')
    .replace(/squared/g, '^2')
    .replace(/cubed/g, '^3');
  const candidate = normalized.match(/[-+*/^%(). 0-9]+/g)?.map((part) => part.trim()).filter((part) => /\d/.test(part) && /[-+*/^%]/.test(part)).sort((a, b) => b.length - a.length)[0];
  return candidate || '';
}

function parseMathExpression(expression) {
  const tokens = String(expression || '').match(/\d+(?:\.\d+)?|[()+\-*/^%]/g) || [];
  let index = 0;

  function peek() {
    return tokens[index];
  }

  function consume(expected) {
    const token = tokens[index];
    if (expected && token !== expected) {
      throw new Error(`Expected ${expected}.`);
    }
    index += 1;
    return token;
  }

  function parsePrimary() {
    const token = peek();
    if (token === '+') {
      consume('+');
      return parsePrimary();
    }
    if (token === '-') {
      consume('-');
      return -parsePrimary();
    }
    if (token === '(') {
      consume('(');
      const value = parseAddSubtract();
      consume(')');
      return value;
    }
    if (!token || !/^\d/.test(token)) {
      throw new Error('Expected a number.');
    }
    consume();
    let value = Number(token);
    while (peek() === '%') {
      consume('%');
      value /= 100;
    }
    return value;
  }

  function parsePower() {
    let value = parsePrimary();
    if (peek() === '^') {
      consume('^');
      value = value ** parsePower();
    }
    return value;
  }

  function parseMultiplyDivide() {
    let value = parsePower();
    while (peek() === '*' || peek() === '/') {
      const operator = consume();
      const right = parsePower();
      value = operator === '*' ? value * right : value / right;
    }
    return value;
  }

  function parseAddSubtract() {
    let value = parseMultiplyDivide();
    while (peek() === '+' || peek() === '-') {
      const operator = consume();
      const right = parseMultiplyDivide();
      value = operator === '+' ? value + right : value - right;
    }
    return value;
  }

  const value = parseAddSubtract();
  if (index !== tokens.length) {
    throw new Error('Unexpected extra input.');
  }
  return value;
}

function tryAnswerMath(message) {
  const expression = extractMathExpression(message);
  if (!expression) {
    return '';
  }
  try {
    const value = parseMathExpression(expression);
    if (!Number.isFinite(value)) {
      return 'That math expression does not have a finite answer.';
    }
    return `${expression.replace(/\s+/g, ' ')} = ${formatNumber(value)}`;
  } catch (_error) {
    return '';
  }
}

function answerBasicConversation(message, memory, documentCount) {
  const lowerMessage = String(message || '').toLowerCase();
  const lastTopic = memory.recentUserTopics.find((topic) => topic && topic !== message);

  if (/\b(how are you|how's it going|how are things)\b/.test(lowerMessage)) {
    return 'I am doing well and ready to help. I can chat, solve basic math, remember the current conversation, and search the local knowledge files when your question needs project-specific details.';
  }

  if (/\b(thanks|thank you|appreciate it)\b/.test(lowerMessage)) {
    return 'You are welcome! Ask me another question whenever you are ready.';
  }

  if (/\b(what can you do|help|capabilities|skills)\b/.test(lowerMessage)) {
    return [
      'I can handle basic conversation, arithmetic, short explanations, and simple follow-up context in this chat.',
      `I can also search ${documentCount} local knowledge file(s) for project-specific answers.`,
      'For facts outside those files, explicitly ask me to search the internet and I can retrieve, summarize, and save webpages locally.',
    ].join(' ');
  }

  if (/\b(who are you|what are you|your name)\b/.test(lowerMessage)) {
    return 'I am the Local AI in this prototype: a small local chat assistant built from rules, conversation memory, arithmetic handling, and local-file retrieval.';
  }

  if (/\b(remember|what did i ask|previous|last thing)\b/.test(lowerMessage)) {
    return lastTopic
      ? `The recent topic I have from this chat is: “${lastTopic}”.`
      : 'I do not have an earlier user topic in this current chat yet.';
  }

  return '';
}

function createFallback(message, memory) {
  const topic = tokenize(message).slice(0, 6).join(', ') || 'that topic';
  const followUp = memory.recentUserTopics.length > 1
    ? 'If this is a follow-up, add one more detail and I will connect it to the current chat context.'
    : 'You can ask conversational questions, basic math, questions about stored knowledge, or explicitly ask me to search the internet.';
  return [
    `I am not fully sure how to answer ${topic} from the local knowledge I have right now.`,
    'I can still help with basic conversation, arithmetic, and project questions grounded in local files.',
    followUp,
  ].join(' ');
}

function createLocalChatResponse({ message, history }) {
  const normalizedMessage = String(message || '').trim();
  const memory = summarizeMemory(history);
  const search = searchKnowledge(normalizedMessage, 4);
  const lowerMessage = normalizedMessage.toLowerCase();

  if (!normalizedMessage) {
    return {
      answer: 'Type a message and I will search the local knowledge folder before responding.',
      memory,
      sources: [],
      documents: search.documents,
    };
  }

  if (/\b(hello|hi|hey|start)\b/.test(lowerMessage)) {
    return {
      answer: [
        'Hi! I am ready to chat locally.',
        'You can ask me everyday questions, basic arithmetic like “20 + 5”, follow-up questions in this conversation, or questions about the local knowledge files.',
        `Right now I loaded ${search.documents.length} local knowledge file(s) from local_knowledge/.`,
      ].join('\n\n'),
      memory,
      sources: [],
      documents: search.documents,
    };
  }

  const mathAnswer = tryAnswerMath(normalizedMessage);
  if (mathAnswer) {
    return {
      answer: mathAnswer,
      memory,
      sources: [],
      documents: search.documents,
    };
  }

  const basicAnswer = answerBasicConversation(normalizedMessage, memory, search.documents.length);
  if (basicAnswer) {
    return {
      answer: basicAnswer,
      memory,
      sources: [],
      documents: search.documents,
    };
  }

  if (!search.matches.length || search.matches[0].score < 1.2) {
    return {
      answer: createFallback(normalizedMessage, memory),
      memory,
      sources: [],
      documents: search.documents,
    };
  }

  const snippets = search.matches.slice(0, 3).map((match, index) => {
    const compact = match.content.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
    return `${index + 1}. ${compact}`;
  });
  const sourceList = search.matches.map((match) => `${match.source}#chunk-${match.chunk}`);
  const memoryNote = memory.recentUserTopics.length
    ? `\n\nConversation memory: I am keeping this turn connected to your recent topic(s): ${memory.recentUserTopics.join(' | ')}.`
    : '';

  return {
    answer: [
      'I searched the local knowledge folder and found relevant material. Based only on those local files:',
      snippets.join('\n\n'),
      `\nSources searched for this reply: ${sourceList.join(', ')}.`,
      'If you want more detail, ask a follow-up and I will reuse the chat history plus the local retrieval results.',
    ].join('\n\n') + memoryNote,
    memory,
    sources: search.matches,
    documents: search.documents,
  };
}



function extractUserProvidedKnowledge(message) {
  const text = String(message || '').trim();
  const match = text.match(/^(?:please\s+)?(?:remember|save|learn|store)\s+(?:this|that|the following)?[:\s]+([\s\S]{20,})$/i);
  return match ? match[1].trim() : '';
}

async function createLocalChatResponseAsync({ message, history }) {
  const normalizedMessage = String(message || '').trim();
  const memory = summarizeMemory(history);
  const userKnowledge = extractUserProvidedKnowledge(normalizedMessage);
  if (userKnowledge) {
    const saved = storeWebKnowledgeItem({
      url: `user-provided:${stableHash(userKnowledge)}`,
      title: 'User-provided knowledge',
      text: userKnowledge,
      summary: summarizeText(userKnowledge, 'User-provided knowledge') || userKnowledge,
      keywords: extractKeywords(userKnowledge),
      sourceType: 'user-provided',
    });
    const search = searchKnowledge(userKnowledge, 4);
    return {
      answer: `${saved.saved ? 'Saved' : 'Already had'} that user-provided knowledge in the persistent local knowledge database. I will use retrieval over saved knowledge in future conversations; this is persistent memory, not model retraining.`,
      memory,
      sources: search.matches,
      documents: search.documents,
    };
  }
  const query = extractSearchQuery(normalizedMessage);
  if (query) {
    activeChatResponses += 1;
    try {
      return await answerWithInternetSearch(normalizedMessage, memory);
    } finally {
      activeChatResponses = Math.max(0, activeChatResponses - 1);
    }
  }
  return createLocalChatResponse({ message, history });
}

function demoGenerate(prompt, settings) {
  const maxTokens = clampNumber(settings.maxTokens, 80, 1, 512);
  const temperature = clampNumber(settings.temperature, 0.8, 0.05, 2);
  const normalizedPrompt = String(prompt || '').trim() || 'Empty prompt';
  const words = normalizedPrompt.split(/\s+/).filter(Boolean);
  const seed = [...normalizedPrompt].reduce((acc, char) => (acc + char.charCodeAt(0)) % 9973, 17);
  const fragments = [
    'local interface response',
    'untrained prototype path',
    'no external service was contacted',
    'settings are ready for checkpointing',
    'wire this panel to trained local weights later',
    'generation controls are being passed through',
  ];
  const generated = [];
  const targetWords = Math.max(10, Math.min(maxTokens, 120));
  for (let index = 0; index < targetWords; index += 1) {
    const promptWord = words.length ? words[(seed + index) % words.length] : 'prototype';
    const fragment = fragments[(seed + index * Math.max(1, Math.round(temperature * 10))) % fragments.length];
    generated.push(index % 3 === 0 ? promptWord : fragment.split(' ')[index % fragment.split(' ').length]);
  }
  return [
    `Prompt: ${normalizedPrompt}`,
    '',
    generated.join(' '),
    '',
    '[Interface demo mode: deterministic local placeholder output. Choose Local Python runtime to call the bundled untrained transformer when PyTorch is installed.]',
  ].join('\n');
}

function runPythonGeneration({ prompt, temperature, maxTokens, topK, checkpointPath, tokenizerPath }) {
  return new Promise((resolve) => {
    const args = [
      '-m',
      'llm_prototype.inference',
      '--config',
      'configs/tiny.json',
      '--prompt',
      String(prompt || ''),
      '--temperature',
      String(clampNumber(temperature, 0.8, 0.05, 2)),
      '--max-new-tokens',
      String(Math.round(clampNumber(maxTokens, 80, 1, 256))),
      '--top-k',
      String(Math.round(clampNumber(topK, 50, 1, 259))),
      '--device',
      'cpu',
    ];
    if (checkpointPath) {
      args.splice(2, 0, '--checkpoint', String(checkpointPath));
    }
    if (tokenizerPath) {
      args.push('--tokenizer', String(tokenizerPath));
    }

    const child = spawn(process.env.PYTHON || 'python3', args, {
      cwd: ROOT,
      env: { ...process.env, PYTHONPATH: path.join(ROOT, 'src') },
    });

    let stdout = '';
    let stderr = '';
    const timeout = setTimeout(() => {
      child.kill('SIGTERM');
      stderr += '\nGeneration timed out after 30 seconds.';
    }, 30_000);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', (error) => {
      clearTimeout(timeout);
      resolve({ ok: false, output: '', console: error.message });
    });
    child.on('close', (code) => {
      clearTimeout(timeout);
      resolve({ ok: code === 0, output: stdout.trim(), console: stderr.trim(), exitCode: code });
    });
  });
}


function clampTrainingInteger(value, fallback, min, max) {
  return Math.round(clampNumber(value, fallback, min, max));
}

function startTrainingProcess(options = {}) {
  if (trainingState.process) {
    throw new Error('Training is already running. Stop it before starting another run.');
  }

  const args = [
    'scripts/train.py',
    '--config', 'configs/tiny.json',
    '--data-dir', 'local_training_data',
    '--output-dir', 'checkpoints',
    '--batch-size', String(clampTrainingInteger(options.batchSize, 8, 1, 64)),
    '--learning-rate', String(clampNumber(options.learningRate, 0.0003, 0.000001, 0.1)),
    '--max-steps', String(clampTrainingInteger(options.maxSteps, 200, 1, 100000)),
    '--eval-interval', String(clampTrainingInteger(options.evalInterval, 25, 1, 10000)),
    '--checkpoint-interval', String(clampTrainingInteger(options.checkpointInterval, 50, 1, 10000)),
    '--device', String(options.device || 'auto'),
  ];
  if (options.resume) {
    const resumePath = path.join(ROOT, String(options.resume));
    if (!resumePath.startsWith(ROOT + path.sep) || !fs.existsSync(resumePath)) {
      throw new Error('Resume checkpoint must be an existing file in this repository.');
    }
    args.push('--resume', String(options.resume));
  }

  trainingState.startedAt = new Date().toISOString();
  trainingState.stoppedAt = null;
  trainingState.exitCode = null;
  trainingState.status = 'running';
  trainingState.latestTrainLoss = null;
  trainingState.latestValidationLoss = null;
  trainingState.latestStep = null;
  trainingState.latestCheckpoint = null;
  trainingState.console = `Starting local training: python3 ${args.join(' ')}\n`;

  const child = spawn(process.env.PYTHON || 'python3', args, {
    cwd: ROOT,
    env: { ...process.env, PYTHONPATH: path.join(ROOT, 'src') },
  });
  trainingState.process = child;

  child.stdout.on('data', (chunk) => appendTrainingConsole(chunk.toString()));
  child.stderr.on('data', (chunk) => appendTrainingConsole(chunk.toString()));
  child.on('error', (error) => {
    appendTrainingConsole(`\nTraining process error: ${error.message}\n`);
    trainingState.status = 'error';
    trainingState.stoppedAt = new Date().toISOString();
    trainingState.process = null;
  });
  child.on('close', (code, signal) => {
    trainingState.exitCode = code;
    trainingState.stoppedAt = new Date().toISOString();
    trainingState.process = null;
    if (signal) {
      trainingState.status = 'stopped';
      appendTrainingConsole(`\nTraining stopped by signal ${signal}.\n`);
    } else if (code === 0) {
      trainingState.status = 'completed';
    } else {
      trainingState.status = 'error';
    }
  });

  return getTrainingStatus();
}

function stopTrainingProcess() {
  if (!trainingState.process) {
    return getTrainingStatus();
  }
  appendTrainingConsole('\nStop requested from browser interface.\n');
  trainingState.status = 'stopping';
  trainingState.process.kill('SIGTERM');
  return getTrainingStatus();
}

async function handleApi(request, response, url) {
  if (request.method === 'GET' && url.pathname === '/api/status') {
    const knowledgeBase = loadKnowledgeBase();
    sendJson(response, 200, {
      name: 'Local Conversation Prototype',
      externalAiServices: false,
      apiKeysRequired: false,
      checkpointSupport: true,
      checkpointDirectory: 'checkpoints/',
      trainingDataDirectory: 'local_training_data/',
      knowledgeDirectory: 'local_knowledge/',
      webKnowledgePath: 'data/web_knowledge.json',
      localKnowledgeFiles: knowledgeBase.documents.length,
      webKnowledgeItems: webKnowledgeState.items.length,
      internetAccess: true,
      backgroundIngestion: webKnowledgeState.background,
      defaultConfig: 'configs/tiny.json',
      runtimes: ['local-chat', 'demo', 'local-python'],
      trainingSupport: true,
      hasLatestModelCheckpoint: fs.existsSync(path.join(CHECKPOINT_DIR, 'latest.pt')), 
    });
    return;
  }

  if (request.method === 'GET' && url.pathname === '/api/knowledge') {
    const knowledgeBase = loadKnowledgeBase();
    sendJson(response, 200, {
      directory: 'local_knowledge/',
      webKnowledgePath: 'data/web_knowledge.json',
      loadedAt: knowledgeBase.loadedAt,
      documents: knowledgeBase.documents,
      chunkCount: knowledgeBase.chunks.length,
      webKnowledge: {
        items: webKnowledgeState.items.length,
        savedAt: webKnowledgeState.savedAt,
        background: webKnowledgeState.background,
      },
    });
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/knowledge/search') {
    const payload = await readRequestJson(request);
    const matches = searchKnowledge(payload.query || '', clampTrainingInteger(payload.limit, 8, 1, 25)).matches;
    sendJson(response, 200, { matches, count: matches.length });
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/knowledge/clear') {
    webKnowledgeState.items = [];
    saveWebKnowledge();
    sendJson(response, 200, { cleared: true, total: 0 });
    return;
  }

  if (request.method === 'GET' && url.pathname === '/api/knowledge/export') {
    sendJson(response, 200, exportWebKnowledge());
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/knowledge/import') {
    const payload = await readRequestJson(request);
    sendJson(response, 200, importWebKnowledge(payload));
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/internet/search') {
    const payload = await readRequestJson(request);
    const query = String(payload.query || '').trim();
    const results = await webSearch(query, clampTrainingInteger(payload.limit, WEB_SEARCH_MAX_RESULTS, 1, 10));
    sendJson(response, 200, { query, results });
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/internet/retrieve') {
    const payload = await readRequestJson(request);
    const stored = await retrieveSummarizeAndStore(payload.url, { sourceType: 'manual-retrieval' });
    sendJson(response, 200, stored);
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/chat') {
    const payload = await readRequestJson(request);
    const result = await createLocalChatResponseAsync({ message: payload.message, history: payload.history });
    sendJson(response, 200, {
      runtime: 'local-chat',
      ...result,
      console: 'Answered with local conversation memory, local/web knowledge retrieval, and optional no-key internet search. No external AI service was contacted.',
    });
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/generate') {
    const payload = await readRequestJson(request);
    const settings = payload.settings || {};
    const runtime = settings.runtime === 'local-python' ? 'local-python' : 'demo';
    if (runtime === 'local-python') {
      const result = await runPythonGeneration({
        prompt: payload.prompt,
        temperature: settings.temperature,
        maxTokens: settings.maxTokens,
        topK: settings.topK,
        checkpointPath: settings.modelCheckpointPath,
        tokenizerPath: settings.tokenizerPath,
      });
      sendJson(response, result.ok ? 200 : 500, {
        runtime,
        output: result.output,
        console: result.console || 'Local Python generation completed.',
        exitCode: result.exitCode,
      });
      return;
    }

    sendJson(response, 200, {
      runtime,
      output: demoGenerate(payload.prompt, settings),
      console: 'Generated with the built-in no-dependency interface demo runtime. No external AI service was contacted.',
    });
    return;
  }

  if (request.method === 'GET' && url.pathname === '/api/checkpoints') {
    const checkpoints = listCheckpoints();
    sendJson(response, 200, { checkpoints, names: checkpoints.map((checkpoint) => checkpoint.name) });
    return;
  }

  if (request.method === 'GET' && url.pathname === '/api/training/status') {
    sendJson(response, 200, getTrainingStatus());
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/training/start') {
    const payload = await readRequestJson(request);
    sendJson(response, 200, startTrainingProcess(payload));
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/training/stop') {
    sendJson(response, 200, stopTrainingProcess());
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/checkpoints/save') {
    const payload = await readRequestJson(request);
    const targetPath = safeCheckpointPath(payload.name);
    const checkpoint = {
      savedAt: new Date().toISOString(),
      type: 'interface-session',
      prompt: String(payload.prompt || ''),
      output: String(payload.output || ''),
      messages: Array.isArray(payload.messages) ? payload.messages : [],
      settings: payload.settings || {},
      note: 'UI checkpoint for local chat messages, prompts, output, and generation controls. Model tensor checkpoints remain supported by src/llm_prototype/model.py.',
    };
    fs.writeFileSync(targetPath, `${JSON.stringify(checkpoint, null, 2)}\n`, 'utf-8');
    sendJson(response, 200, { saved: path.basename(targetPath), checkpoint });
    return;
  }

  if (request.method === 'GET' && url.pathname.startsWith('/api/checkpoints/load/')) {
    const name = decodeURIComponent(url.pathname.split('/').pop() || '');
    const targetPath = safeCheckpointPath(name);
    if (!fs.existsSync(targetPath)) {
      sendJson(response, 404, { error: 'Checkpoint not found.' });
      return;
    }
    sendJson(response, 200, JSON.parse(fs.readFileSync(targetPath, 'utf-8')));
    return;
  }

  sendJson(response, 404, { error: 'API route not found.' });
}

function serveStatic(request, response, url) {
  const requestedPath = url.pathname === '/' ? '/index.html' : url.pathname;
  const filePath = path.normalize(path.join(PUBLIC_DIR, requestedPath));
  if (!filePath.startsWith(PUBLIC_DIR + path.sep)) {
    sendText(response, 403, 'Forbidden');
    return;
  }
  fs.readFile(filePath, (error, content) => {
    if (error) {
      sendText(response, 404, 'Not found');
      return;
    }
    response.writeHead(200, {
      'Content-Type': MIME_TYPES[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    response.end(content);
  });
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
  try {
    if (url.pathname.startsWith('/api/')) {
      await handleApi(request, response, url);
      return;
    }
    serveStatic(request, response, url);
  } catch (error) {
    sendJson(response, 500, { error: error.message });
  }
});

if (require.main === module) {
  server.listen(PORT, () => {
    startBackgroundIngestion();
    console.log(`Local conversation prototype running at http://localhost:${PORT}`);
    console.log('No external AI service is used. Add local text files under local_knowledge/ or use optional no-key web retrieval to expand local memory.');
  });
}

module.exports = {
  createLocalChatResponse,
  createLocalChatResponseAsync,
  extractSearchQuery,
  extractUserProvidedKnowledge,
  searchStoredWebKnowledge,
  parseMathExpression,
  tryAnswerMath,
  server,
};
