const q = (selector) => document.querySelector(selector);
let staticPreviewMode = false;
let knowledgeRefreshTimer = null;

const elements = {
  serverStatus: q('#serverStatus'),
  runtimeStatus: q('#runtimeStatus'),
  chatHistory: q('#chatHistory'),
  chatForm: q('#chatForm'),
  messageInput: q('#messageInput'),
  sendButton: q('#sendButton'),
  clearChatButton: q('#clearChatButton'),
  refreshKnowledgeButton: q('#refreshKnowledgeButton'),
  exportKnowledgeButton: q('#exportKnowledgeButton'),
  importKnowledgeButton: q('#importKnowledgeButton'),
  clearKnowledgeButton: q('#clearKnowledgeButton'),
  knowledgeImportInput: q('#knowledgeImportInput'),
  backgroundStatus: q('#backgroundStatus'),
  knowledgeCount: q('#knowledgeCount'),
  chunkCount: q('#chunkCount'),
  knowledgeList: q('#knowledgeList'),
  promptInput: q('#promptInput'),
  generateButton: q('#generateButton'),
  clearButton: q('#clearButton'),
  clearConsoleButton: q('#clearConsoleButton'),
  samplePromptButton: q('#samplePromptButton'),
  outputArea: q('#outputArea'),
  outputMeta: q('#outputMeta'),
  consoleArea: q('#consoleArea'),
  runtimeSelect: q('#runtimeSelect'),
  temperatureInput: q('#temperatureInput'),
  temperatureValue: q('#temperatureValue'),
  maxTokensInput: q('#maxTokensInput'),
  topKInput: q('#topKInput'),
  modelCheckpointPath: q('#modelCheckpointPath'),
  tokenizerPath: q('#tokenizerPath'),
  checkpointName: q('#checkpointName'),
  checkpointSelect: q('#checkpointSelect'),
  saveCheckpointButton: q('#saveCheckpointButton'),
  loadCheckpointButton: q('#loadCheckpointButton'),
  refreshCheckpointsButton: q('#refreshCheckpointsButton'),
  trainingStatus: q('#trainingStatus'),
  latestTrainLoss: q('#latestTrainLoss'),
  latestValidationLoss: q('#latestValidationLoss'),
  latestCheckpoint: q('#latestCheckpoint'),
  trainBatchSize: q('#trainBatchSize'),
  trainLearningRate: q('#trainLearningRate'),
  trainMaxSteps: q('#trainMaxSteps'),
  trainEvalInterval: q('#trainEvalInterval'),
  trainCheckpointInterval: q('#trainCheckpointInterval'),
  trainDevice: q('#trainDevice'),
  trainResumeCheckpoint: q('#trainResumeCheckpoint'),
  startTrainingButton: q('#startTrainingButton'),
  stopTrainingButton: q('#stopTrainingButton'),
  refreshTrainingButton: q('#refreshTrainingButton'),
};

const messages = [
  {
    role: 'assistant',
    content:
      'Hi — I am a local chat and article/webpage interface. Paste a URL to summarize or inspect a page. I only search the wider internet when your prompt includes the exact phrase “search the internet for”.',
    sources: [],
  },
];

function logConsole(message, level = 'info') {
  const timestamp = new Date().toLocaleTimeString();
  const prefix = level === 'error' ? 'ERROR' : 'INFO';
  if (!elements.consoleArea) {
    return;
  }
  elements.consoleArea.textContent += `\n[${timestamp}] ${prefix}: ${message}`;
  elements.consoleArea.scrollTop = elements.consoleArea.scrollHeight;
}

function readSettings() {
  return {
    runtime: elements.runtimeSelect?.value || 'demo',
    temperature: Number(elements.temperatureInput?.value || 0.8),
    maxTokens: Number(elements.maxTokensInput?.value || 80),
    topK: Number(elements.topKInput?.value || 50),
    modelCheckpointPath: elements.modelCheckpointPath?.value.trim() || '',
    tokenizerPath: elements.tokenizerPath?.value.trim() || '',
  };
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json')
    ? await response.json()
    : { error: `Request failed with status ${response.status}` };
  if (!response.ok) {
    const error = new Error(data.error || data.console || `Request failed with status ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function isApiUnavailable(error) {
  return error?.status === 404 || /Failed to fetch|Request failed with status 404|API route not found/i.test(error?.message || '');
}

function createStaticPreviewResponse(message) {
  const arithmetic = message.match(/[-+*/(). 0-9]+/g)?.find((part) => /[0-9]/.test(part) && /[+*/-]/.test(part));
  if (arithmetic && /^[0-9+*\/().\s-]+$/.test(arithmetic)) {
    try {
      const value = Function(`"use strict"; return (${arithmetic});`)();
      if (Number.isFinite(value)) {
        return `Static GitHub Pages preview: ${arithmetic.trim()} = ${value}. Start the local Node server with npm start for chat, URL retrieval, and knowledge features.`;
      }
    } catch (error) {
      // Fall through to the normal static preview message.
    }
  }
  return 'Static GitHub Pages preview loaded the application shell, but GitHub Pages cannot run the local Node API. Start the project with npm start to use chat responses, article retrieval, saved knowledge, and training features.';
}

function renderMessages() {
  elements.chatHistory.innerHTML = '';
  messages.forEach((message) => {
    const article = document.createElement('article');
    article.className = `chat-message ${message.role}`;

    const label = document.createElement('strong');
    label.textContent = message.role === 'user' ? 'You' : 'Local AI';

    const body = document.createElement('p');
    body.textContent = message.content;

    article.append(label, body);

    if (message.sources?.length) {
      const sources = document.createElement('div');
      sources.className = 'source-list';
      sources.textContent = `Sources: ${message.sources.map((source) => `${source.source}#chunk-${source.chunk}`).join(', ')}`;
      article.append(sources);
    }

    elements.chatHistory.append(article);
  });
  elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
}

function setMessages(nextMessages) {
  messages.splice(0, messages.length, ...nextMessages);
  renderMessages();
}

async function sendMessage(event) {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message) {
    return;
  }

  messages.push({ role: 'user', content: message });
  elements.messageInput.value = '';
  elements.sendButton.disabled = true;
  elements.outputMeta.textContent = staticPreviewMode ? 'Static preview' : (/search the internet for/i.test(message) ? 'Searching web…' : (/https?:\/\//i.test(message) ? 'Retrieving page…' : 'Thinking locally…'));
  renderMessages();

  try {
    if (staticPreviewMode) {
      throw Object.assign(new Error('Static preview mode'), { status: 404 });
    }
    const data = await requestJson('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message, history: messages.slice(-12) }),
    });
    messages.push({ role: 'assistant', content: data.answer, sources: data.sources || [] });
    elements.outputArea.textContent = data.answer;
    elements.outputMeta.textContent = data.webResults?.length ? `${data.webResults.length} web result(s)` : (data.sources?.length ? `${data.sources.length} knowledge match(es)` : 'Fallback');
    logConsole(data.console || 'Local chat response finished.');
    renderMessages();
    await refreshKnowledge();
  } catch (error) {
    if (isApiUnavailable(error)) {
      const staticResponse = createStaticPreviewResponse(message);
      messages.push({ role: 'assistant', content: staticResponse, sources: [] });
      elements.outputArea.textContent = staticResponse;
      elements.outputMeta.textContent = 'Static preview';
      logConsole('Static preview mode: local API is not available on this host.');
      renderMessages();
    } else {
      elements.outputMeta.textContent = 'Error';
      logConsole(error.message, 'error');
    }
  } finally {
    elements.sendButton.disabled = false;
    elements.messageInput.focus();
  }
}

function renderKnowledge(data) {
  if (!elements.knowledgeCount || !elements.knowledgeList) {
    const background = data.webKnowledge?.background || {};
    if (elements.backgroundStatus) {
      elements.backgroundStatus.textContent = `Background ingestion: ${background.enabled ? 'enabled' : 'disabled'}${background.running ? ' • running' : ''}${background.lastRunAt ? ` • last run ${new Date(background.lastRunAt).toLocaleTimeString()}` : ''}`;
    }
    return;
  }
  const documents = data.documents || [];
  const webCount = data.webKnowledge?.items || documents.filter((document) => document.type === 'web-knowledge').length;
  const localCount = documents.filter((document) => document.type !== 'web-knowledge').length;
  elements.knowledgeCount.textContent = `${localCount} file${localCount === 1 ? '' : 's'} • ${webCount} web item${webCount === 1 ? '' : 's'}`;
  if (typeof data.chunkCount === 'number') {
    elements.chunkCount.textContent = `${data.chunkCount} chunks indexed`;
  } else {
    const chunks = documents.reduce((total, document) => total + (document.chunks || 0), 0);
    elements.chunkCount.textContent = `${chunks} chunks indexed`;
  }
  elements.knowledgeList.innerHTML = '';
  if (elements.backgroundStatus) {
    const background = data.webKnowledge?.background || {};
    elements.backgroundStatus.textContent = `Background web ingestion: ${background.enabled ? 'enabled' : 'disabled'}${background.running ? ' • running' : ''}${background.lastRunAt ? ` • last run ${new Date(background.lastRunAt).toLocaleTimeString()}` : ''}${background.lastError ? ` • ${background.lastError}` : ''}`;
  }
  documents.forEach((localDocument) => {
    const item = window.document.createElement('li');
    const label = localDocument.type === 'web-knowledge' ? 'web' : 'file';
    item.innerHTML = `<strong>${localDocument.title || localDocument.source}</strong><small>${label} • ${localDocument.chunks} chunk(s) • ${localDocument.characters} characters${localDocument.dateSaved ? ` • saved ${new Date(localDocument.dateSaved).toLocaleDateString()}` : ''}</small>`;
    elements.knowledgeList.append(item);
  });
  if (!documents.length) {
    const item = window.document.createElement('li');
    item.textContent = 'No local knowledge files found yet.';
    elements.knowledgeList.append(item);
  }
}

async function refreshKnowledge() {
  if (staticPreviewMode) {
    return;
  }
  try {
    const data = await requestJson('/api/knowledge');
    renderKnowledge(data);
    logConsole(`Loaded ${data.documents.length} knowledge document(s) from ${data.directory} and ${data.webKnowledgePath}.`);
  } catch (error) {
    if (isApiUnavailable(error)) {
      staticPreviewMode = true;
      if (knowledgeRefreshTimer) {
        clearInterval(knowledgeRefreshTimer);
        knowledgeRefreshTimer = null;
      }
      return;
    }
    logConsole(error.message, 'error');
  }
}

async function refreshCheckpoints() {
  const data = await requestJson('/api/checkpoints');
  const checkpoints = (data.checkpoints || []).map((checkpoint) => (typeof checkpoint === 'string' ? { name: checkpoint, path: `checkpoints/${checkpoint}`, type: checkpoint.endsWith('.pt') ? 'model' : 'session' } : checkpoint));
  elements.checkpointSelect.innerHTML = '';
  if (!checkpoints.length) {
    const option = window.document.createElement('option');
    option.textContent = 'No checkpoints saved yet';
    option.value = '';
    elements.checkpointSelect.append(option);
    return;
  }
  checkpoints.forEach((checkpoint) => {
    const option = window.document.createElement('option');
    option.value = checkpoint.name;
    option.dataset.path = checkpoint.path || `checkpoints/${checkpoint.name}`;
    option.dataset.type = checkpoint.type || 'session';
    option.textContent = `${checkpoint.name} (${checkpoint.type || 'session'})`;
    elements.checkpointSelect.append(option);
  });
}



async function clearStoredKnowledge() {
  if (!window.confirm('Clear saved web knowledge from data/web_knowledge.json? Local files in local_knowledge/ will not be deleted.')) {
    return;
  }
  try {
    const data = await requestJson('/api/knowledge/clear', { method: 'POST', body: JSON.stringify({}) });
    logConsole(`Cleared saved web knowledge. ${data.total} web item(s) remain.`);
    await refreshKnowledge();
  } catch (error) {
    logConsole(error.message, 'error');
  }
}

async function exportStoredKnowledge() {
  try {
    const data = await requestJson('/api/knowledge/export');
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `web-knowledge-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
    logConsole(`Exported ${data.items?.length || 0} saved web knowledge item(s).`);
  } catch (error) {
    logConsole(error.message, 'error');
  }
}

async function importStoredKnowledgeFile(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  try {
    const payload = JSON.parse(await file.text());
    const data = await requestJson('/api/knowledge/import', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    logConsole(`Imported ${data.added} web knowledge item(s). ${data.total} total saved.`);
    await refreshKnowledge();
  } catch (error) {
    logConsole(error.message, 'error');
  } finally {
    event.target.value = '';
  }
}

function formatLoss(value) {
  return Number.isFinite(value) ? value.toFixed(4) : '—';
}

function readTrainingSettings() {
  return {
    batchSize: Number(elements.trainBatchSize?.value || 8),
    learningRate: Number(elements.trainLearningRate?.value || 0.0003),
    maxSteps: Number(elements.trainMaxSteps?.value || 200),
    evalInterval: Number(elements.trainEvalInterval?.value || 25),
    checkpointInterval: Number(elements.trainCheckpointInterval?.value || 50),
    device: elements.trainDevice?.value || 'auto',
    resume: elements.trainResumeCheckpoint?.value.trim() || '',
  };
}

function renderTrainingStatus(status) {
  if (!elements.trainingStatus) {
    return;
  }
  elements.trainingStatus.textContent = status.running ? 'Running' : status.status || 'idle';
  elements.latestTrainLoss.textContent = formatLoss(status.latestTrainLoss);
  elements.latestValidationLoss.textContent = formatLoss(status.latestValidationLoss);
  elements.latestCheckpoint.textContent = status.latestCheckpoint || (status.hasLatestCheckpoint ? 'checkpoints/latest.pt' : '—');
  elements.startTrainingButton.disabled = Boolean(status.running);
  elements.stopTrainingButton.disabled = !status.running;
  if (status.console) {
    elements.consoleArea.textContent = `Console is ready.\n\n--- Training ---\n${status.console}`;
    elements.consoleArea.scrollTop = elements.consoleArea.scrollHeight;
  }
}

async function refreshTrainingStatus() {
  try {
    const status = await requestJson('/api/training/status');
    renderTrainingStatus(status);
    await refreshCheckpoints();
  } catch (error) {
    logConsole(error.message, 'error');
  }
}

async function startTraining() {
  try {
    const status = await requestJson('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(readTrainingSettings()),
    });
    renderTrainingStatus(status);
    logConsole('Started local training. This uses local_training_data/ only and may be slow on CPU.');
  } catch (error) {
    logConsole(error.message, 'error');
  }
}

async function stopTraining() {
  try {
    const status = await requestJson('/api/training/stop', { method: 'POST', body: JSON.stringify({}) });
    renderTrainingStatus(status);
    logConsole('Stop requested for local training.');
  } catch (error) {
    logConsole(error.message, 'error');
  }
}

async function loadStatus() {
  try {
    const status = await requestJson('/api/status');
    staticPreviewMode = false;
    elements.serverStatus.textContent = status.name;
    elements.runtimeStatus.textContent = `Chat + webpage mode • ${status.webKnowledgeItems || 0} saved web items • API keys required: ${status.apiKeysRequired ? 'yes' : 'no'}`;
    logConsole(`Server ready. Knowledge directory: ${status.knowledgeDirectory}`);
    return true;
  } catch (error) {
    if (isApiUnavailable(error)) {
      staticPreviewMode = true;
      elements.serverStatus.textContent = 'Static GitHub Pages preview';
      elements.runtimeStatus.textContent = 'Interface loaded without the local Node API. Run npm start for the full app.';
      if (elements.backgroundStatus) {
        elements.backgroundStatus.textContent = 'Static preview: local API polling is disabled on GitHub Pages.';
      }
      logConsole('Static preview mode: run npm start locally to enable API-backed chat and retrieval.');
    } else {
      elements.serverStatus.textContent = 'Server status failed';
      elements.runtimeStatus.textContent = error.message;
      logConsole(error.message, 'error');
    }
    return false;
  }
}

async function generate() {
  elements.generateButton.disabled = true;
  elements.outputMeta.textContent = 'Generating…';
  try {
    const data = await requestJson('/api/generate', {
      method: 'POST',
      body: JSON.stringify({ prompt: elements.promptInput.value, settings: readSettings() }),
    });
    elements.outputArea.textContent = data.output || '(No output returned.)';
    elements.outputMeta.textContent = data.runtime;
    logConsole(data.console || 'Generation finished.');
  } catch (error) {
    elements.outputMeta.textContent = 'Error';
    logConsole(error.message, 'error');
  } finally {
    elements.generateButton.disabled = false;
  }
}

async function saveCheckpoint() {
  try {
    const data = await requestJson('/api/checkpoints/save', {
      method: 'POST',
      body: JSON.stringify({
        name: elements.checkpointName.value,
        prompt: elements.promptInput.value,
        output: elements.outputArea.textContent,
        messages,
        settings: readSettings(),
      }),
    });
    logConsole(`Saved checkpoint ${data.saved}.`);
    await refreshCheckpoints();
    elements.checkpointSelect.value = data.saved;
  } catch (error) {
    logConsole(error.message, 'error');
  }
}

async function loadCheckpoint() {
  const selected = elements.checkpointSelect.value;
  if (!selected) {
    logConsole('Select a checkpoint to load.', 'error');
    return;
  }
  try {
    const selectedOption = elements.checkpointSelect.selectedOptions[0];
    if (selectedOption?.dataset.type === 'model' || selected.endsWith('.pt')) {
      elements.runtimeSelect.value = 'local-python';
      elements.modelCheckpointPath.value = selectedOption?.dataset.path || `checkpoints/${selected}`;
      elements.outputMeta.textContent = 'Model checkpoint selected';
      logConsole(`Selected model checkpoint ${elements.modelCheckpointPath.value} for local Python generation.`);
      return;
    }
    const checkpoint = await requestJson(`/api/checkpoints/load/${encodeURIComponent(selected)}`);
    elements.promptInput.value = checkpoint.prompt || '';
    elements.outputArea.textContent = checkpoint.output || '';
    const settings = checkpoint.settings || {};
    elements.runtimeSelect.value = settings.runtime || 'demo';
    elements.temperatureInput.value = settings.temperature || 0.8;
    elements.temperatureValue.textContent = elements.temperatureInput.value;
    elements.maxTokensInput.value = settings.maxTokens || 80;
    elements.topKInput.value = settings.topK || 50;
    elements.modelCheckpointPath.value = settings.modelCheckpointPath || '';
    elements.tokenizerPath.value = settings.tokenizerPath || '';
    if (Array.isArray(checkpoint.messages) && checkpoint.messages.length) {
      setMessages(checkpoint.messages);
    }
    elements.outputMeta.textContent = 'Loaded';
    logConsole(`Loaded checkpoint ${selected}.`);
  } catch (error) {
    logConsole(error.message, 'error');
  }
}

function clearChat() {
  setMessages([
    {
      role: 'assistant',
      content: 'Conversation memory cleared. Ask a normal chat question, paste a URL for article/page help, or use the exact phrase “search the internet for” to retrieve new pages.',
      sources: [],
    },
  ]);
  elements.outputArea.textContent = 'Responses will appear here.';
  elements.outputMeta.textContent = 'Cleared';
  logConsole('Chat history cleared.');
}

function on(element, eventName, handler) {
  if (element) {
    element.addEventListener(eventName, handler);
  }
}

function wireEvents() {
  on(elements.temperatureInput, 'input', () => {
    if (elements.temperatureValue) {
      elements.temperatureValue.textContent = elements.temperatureInput.value;
    }
  });
  on(elements.chatForm, 'submit', sendMessage);
  on(elements.messageInput, 'keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      elements.chatForm.requestSubmit();
    }
  });
  on(elements.generateButton, 'click', generate);
  on(elements.clearButton, 'click', () => {
    elements.outputArea.textContent = 'Responses will appear here.';
    elements.outputMeta.textContent = 'Cleared';
    logConsole('Output cleared.');
  });
  on(elements.clearChatButton, 'click', clearChat);
  on(elements.refreshKnowledgeButton, 'click', refreshKnowledge);
  on(elements.exportKnowledgeButton, 'click', exportStoredKnowledge);
  on(elements.importKnowledgeButton, 'click', () => elements.knowledgeImportInput?.click());
  on(elements.knowledgeImportInput, 'change', importStoredKnowledgeFile);
  on(elements.clearKnowledgeButton, 'click', clearStoredKnowledge);
  on(elements.clearConsoleButton, 'click', () => {
    elements.consoleArea.textContent = 'Console is ready.';
  });
  on(elements.samplePromptButton, 'click', () => {
    elements.messageInput.value = 'Summarize https://example.com/ and list names, dates, and numbers mentioned.';
    elements.messageInput.focus();
  });
  on(elements.saveCheckpointButton, 'click', saveCheckpoint);
  on(elements.loadCheckpointButton, 'click', loadCheckpoint);
  on(elements.refreshCheckpointsButton, 'click', refreshCheckpoints);
  on(elements.startTrainingButton, 'click', startTraining);
  on(elements.stopTrainingButton, 'click', stopTraining);
  on(elements.refreshTrainingButton, 'click', refreshTrainingStatus);
}

async function initializeApp() {
  renderMessages();
  wireEvents();
  const apiAvailable = await loadStatus();
  if (!apiAvailable) {
    return;
  }
  await refreshKnowledge();
  if (elements.checkpointSelect) {
    refreshCheckpoints().catch((error) => logConsole(error.message, 'error'));
  }
  if (elements.trainingStatus) {
    refreshTrainingStatus();
    setInterval(refreshTrainingStatus, 5000);
  }
  knowledgeRefreshTimer = setInterval(refreshKnowledge, 30000);
}

initializeApp();
