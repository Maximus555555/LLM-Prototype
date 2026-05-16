const elements = {
  serverStatus: document.querySelector('#serverStatus'),
  runtimeStatus: document.querySelector('#runtimeStatus'),
  chatHistory: document.querySelector('#chatHistory'),
  chatForm: document.querySelector('#chatForm'),
  messageInput: document.querySelector('#messageInput'),
  sendButton: document.querySelector('#sendButton'),
  clearChatButton: document.querySelector('#clearChatButton'),
  refreshKnowledgeButton: document.querySelector('#refreshKnowledgeButton'),
  exportKnowledgeButton: document.querySelector('#exportKnowledgeButton'),
  importKnowledgeButton: document.querySelector('#importKnowledgeButton'),
  clearKnowledgeButton: document.querySelector('#clearKnowledgeButton'),
  knowledgeImportInput: document.querySelector('#knowledgeImportInput'),
  backgroundStatus: document.querySelector('#backgroundStatus'),
  knowledgeCount: document.querySelector('#knowledgeCount'),
  chunkCount: document.querySelector('#chunkCount'),
  knowledgeList: document.querySelector('#knowledgeList'),
  promptInput: document.querySelector('#promptInput'),
  generateButton: document.querySelector('#generateButton'),
  clearButton: document.querySelector('#clearButton'),
  clearConsoleButton: document.querySelector('#clearConsoleButton'),
  samplePromptButton: document.querySelector('#samplePromptButton'),
  outputArea: document.querySelector('#outputArea'),
  outputMeta: document.querySelector('#outputMeta'),
  consoleArea: document.querySelector('#consoleArea'),
  runtimeSelect: document.querySelector('#runtimeSelect'),
  temperatureInput: document.querySelector('#temperatureInput'),
  temperatureValue: document.querySelector('#temperatureValue'),
  maxTokensInput: document.querySelector('#maxTokensInput'),
  topKInput: document.querySelector('#topKInput'),
  modelCheckpointPath: document.querySelector('#modelCheckpointPath'),
  tokenizerPath: document.querySelector('#tokenizerPath'),
  checkpointName: document.querySelector('#checkpointName'),
  checkpointSelect: document.querySelector('#checkpointSelect'),
  saveCheckpointButton: document.querySelector('#saveCheckpointButton'),
  loadCheckpointButton: document.querySelector('#loadCheckpointButton'),
  refreshCheckpointsButton: document.querySelector('#refreshCheckpointsButton'),
  trainingStatus: document.querySelector('#trainingStatus'),
  latestTrainLoss: document.querySelector('#latestTrainLoss'),
  latestValidationLoss: document.querySelector('#latestValidationLoss'),
  latestCheckpoint: document.querySelector('#latestCheckpoint'),
  trainBatchSize: document.querySelector('#trainBatchSize'),
  trainLearningRate: document.querySelector('#trainLearningRate'),
  trainMaxSteps: document.querySelector('#trainMaxSteps'),
  trainEvalInterval: document.querySelector('#trainEvalInterval'),
  trainCheckpointInterval: document.querySelector('#trainCheckpointInterval'),
  trainResumeCheckpoint: document.querySelector('#trainResumeCheckpoint'),
  startTrainingButton: document.querySelector('#startTrainingButton'),
  stopTrainingButton: document.querySelector('#stopTrainingButton'),
  refreshTrainingButton: document.querySelector('#refreshTrainingButton'),
};

const messages = [
  {
    role: 'assistant',
    content:
      'Hi — I am a local retrieval-and-rules conversation interface. Add files in local_knowledge/ or ask me to search the internet; I save retrieved summaries locally without OpenAI or API keys.',
    sources: [],
  },
];

function logConsole(message, level = 'info') {
  const timestamp = new Date().toLocaleTimeString();
  const prefix = level === 'error' ? 'ERROR' : 'INFO';
  elements.consoleArea.textContent += `\n[${timestamp}] ${prefix}: ${message}`;
  elements.consoleArea.scrollTop = elements.consoleArea.scrollHeight;
}

function readSettings() {
  return {
    runtime: elements.runtimeSelect.value,
    temperature: Number(elements.temperatureInput.value),
    maxTokens: Number(elements.maxTokensInput.value),
    topK: Number(elements.topKInput.value),
    modelCheckpointPath: elements.modelCheckpointPath.value.trim(),
    tokenizerPath: elements.tokenizerPath.value.trim(),
  };
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || data.console || `Request failed with status ${response.status}`);
  }
  return data;
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
  elements.outputMeta.textContent = /\b(search|internet|web|online|look up|browse)\b/i.test(message) ? 'Searching web…' : 'Searching knowledge…';
  renderMessages();

  try {
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
    elements.outputMeta.textContent = 'Error';
    logConsole(error.message, 'error');
  } finally {
    elements.sendButton.disabled = false;
    elements.messageInput.focus();
  }
}

function renderKnowledge(data) {
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
  try {
    const data = await requestJson('/api/knowledge');
    renderKnowledge(data);
    logConsole(`Loaded ${data.documents.length} knowledge document(s) from ${data.directory} and ${data.webKnowledgePath}.`);
  } catch (error) {
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
    batchSize: Number(elements.trainBatchSize.value),
    learningRate: Number(elements.trainLearningRate.value),
    maxSteps: Number(elements.trainMaxSteps.value),
    evalInterval: Number(elements.trainEvalInterval.value),
    checkpointInterval: Number(elements.trainCheckpointInterval.value),
    resume: elements.trainResumeCheckpoint.value.trim(),
  };
}

function renderTrainingStatus(status) {
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
    elements.serverStatus.textContent = status.name;
    elements.runtimeStatus.textContent = `${status.localKnowledgeFiles} knowledge docs • ${status.webKnowledgeItems || 0} web items • API keys required: ${status.apiKeysRequired ? 'yes' : 'no'}`;
    logConsole(`Server ready. Knowledge directory: ${status.knowledgeDirectory}`);
  } catch (error) {
    elements.serverStatus.textContent = 'Server status failed';
    elements.runtimeStatus.textContent = error.message;
    logConsole(error.message, 'error');
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
      content: 'Conversation memory cleared. Ask a new question and I will search saved local/web knowledge before answering; explicitly ask me to search the internet to retrieve new pages.',
      sources: [],
    },
  ]);
  elements.outputArea.textContent = 'Responses will appear here.';
  elements.outputMeta.textContent = 'Cleared';
  logConsole('Chat history cleared.');
}

function wireEvents() {
  elements.temperatureInput.addEventListener('input', () => {
    elements.temperatureValue.textContent = elements.temperatureInput.value;
  });
  elements.chatForm.addEventListener('submit', sendMessage);
  elements.messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      elements.chatForm.requestSubmit();
    }
  });
  elements.generateButton.addEventListener('click', generate);
  elements.clearButton.addEventListener('click', () => {
    elements.outputArea.textContent = 'Responses will appear here.';
    elements.outputMeta.textContent = 'Cleared';
    logConsole('Output cleared.');
  });
  elements.clearChatButton.addEventListener('click', clearChat);
  elements.refreshKnowledgeButton.addEventListener('click', refreshKnowledge);
  elements.exportKnowledgeButton.addEventListener('click', exportStoredKnowledge);
  elements.importKnowledgeButton.addEventListener('click', () => elements.knowledgeImportInput.click());
  elements.knowledgeImportInput.addEventListener('change', importStoredKnowledgeFile);
  elements.clearKnowledgeButton.addEventListener('click', clearStoredKnowledge);
  elements.clearConsoleButton.addEventListener('click', () => {
    elements.consoleArea.textContent = 'Console is ready.';
  });
  elements.samplePromptButton.addEventListener('click', () => {
    elements.messageInput.value = 'Search the internet for practical programming education resources';
    elements.messageInput.focus();
  });
  elements.saveCheckpointButton.addEventListener('click', saveCheckpoint);
  elements.loadCheckpointButton.addEventListener('click', loadCheckpoint);
  elements.refreshCheckpointsButton.addEventListener('click', refreshCheckpoints);
  elements.startTrainingButton.addEventListener('click', startTraining);
  elements.stopTrainingButton.addEventListener('click', stopTraining);
  elements.refreshTrainingButton.addEventListener('click', refreshTrainingStatus);
}

renderMessages();
wireEvents();
loadStatus();
refreshKnowledge();
refreshCheckpoints().catch((error) => logConsole(error.message, 'error'));
refreshTrainingStatus();
setInterval(refreshTrainingStatus, 5000);
setInterval(refreshKnowledge, 15000);
