const elements = {
  serverStatus: document.querySelector('#serverStatus'),
  runtimeStatus: document.querySelector('#runtimeStatus'),
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
};

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

async function refreshCheckpoints() {
  const data = await requestJson('/api/checkpoints');
  elements.checkpointSelect.innerHTML = '';
  if (!data.checkpoints.length) {
    const option = document.createElement('option');
    option.textContent = 'No checkpoints saved yet';
    option.value = '';
    elements.checkpointSelect.append(option);
    return;
  }
  data.checkpoints.forEach((checkpoint) => {
    const option = document.createElement('option');
    option.value = checkpoint;
    option.textContent = checkpoint;
    elements.checkpointSelect.append(option);
  });
}

async function loadStatus() {
  try {
    const status = await requestJson('/api/status');
    elements.serverStatus.textContent = status.name;
    elements.runtimeStatus.textContent = `${status.runtimes.length} runtimes • API keys required: ${status.apiKeysRequired ? 'yes' : 'no'}`;
    logConsole(`Server ready. Checkpoints directory: ${status.checkpointDirectory}`);
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
    elements.outputMeta.textContent = 'Loaded';
    logConsole(`Loaded checkpoint ${selected}.`);
  } catch (error) {
    logConsole(error.message, 'error');
  }
}

function wireEvents() {
  elements.temperatureInput.addEventListener('input', () => {
    elements.temperatureValue.textContent = elements.temperatureInput.value;
  });
  elements.generateButton.addEventListener('click', generate);
  elements.clearButton.addEventListener('click', () => {
    elements.outputArea.textContent = 'Generated text will appear here.';
    elements.outputMeta.textContent = 'Cleared';
    logConsole('Output cleared.');
  });
  elements.clearConsoleButton.addEventListener('click', () => {
    elements.consoleArea.textContent = 'Console is ready.';
  });
  elements.samplePromptButton.addEventListener('click', () => {
    elements.promptInput.value = 'Summarize the purpose of this local LLM prototype interface in three bullet points.';
    elements.promptInput.focus();
  });
  elements.saveCheckpointButton.addEventListener('click', saveCheckpoint);
  elements.loadCheckpointButton.addEventListener('click', loadCheckpoint);
}

wireEvents();
loadStatus();
refreshCheckpoints().catch((error) => logConsole(error.message, 'error'));
