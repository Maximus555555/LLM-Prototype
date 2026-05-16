const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const CHECKPOINT_DIR = path.join(ROOT, 'checkpoints');
const PORT = Number(process.env.PORT || 3000);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml; charset=utf-8',
};

fs.mkdirSync(CHECKPOINT_DIR, { recursive: true });

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

async function handleApi(request, response, url) {
  if (request.method === 'GET' && url.pathname === '/api/status') {
    sendJson(response, 200, {
      name: 'LLM Prototype Interface',
      externalAiServices: false,
      apiKeysRequired: false,
      checkpointSupport: true,
      checkpointDirectory: 'checkpoints/',
      defaultConfig: 'configs/tiny.json',
      runtimes: ['demo', 'local-python'],
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
    const files = fs
      .readdirSync(CHECKPOINT_DIR, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
      .map((entry) => entry.name)
      .sort();
    sendJson(response, 200, { checkpoints: files });
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
      settings: payload.settings || {},
      note: 'UI checkpoint for prompts, output, and generation controls. Model tensor checkpoints remain supported by src/llm_prototype/model.py.',
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
  const url = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`);
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

server.listen(PORT, '0.0.0.0', () => {
  console.log(`LLM Prototype interface running at http://localhost:${PORT}`);
  console.log('No API keys or external AI services are required.');
});
