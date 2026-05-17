#!/usr/bin/env node
const path = require('node:path');
const { spawn } = require('node:child_process');

const command = (process.argv[2] || 'start').toLowerCase();
const allowedCommands = new Set(['start', 'serve', 'run']);

if (!allowedCommands.has(command)) {
  console.error(`Unsupported GitHub npm command: ${command}`);
  console.error('Use `npm run github start` or `npm run github:start` to launch the local app.');
  process.exit(1);
}

const serverPath = path.join(__dirname, '..', 'server.js');
const child = spawn(process.execPath, [serverPath], {
  stdio: 'inherit',
  env: process.env,
});

const forwardSignal = (signal) => {
  if (!child.killed) {
    child.kill(signal);
  }
};

process.on('SIGINT', () => forwardSignal('SIGINT'));
process.on('SIGTERM', () => forwardSignal('SIGTERM'));

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code || 0);
});
