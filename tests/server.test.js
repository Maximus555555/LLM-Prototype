const assert = require('node:assert/strict');
const test = require('node:test');

const { createLocalChatResponse, tryAnswerMath } = require('../server');

function answer(message, history = []) {
  return createLocalChatResponse({ message, history }).answer;
}

test('answers basic arithmetic without using fallback retrieval guidance', () => {
  const response = answer('What is 20+5?');

  assert.match(response, /20\+5 = 25/);
  assert.doesNotMatch(response, /Try asking about the files listed/i);
  assert.doesNotMatch(response, /I remember you recently asked about/i);
});

test('supports operator precedence and parentheses in arithmetic', () => {
  assert.equal(tryAnswerMath('calculate (2 + 3) * 4'), '(2 + 3) * 4 = 20');
  assert.equal(tryAnswerMath('what is 2 + 3 * 4'), '2 + 3 * 4 = 14');
});

test('handles conversational check-ins directly', () => {
  const response = answer('How are you?', [{ role: 'user', content: 'Hello' }]);

  assert.match(response, /doing well/i);
  assert.match(response, /basic math/i);
  assert.doesNotMatch(response, /local-knowledge match/i);
  assert.doesNotMatch(response, /Try asking about the files listed/i);
});

test('fallback is helpful without dumping recent user messages', () => {
  const response = answer('Tell me about an unknown made up topic', [
    { role: 'user', content: 'Hello' },
    { role: 'user', content: 'What is 20+5' },
    { role: 'user', content: 'How are you?' },
  ]);

  assert.doesNotMatch(response, /I remember you recently asked about/i);
  assert.doesNotMatch(response, /Hello \| What is 20\+5 \| How are you/i);
  assert.match(response, /basic conversation, arithmetic/i);
});

test('detects explicit internet search requests without requiring paid services', () => {
  const { extractSearchQuery } = require('../server');

  assert.equal(extractSearchQuery('Search the internet for local-first RAG patterns'), 'local-first RAG patterns');
  assert.equal(extractSearchQuery('SEARCH THE INTERNET FOR local-first RAG patterns'), 'local-first RAG patterns');
  assert.equal(extractSearchQuery('what are the best small local AI models'), '');
  assert.equal(extractSearchQuery('look this up'), '');
  assert.equal(extractSearchQuery('find current news about AI'), '');
  assert.equal(extractSearchQuery('Tell me a joke'), '');
});

test('detects explicit user-provided knowledge for persistent memory', () => {
  const { extractUserProvidedKnowledge } = require('../server');

  assert.equal(
    extractUserProvidedKnowledge('Remember this: retrieval memory simulates learning without retraining model weights.'),
    'retrieval memory simulates learning without retraining model weights.',
  );
  assert.equal(extractUserProvidedKnowledge('Can you remember what I asked?'), '');
});


test('extracts article URLs and searchable article details', () => {
  const { extractUrls, findArticleDetails, extractNamesDatesNumbers } = require('../server');
  const page = {
    text: 'Jane Doe announced Project Atlas on May 5, 2026. The article says the pilot includes 42 schools and 12,000 students. A separate paragraph discusses unrelated background material.',
  };

  assert.deepEqual(extractUrls('Summarize https://example.com/article.'), ['https://example.com/article']);
  assert.match(findArticleDetails(page, 'how many schools are in the pilot')[0], /42 schools/);
  assert.deepEqual(extractNamesDatesNumbers(page.text).dates, ['May 5, 2026']);
});

test('GitHub Pages root serves the application shell directly', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const rootIndex = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

  assert.match(rootIndex, /<main class="app-shell">/);
  assert.match(rootIndex, /id="chatForm"/);
  assert.match(rootIndex, /href="public\/styles\.css"/);
  assert.match(rootIndex, /src="public\/app\.js"/);
  assert.doesNotMatch(rootIndex, /http-equiv="refresh"/i);
});

test('public app uses relative assets for project GitHub Pages paths', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const publicIndex = fs.readFileSync(path.join(__dirname, '..', 'public', 'index.html'), 'utf8');

  assert.match(publicIndex, /href="styles\.css"/);
  assert.match(publicIndex, /src="app\.js"/);
  assert.doesNotMatch(publicIndex, /href="\/styles\.css"/);
  assert.doesNotMatch(publicIndex, /src="\/app\.js"/);
});
