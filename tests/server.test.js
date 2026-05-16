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
