import fs from 'node:fs';

const html = fs.readFileSync('static/quantex.html', 'utf8');

const anchor = html.indexOf('const MODULES=[');
if (anchor === -1) { console.error('MODULES array not found'); process.exit(1); }

const arrStart = html.indexOf('[', anchor);
let depth = 0, inStr = false, strChar = null, escape = false, arrEnd = -1;
for (let j = arrStart; j < html.length; j++) {
  const c = html[j];
  if (escape) { escape = false; continue; }
  if (c === '\\') { escape = true; continue; }
  if (inStr) {
    if (c === strChar) inStr = false;
    continue;
  }
  if (c === '"' || c === "'") { inStr = true; strChar = c; continue; }
  if (c === '[') depth++;
  else if (c === ']') { depth--; if (depth === 0) { arrEnd = j; break; } }
}
if (arrEnd === -1) { console.error('Could not find end of MODULES array'); process.exit(1); }

const MODULES = eval(html.slice(arrStart, arrEnd + 1));

const violations = [];
const moduleCounts = {};
let totalQs = 0;

for (const mod of MODULES) {
  const quiz = mod.quiz || [];
  moduleCounts[mod.id] = { level: mod.level, count: quiz.length };
  totalQs += quiz.length;
  for (let qi = 0; qi < quiz.length; qi++) {
    const q = quiz[qi];
    const loc = `${mod.id}[${qi}]`;
    if (typeof q.q !== 'string' || q.q.trim() === '') {
      violations.push(`${loc}: 'q' missing or empty`);
    }
    if (!Array.isArray(q.opts) || q.opts.length !== 4) {
      const got = Array.isArray(q.opts) ? `array of ${q.opts.length}` : typeof q.opts;
      violations.push(`${loc}: 'opts' must be array of exactly 4 (got ${got})`);
    } else {
      for (let oi = 0; oi < q.opts.length; oi++) {
        if (typeof q.opts[oi] !== 'string' || q.opts[oi].trim() === '') {
          violations.push(`${loc}: opt[${oi}] missing or empty`);
        }
      }
    }
    if (!Number.isInteger(q.correct) || q.correct < 0 || q.correct > 3) {
      violations.push(`${loc}: 'correct' must be integer in [0,3] (got ${JSON.stringify(q.correct)})`);
    } else if (Array.isArray(q.opts) && q.correct >= q.opts.length) {
      violations.push(`${loc}: 'correct' ${q.correct} out of bounds for opts length ${q.opts.length}`);
    }
    if (typeof q.explain !== 'string' || q.explain.trim() === '') {
      violations.push(`${loc}: 'explain' missing or empty`);
    }
  }
}

console.log(`Total questions: ${totalQs}`);
console.log(`Modules: ${MODULES.length}`);
console.log('');
console.log('Per-module counts:');
for (const mod of MODULES) {
  const c = moduleCounts[mod.id];
  console.log(`  ${c.level} ${mod.id.padEnd(16)} ${c.count}`);
}
console.log('');
const l1Sum = MODULES.filter(m => m.level === 'L1').reduce((a,m) => a + (m.quiz || []).length, 0);
const l2Sum = MODULES.filter(m => m.level === 'L2').reduce((a,m) => a + (m.quiz || []).length, 0);
console.log(`L1 total: ${l1Sum}`);
console.log(`L2 total: ${l2Sum}`);
console.log(`Sum: ${l1Sum + l2Sum}`);
console.log('');

const correctByPos = [0, 0, 0, 0];
for (const mod of MODULES) {
  for (const q of (mod.quiz || [])) {
    if (Number.isInteger(q.correct) && q.correct >= 0 && q.correct <= 3) {
      correctByPos[q.correct]++;
    }
  }
}
console.log(`Correct-answer distribution: A=${correctByPos[0]}  B=${correctByPos[1]}  C=${correctByPos[2]}  D=${correctByPos[3]}`);
console.log('');

if (violations.length === 0) {
  console.log('No violations found. Structure is sound.');
} else {
  console.log(`VIOLATIONS (${violations.length}):`);
  for (const v of violations) console.log(`  ${v}`);
  process.exit(1);
}
