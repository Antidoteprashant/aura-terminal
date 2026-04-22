/* ============================================================
   RAG Terminal — app.js
   Vanilla JS — no frameworks, no build step
   ============================================================ */

'use strict';

// ── API endpoints ─────────────────────────────────────────────
const API = {
  UPLOAD:        '/api/upload',
  QUERY:         '/api/query',
  SUMMARIZE:     '/api/summarize',
  DOCUMENTS:     '/api/documents',
  DOCUMENT:      (id) => `/api/documents/${id}`,
  HEALTH:        '/api/health',
  CONVERSATIONS: '/api/conversations',
};

// ── State ─────────────────────────────────────────────────────
const state = {
  conversationId: crypto.randomUUID(),
  documents:      [],     // [{id, filename, chunks, upload_date}]
  streaming:      false,
  cmdHistory:     [],     // past commands, newest first
  historyIdx:     -1,     // -1 = not navigating
};

// ── DOM references ────────────────────────────────────────────
const output      = document.getElementById('output');
const cmdInput    = document.getElementById('cmd-input');
const inputMirror = document.getElementById('input-mirror');
const cursorEl    = document.getElementById('cursor');
const fileInput   = document.getElementById('file-input');
const dropZone    = document.getElementById('drop-zone');
const ollamaBadge = document.getElementById('ollama-status');
const modelBadge  = document.getElementById('model-info');
const clockEl     = document.getElementById('clock');
const micToggle   = document.getElementById('mic-toggle');
const ttsToggle   = document.getElementById('tts-toggle');
const voiceBtn    = document.getElementById('voice-btn');
const uploadBtn   = document.getElementById('upload-btn');

// ── Utilities ─────────────────────────────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function scrollToBottom() {
  output.scrollTop = output.scrollHeight;
}

function padEnd(str, n) {
  return String(str).slice(0, n).padEnd(n);
}

// ── Cursor positioning ─────────────────────────────────────────
// Mirrors the input text to a hidden span so we know the pixel width,
// then places the block cursor right after the last character.
function syncCursor() {
  // Mirror only the text to the LEFT of the caret so the block cursor
  // tracks the actual insertion point, not always the end of the string.
  inputMirror.textContent = cmdInput.value.slice(0, cmdInput.selectionStart ?? cmdInput.value.length);
}

// ── Output primitives ─────────────────────────────────────────
function createBlock(cls) {
  const div = document.createElement('div');
  div.className = 'output-block' + (cls ? ' ' + cls : '');
  output.appendChild(div);
  scrollToBottom();
  return div;
}

/**
 * Append a text block to the terminal output.
 * @param {string} text
 * @param {string} cls  — one of: response | error | success | info | warning | doc-table
 * @returns {HTMLPreElement}
 */
function print(text, cls = 'response') {
  const block = createBlock(cls);
  const pre = document.createElement('pre');
  pre.textContent = text;
  block.appendChild(pre);
  scrollToBottom();
  return pre;
}

function printCmd(cmd) {
  const block = createBlock('cmd-block');
  block.textContent = '> ' + cmd;
}

function printError(msg)   { print('✗  ' + msg, 'error');   }
function printSuccess(msg) { print('✓  ' + msg, 'success'); }
function printInfo(msg)    { print(msg, 'info');             }

function printSources(sources) {
  if (!sources || sources.length === 0) return;
  // Deduplicate by doc_id
  const seen = new Set();
  const unique = sources.filter((s) => {
    if (seen.has(s.doc_id)) return false;
    seen.add(s.doc_id);
    return true;
  });
  const block = createBlock('sources');
  block.textContent = '── Sources: ' + unique.map((s) => s.filename).join(', ');
}

// ── Thinking indicator ────────────────────────────────────────
function _makeThinkingEl() {
  const wrap = document.createElement('div');
  wrap.className = 'thinking-block';

  const label = document.createElement('span');
  label.className = 'thinking-label';
  label.textContent = 'thinking';

  const dots = document.createElement('span');
  dots.className = 'thinking-dots';
  for (let i = 0; i < 3; i++) {
    const d = document.createElement('span');
    d.className = 'thinking-dot';
    dots.appendChild(d);
  }

  wrap.appendChild(label);
  wrap.appendChild(dots);
  return wrap;
}

// ── SSE stream consumer ───────────────────────────────────────
/**
 * POST to a streaming SSE endpoint and invoke callbacks per event.
 *
 * @param {string}   url
 * @param {object}   body
 * @param {Function} onToken       — async (token: string) => void
 * @param {Function} onDone        — (data: object) => void  (optional)
 * @param {Function} onContextNote — (note: string) => void  (optional)
 */
async function streamSSE(url, body, onToken, onDone, onContextNote) {
  state.streaming = true;
  cmdInput.disabled = true;
  // Hide the input-bar cursor while streaming — the response area has its own
  cursorEl.style.visibility = 'hidden';

  try {
    const resp = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ message: `HTTP ${resp.status}` }));
      printError(err.message || `HTTP ${resp.status}`);
      return;
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let gotDone = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();   // last (possibly incomplete) line kept for next chunk

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          let evt;
          try { evt = JSON.parse(raw); } catch { continue; }

          if (evt.context_note !== undefined) {
            onContextNote && onContextNote(evt.context_note);
          }
          if (evt.token !== undefined) {
            await onToken(evt.token);
          }
          if (evt.done) {
            gotDone = true;
            onDone && onDone(evt);
          }
        }
      }
    } catch (readErr) {
      // Stream dropped mid-response (network loss, server restart)
      if (!gotDone) {
        printError('Stream interrupted — ' + (readErr.message || 'connection lost'));
      }
    }
  } catch (err) {
    printError('Connection error — ' + (err.message || 'could not reach server'));
  } finally {
    state.streaming = false;
    cmdInput.disabled = false;
    // Restore input-bar cursor, then focus so blur→visibility logic runs correctly
    cursorEl.style.visibility = 'visible';
    cmdInput.focus();
  }
}

// ── Command: ask a question ───────────────────────────────────
async function handleQuery(question) {
  const block = createBlock('response');

  // Thinking indicator — visible until the first token arrives
  const thinkEl = _makeThinkingEl();
  block.appendChild(thinkEl);
  scrollToBottom();

  const pre = document.createElement('pre');
  pre.className = 'response-text streaming';

  // Blinking cursor that lives inside the response block while streaming
  const streamCursor = document.createElement('span');
  streamCursor.className = 'stream-cursor';
  streamCursor.textContent = '█';

  let firstToken = true;
  let sources    = [];

  await streamSSE(
    API.QUERY,
    { question, conversation_id: state.conversationId },
    async (token) => {
      if (firstToken) {
        thinkEl.remove();           // swap out the dots
        block.appendChild(pre);
        block.appendChild(streamCursor);
        firstToken = false;
      }
      pre.textContent += token;
      scrollToBottom();
      await sleep(20);
    },
    (data) => { sources = data.sources || []; },
    (note) => {
      // Show contextual rewrite note above the response
      const noteEl = document.createElement('div');
      noteEl.className = 'context-note';
      noteEl.textContent = '↻ ' + note;
      block.insertBefore(noteEl, thinkEl);
      scrollToBottom();
    }
  );

  streamCursor.remove();          // cursor gone when done
  pre.classList.remove('streaming');
  // If Ollama was offline the thinkEl may still be showing
  if (firstToken) thinkEl.remove();
  printSources(sources);

  if (window.VoiceManager && window.VoiceManager.ttsEnabled) {
    window.VoiceManager.speak(pre.textContent);
  }
}

// ── Command: /summarize <filename> ───────────────────────────
async function handleSummarize(args) {
  const filename = args.trim();
  if (!filename) {
    printError('Usage: /summarize <filename>');
    printInfo('Example: /summarize notes.pdf');
    return;
  }

  await refreshDocs(false);
  const doc = state.documents.find((d) => d.filename === filename);
  if (!doc) {
    const avail = state.documents.map((d) => d.filename).join(', ') || 'none uploaded';
    printError(`Document '${filename}' not found.`);
    printInfo('Available: ' + avail);
    return;
  }

  printInfo(`Summarising ${filename}…`);

  const block = createBlock('response');
  const thinkEl2 = _makeThinkingEl();
  block.appendChild(thinkEl2);
  scrollToBottom();

  const pre = document.createElement('pre');
  pre.className = 'response-text streaming';

  const streamCursor2 = document.createElement('span');
  streamCursor2.className = 'stream-cursor';
  streamCursor2.textContent = '█';

  let firstToken2 = true;

  await streamSSE(
    API.SUMMARIZE,
    { doc_id: doc.id },
    async (token) => {
      if (firstToken2) {
        thinkEl2.remove();
        block.appendChild(pre);
        block.appendChild(streamCursor2);
        firstToken2 = false;
      }
      pre.textContent += token;
      scrollToBottom();
      await sleep(20);
    },
    null
  );

  streamCursor2.remove();
  if (firstToken2) thinkEl2.remove();
  pre.classList.remove('streaming');
}

// ── Command: /upload ──────────────────────────────────────────
function handleUpload() {
  fileInput.value = '';
  fileInput.click();
}

// Upload one or many files sequentially
async function uploadFiles(files) {
  const allowed = [...files].filter((f) => {
    const ext = f.name.split('.').pop().toLowerCase();
    return ['pdf', 'txt', 'md'].includes(ext);
  });
  const rejected = [...files].filter((f) => !allowed.includes(f));

  for (const f of rejected) {
    const ext = f.name.split('.').pop();
    printError(`Rejected '${f.name}' — unsupported type .${ext}. Allowed: pdf, txt, md`);
  }

  for (const f of allowed) {
    await uploadSingleFile(f);
  }

  if (allowed.length > 0) {
    await refreshDocs(false);   // update cached list silently
    await updateStatusBar();    // refresh doc count in status bar
  }
}

async function uploadSingleFile(file) {
  const block  = createBlock('upload-block');
  const label  = document.createElement('div');
  label.textContent = `↑  Uploading ${file.name}…`;
  block.appendChild(label);

  const barEl = document.createElement('div');
  barEl.className = 'progress-bar';
  block.appendChild(barEl);

  // Animated fake progress bar (server-side is the bottleneck)
  let pct = 0;
  const ticker = setInterval(() => {
    pct = Math.min(pct + Math.random() * 14 + 3, 88);
    renderBar(barEl, pct);
  }, 160);

  const fail = (msg) => {
    clearInterval(ticker);
    label.textContent = `✗  ${file.name}: ${msg}`;
    label.className = 'error-text';
    barEl.remove();
    scrollToBottom();
  };

  try {
    const form = new FormData();
    form.append('file', file);
    const resp = await fetch(API.UPLOAD, { method: 'POST', body: form });

    // Flask returns plain HTML for 413, not JSON
    if (resp.status === 413) {
      return fail('File too large. Maximum upload size is 16 MB.');
    }

    // Guard against non-JSON responses (server errors, proxies, etc.)
    const contentType = resp.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      return fail(`Server error (HTTP ${resp.status}).`);
    }

    const data = await resp.json();
    clearInterval(ticker);
    renderBar(barEl, 100);

    if (data.status === 'success') {
      label.textContent = `✓  ${file.name}  —  ${data.chunks} chunks ingested`;
      label.className = 'success-text';
    } else {
      label.textContent = `✗  ${file.name}: ${data.message}`;
      label.className = 'error-text';
      barEl.remove();
    }
  } catch (err) {
    fail(err.message || 'Network error.');
    return;
  }

  scrollToBottom();
}

function renderBar(el, pct) {
  const filled = Math.round(pct / 5);         // 0–20 blocks
  const empty  = 20 - filled;
  el.textContent = '[' + '█'.repeat(filled) + '░'.repeat(empty) + '] ' + Math.floor(pct) + '%';
}

// ── Command: /docs, /ls ───────────────────────────────────────
async function refreshDocs(show = true) {
  try {
    const resp = await fetch(API.DOCUMENTS);
    const data = await resp.json();
    state.documents = data.documents || [];
    if (show) renderDocsTable(state.documents);
  } catch (err) {
    if (show) printError('Could not fetch documents: ' + err.message);
  }
}

function renderDocsTable(docs) {
  if (docs.length === 0) {
    printInfo([
      'No documents uploaded.',
      'Use /upload to choose files, or drag & drop PDF/TXT/MD files onto the terminal.',
    ].join('\n'));
    return;
  }

  // Column widths (content only, borders/padding added separately)
  const C_ID   = 10;
  const C_NAME = Math.min(30, Math.max(12, Math.max(...docs.map((d) => d.filename.length))));
  const C_CHK  = 6;
  const C_DATE = 8;

  // Inner row width = C_ID + 3 + C_NAME + 3 + C_CHK + 3 + C_DATE + 2
  // (each │ sep takes " │ " = 3 chars, outer │ takes "│ " = 2 left, " │" = 2 right)
  const innerW = C_ID + 3 + C_NAME + 3 + C_CHK + 3 + C_DATE;

  const h = (a, b, c, d) =>
    `├${'─'.repeat(a+2)}┼${'─'.repeat(b+2)}┼${'─'.repeat(c+2)}┼${'─'.repeat(d+2)}┤`;
  const row = (a, b, c, d) =>
    `│ ${a} │ ${b} │ ${c} │ ${d} │`;

  const top    = `┌${'─'.repeat(C_ID+2)}┬${'─'.repeat(C_NAME+2)}┬${'─'.repeat(C_CHK+2)}┬${'─'.repeat(C_DATE+2)}┐`;
  const banner = `│ UPLOADED DOCUMENTS${' '.repeat(innerW - 17)}│`;
  const divH   = h(C_ID, C_NAME, C_CHK, C_DATE);
  const header = row(
    padEnd('ID',       C_ID),
    padEnd('Filename', C_NAME),
    padEnd('Chunks',   C_CHK),
    padEnd('Date',     C_DATE)
  );
  const bot = `└${'─'.repeat(C_ID+2)}┴${'─'.repeat(C_NAME+2)}┴${'─'.repeat(C_CHK+2)}┴${'─'.repeat(C_DATE+2)}┘`;

  const dataRows = docs.map((d) =>
    row(
      padEnd(d.id.slice(0, C_ID),  C_ID),
      padEnd(d.filename,            C_NAME),
      String(d.chunks).padStart(C_CHK),
      padEnd((d.upload_date || '').slice(5), C_DATE)   // MM-DD
    )
  );

  const table = [top, banner, divH, header, divH, ...dataRows, bot].join('\n');
  print(table, 'doc-table');
}

// ── Command: /delete <filename> ───────────────────────────────
async function handleDelete(args) {
  const filename = args.trim();
  if (!filename) {
    printError('Usage: /delete <filename>');
    return;
  }

  await refreshDocs(false);
  const matches = state.documents.filter((d) => d.filename === filename);

  if (matches.length === 0) {
    const avail = state.documents.map((d) => d.filename).join(', ') || 'none uploaded';
    printError(`Document '${filename}' not found.`);
    printInfo('Uploaded: ' + avail);
    return;
  }

  for (const doc of matches) {
    try {
      const resp = await fetch(API.DOCUMENT(doc.id), { method: 'DELETE' });
      const data = await resp.json();
      if (data.status === 'deleted') {
        printSuccess(`Deleted '${filename}'  (id: ${doc.id.slice(0, 8)}…)`);
      } else {
        printError('Delete failed: ' + (data.message || 'unknown error'));
      }
    } catch (err) {
      printError('Delete error: ' + err.message);
    }
  }

  await refreshDocs(false);
  await updateStatusBar();
}

// ── Command: /status ──────────────────────────────────────────
async function handleStatus() {
  try {
    const resp = await fetch(API.HEALTH);
    const data = await resp.json();
    const ok = data.backend !== 'none';
    const lines = [
      `${ok ? '●' : '○'}  LLM Backend: ${(data.backend || 'none').toUpperCase()}`,
      `   Ollama : ${data.ollama  ? '✓ online'  : '✗ offline'}`,
      `   Groq   : ${data.groq    ? '✓ online'  : (data.groq === false ? '✗ offline / no key' : '—')}`,
      `   Model  : ${data.model  || '—'}`,
      `   Docs   : ${data.documents}`,
    ].join('\n');
    print(lines, ok ? 'success' : 'error');
    await updateStatusBar(data);
  } catch (err) {
    printError('Could not reach server: ' + err.message);
  }
}

// ── Command: /help ────────────────────────────────────────────
function showHelp() {
  const lines = [
    '┌──────────────────────┬──────────────────────────────────────────┐',
    '│  COMMAND             │  DESCRIPTION                             │',
    '├──────────────────────┼──────────────────────────────────────────┤',
    '│  <question>          │  Ask about your uploaded documents       │',
    '│  /upload             │  Choose files to upload (PDF/TXT/MD)     │',
    '│  /docs  /ls          │  List uploaded documents                 │',
    '│  /summarize <file>   │  Summarise a document                    │',
    '│  /delete <file>      │  Delete a document and its embeddings    │',
    '│  /status             │  Check Ollama connection & stats         │',
    '│  /voice              │  Toggle Speech-To-Text (or click MIC)    │',
    '│  /tts                │  Toggle Text-To-Speech (or click TTS)    │',
    '│  /clear              │  Clear terminal output                   │',
    '│  /help               │  Show this help                          │',
    '├──────────────────────┼──────────────────────────────────────────┤',
    '│  Drag & drop         │  Drop PDF/TXT/MD files onto terminal     │',
    '│  ↑ / ↓ arrows        │  Navigate command history                │',
    '└──────────────────────┴──────────────────────────────────────────┘',
  ].join('\n');
  print(lines, 'doc-table');
}

// ── Voice Integration ─────────────────────────────────────────

function handleVoiceToggle() {
  if (!window.VoiceManager) return;
  const isNowListening = window.VoiceManager.toggleListening(
    (interim, final) => {
      cmdInput.value = (final || interim).trim();
      syncCursor();
      if (final) {
        dispatch(cmdInput.value);
        cmdInput.value = '';
        syncCursor();
      }
    },
    () => {
      micToggle.textContent = '[MIC: OFF]';
      micToggle.classList.remove('active');
    }
  );
  
  if (isNowListening) {
    micToggle.textContent = '[MIC: ON\u25CF]';
    micToggle.classList.add('active');
    if (voiceBtn) voiceBtn.classList.add('listening');
    printInfo("Listening for voice commands...");
  } else {
    micToggle.textContent = '[MIC: OFF]';
    micToggle.classList.remove('active');
    if (voiceBtn) voiceBtn.classList.remove('listening');
  }
}

function handleTtsToggle() {
  if (!window.VoiceManager) return;
  const isEnabled = window.VoiceManager.toggleTTS();
  if (isEnabled) {
    ttsToggle.textContent = '[TTS: ON]';
    ttsToggle.classList.add('active');
    printSuccess("Voice feedback enabled.");
  } else {
    ttsToggle.textContent = '[TTS: OFF]';
    ttsToggle.classList.remove('active');
    printInfo("Voice feedback disabled.");
    window.VoiceManager.stopSpeaking();
  }
}

// ── Command: /clear ───────────────────────────────────────────
function clearOutput() {
  output.innerHTML = '';
  renderHeader();
}

// ── Command dispatcher ────────────────────────────────────────
async function dispatch(input) {
  printCmd(input);

  const trimmed   = input.trimStart();
  const firstWord = trimmed.split(/\s+/)[0].toLowerCase();
  const rest      = trimmed.slice(firstWord.length).trimStart();

  switch (firstWord) {
    case '/help':      showHelp();                  break;
    case '/docs':
    case '/ls':        await refreshDocs(true);     break;
    case '/upload':    handleUpload();              break;
    case '/clear':     clearOutput();               break;
    case '/status':    await handleStatus();        break;
    case '/voice':     handleVoiceToggle();         break;
    case '/tts':       handleTtsToggle();           break;
    case '/summarize': await handleSummarize(rest); break;
    case '/delete':    await handleDelete(rest);    break;
    default:
      if (firstWord.startsWith('/')) {
        printError(`Unknown command: '${firstWord}'`);
        printInfo('Type /help to see available commands.');
      } else {
        await handleQuery(input);
      }
  }
}

// ── Input handler (Enter + arrow history) ─────────────────────
function handleKeydown(e) {
  if (state.streaming) {
    // Only allow Ctrl+C style abort (future enhancement) during stream
    return;
  }

  if (e.key === 'Enter') {
    const cmd = cmdInput.value.trim();
    if (!cmd) return;

    // Push to history (no duplicates at front)
    if (state.cmdHistory[0] !== cmd) {
      state.cmdHistory.unshift(cmd);
      if (state.cmdHistory.length > 200) state.cmdHistory.pop();
    }
    state.historyIdx = -1;

    cmdInput.value = '';
    syncCursor();
    dispatch(cmd);

  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    const next = state.historyIdx + 1;
    if (next < state.cmdHistory.length) {
      state.historyIdx = next;
      cmdInput.value = state.cmdHistory[next];
      // Move caret to end
      setTimeout(() => cmdInput.setSelectionRange(cmdInput.value.length, cmdInput.value.length), 0);
      syncCursor();
    }

  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    const prev = state.historyIdx - 1;
    if (prev >= 0) {
      state.historyIdx = prev;
      cmdInput.value = state.cmdHistory[prev];
    } else {
      state.historyIdx = -1;
      cmdInput.value = '';
    }
    syncCursor();
  }
}

// ── Cursor sync on input ──────────────────────────────────────
function handleInputChange() {
  syncCursor();
}

// ── Drag & drop ───────────────────────────────────────────────
function setupDragDrop() {
  // Use a depth counter so that moving between child elements doesn't
  // flicker the overlay (each dragenter increments, dragleave decrements).
  let dragDepth = 0;

  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragDepth++;
    dropZone.classList.add('active');
  });

  document.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  });

  document.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) dropZone.classList.remove('active');
  });

  document.addEventListener('drop', async (e) => {
    e.preventDefault();
    dragDepth = 0;
    dropZone.classList.remove('active');
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length) {
      printCmd('[drag & drop upload]');
      await uploadFiles(files);
    }
  });

  // File input (triggered by /upload command)
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      uploadFiles([...fileInput.files]);
      fileInput.value = '';   // reset so same file can be re-uploaded
    }
  });
}

// ── Status bar ────────────────────────────────────────────────
function _setBadge(data) {
  // Build DOM safely — no innerHTML
  ollamaBadge.textContent = '';

  const dot = document.createElement('span');
  dot.className = 'status-dot';
  dot.textContent = '●';

  const backend = data?.backend || 'none';
  const online  = backend !== 'none';

  let label;
  if (backend === 'ollama') label = 'OLLAMA';
  else if (backend === 'groq') label = 'GROQ';
  else label = 'OFFLINE';

  ollamaBadge.appendChild(document.createTextNode('[LLM: '));
  ollamaBadge.appendChild(dot);
  ollamaBadge.appendChild(document.createTextNode(`${label}]`));
  ollamaBadge.className = online ? 'online' : 'offline';
}

async function updateStatusBar(data) {
  if (!data) {
    try {
      const resp = await fetch(API.HEALTH);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      data = await resp.json();
    } catch {
      _setBadge(null);
      return;
    }
  }

  _setBadge(data);
  modelBadge.textContent = `MODEL: ${data.model || '—'} | DOCS: ${data.documents ?? '—'}`;
}

function pollStatus() {
  updateStatusBar(null).catch(() => {});
}

// ── Clock ─────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2, '0');
  const m = String(now.getMinutes()).padStart(2, '0');
  const s = String(now.getSeconds()).padStart(2, '0');
  clockEl.textContent = `${h}:${m}:${s}`;
}

// ── Header render ─────────────────────────────────────────────
function renderHeader() {
  const header = document.getElementById('header');
  header.innerHTML = '';

  const pre = document.createElement('pre');
  pre.className = 'ascii-header';
  // Box-drawing characters preserved exactly
  pre.textContent = [
    '╔══════════════════════════════════════════════════════╗',
    '║  █████╗ ██╗   ██╗██████╗  █████╗                     ║',
    '║ ██╔══██╗██║   ██║██╔══██╗██╔══██╗                    ║',
    '║ ███████║██║   ██║██████╔╝███████║                    ║',
    '║ ██╔══██║██║   ██║██╔══██╗██╔══██║                    ║',
    '║ ██║  ██║╚██████╔╝██║  ██║██║  ██║   AURA TERMINAL    ║',
    '║ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   Gemma 4 · Chroma ║',
    '╚══════════════════════════════════════════════════════╝',
  ].join('\n');
  header.appendChild(pre);
}

function printWelcome() {
  printInfo([
    'Type a question to query your documents, or try a command:',
    '  /upload   — add a document (PDF, TXT, MD)',
    '  /help     — see all commands',
    '  /status   — check Ollama connection',
  ].join('\n'));
}

// ── Focus helpers ─────────────────────────────────────────────
function refocusInput() {
  if (!state.streaming) cmdInput.focus();
}

// ── Onboarding Logic ──────────────────────────────────────────
async function startOnboarding() {
  const overlay = document.getElementById('onboarding-overlay');
  const skipBtn = document.getElementById('skip-onboarding');
  
  let onboardingDone = false;

  const finish = () => {
    if (onboardingDone) return;
    onboardingDone = true;
    overlay.classList.add('hidden');
    
    // Resume app initialization
    printWelcome();
    cmdInput.focus();
    
    // Remove from DOM after transition
    setTimeout(() => overlay.remove(), 800);
  };

  skipBtn.addEventListener('click', finish);

  // Auto-finish after 7 seconds
  setTimeout(finish, 7500); // 7.5s to account for last animation
}

// ── Initialisation ────────────────────────────────────────────
function init() {
  renderHeader();
  
  // Delay welcome and focus for onboarding
  startOnboarding();

  // Input events
  cmdInput.addEventListener('keydown', handleKeydown);
  cmdInput.addEventListener('input',   handleInputChange);
  // keyup catches arrow keys, Home/End; click handles mouse repositioning
  cmdInput.addEventListener('keyup',   syncCursor);
  cmdInput.addEventListener('click',   syncCursor);

  // Clicking anywhere in the terminal re-focuses input
  output.addEventListener('click', refocusInput);
  document.getElementById('header').addEventListener('click', refocusInput);

  setupDragDrop();

  // Clock — update every second
  updateClock();
  setInterval(updateClock, 1000);

  // Status bar — initial + poll every 30 s
  pollStatus();
  setInterval(pollStatus, 30_000);

  if (micToggle && ttsToggle) {
    micToggle.addEventListener('click', handleVoiceToggle);
    ttsToggle.addEventListener('click', handleTtsToggle);
  }

  // Input-bar action buttons
  if (voiceBtn) voiceBtn.addEventListener('click', handleVoiceToggle);
  if (uploadBtn) uploadBtn.addEventListener('click', handleUpload);

  // Pre-load the document list silently so /summarize and /delete work
  refreshDocs(false);
}

document.addEventListener('DOMContentLoaded', init);
