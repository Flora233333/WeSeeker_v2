// SSE 客户端：fetch + ReadableStream（支持 POST body）
export async function* sseStream(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`HTTP ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of block.split('\n')) {
          if (line.startsWith('data:')) {
            const payload = line.slice(5).trim();
            if (payload) {
              try { yield JSON.parse(payload); } catch (e) { /* skip */ }
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export const api = {
  chat: (text, threadId) => sseStream('/api/chat', { text, thread_id: threadId }),
  resume: (threadId, sendToken, approved) =>
    sseStream('/api/resume', { thread_id: threadId, send_token: sendToken, approved }),
  newThread: async () => {
    const resp = await fetch('/api/new_thread', { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  },
  clearThread: async (threadId) => {
    const resp = await fetch('/api/clear_thread', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: threadId }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  },
};
