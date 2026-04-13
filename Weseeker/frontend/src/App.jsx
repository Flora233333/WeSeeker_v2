import { useReducer, useState, useCallback } from 'react';
import { reducer, initialState } from './reducer.js';
import { api } from './api.js';
import Conversation from './components/Conversation.jsx';
import Composer from './components/Composer.jsx';
import EventTimeline from './components/EventTimeline.jsx';

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [input, setInput] = useState('');

  const consumeStream = useCallback(async (gen, { expectResolution = false } = {}) => {
    let streamError = null;
    let resolution = null;
    try {
      for await (const ev of gen) {
        if (ev.type === 'error') streamError = ev.message || 'Unknown error';
        if (ev.type === 'interrupt_resolved') resolution = ev;
        dispatch(ev);
      }
    } catch (e) {
      streamError = String(e.message || e);
      dispatch({ type: 'error', ts: Date.now(), message: streamError });
      dispatch({ type: 'run_finished', ts: Date.now(), reason: 'error' });
    }

    if (expectResolution) {
      if (resolution?.status === 'confirmed' || resolution?.status === 'rejected') {
        return { ok: true, resolution };
      }

      return {
        ok: false,
        resolution,
        error: resolution?.message || streamError || '未收到本次确认操作的明确终态。',
      };
    }

    return streamError
      ? { ok: false, error: streamError, resolution }
      : { ok: true, resolution };
  }, []);

  const send = useCallback(async () => {
    const t = input.trim();
    if (!t || state.running) return;
    setInput('');
    dispatch({ type: 'user_local', ts: Date.now(), text: t });
    await consumeStream(api.chat(t, state.threadId));
  }, [input, state.running, state.threadId, consumeStream]);

  const onResolveInterrupt = useCallback(async (approved, sendToken) => {
    dispatch({ type: 'interrupt_submit_start', ts: Date.now(), send_token: sendToken, approved });
    const result = await consumeStream(
      api.resume(state.threadId, sendToken, approved),
      { expectResolution: true },
    );
    if (result.ok) {
      dispatch({ type: 'interrupt_submit_success', ts: Date.now(), send_token: sendToken, approved });
    } else {
      dispatch({
        type: 'interrupt_submit_error',
        ts: Date.now(),
        send_token: sendToken,
        approved,
        message: result.error,
      });
    }
  }, [state.threadId, consumeStream]);

  const clearThread = useCallback(async () => {
    if (state.running || !state.threadId) return;
    try {
      await api.clearThread(state.threadId);
      dispatch({ type: 'thread_cleared', thread_id: state.threadId, ts: Date.now() });
    } catch (e) {
      dispatch({ type: 'error', ts: Date.now(), message: String(e.message || e) });
    }
  }, [state.running, state.threadId]);

  const newThread = useCallback(async () => {
    if (state.running) return;
    try {
      const { thread_id: threadId } = await api.newThread();
      dispatch({ type: 'new_thread', thread_id: threadId, ts: Date.now() });
    } catch (e) {
      dispatch({ type: 'error', ts: Date.now(), message: String(e.message || e) });
    }
  }, [state.running]);

  return (
    <div className="h-full flex flex-col bg-bg">
      {/* Header */}
      <header className="flex-shrink-0 h-14 px-6 flex items-center justify-between border-b border-line bg-bg">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-clay flex items-center justify-center text-white text-[13px] font-semibold">W</div>
          <div className="font-serif text-[19px] tracking-tight">
            WeSeeker <span className="text-ink-3 font-normal">Debug</span>
          </div>
          <span className="chip">
            <span className={`w-1.5 h-1.5 rounded-full ${state.running ? 'bg-warn pulse-dot' : 'bg-ok'}`} />
            {state.running ? 'running' : 'ready'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="chip font-mono">deepseek-v3.2</span>
          {state.threadId && (
            <span className="chip font-mono">thread {state.threadId.slice(0, 8)}…</span>
          )}
          {state.usage && (
            <span className="chip">
              tokens {state.usage.input + state.usage.output}
              {state.usage.reasoning > 0 && ` · r ${state.usage.reasoning}`}
            </span>
          )}
          <span className="chip">tools {state.toolCount}</span>
          <button onClick={newThread} className="chip hover:bg-clay-soft">New thread</button>
        </div>
      </header>

      {/* Three columns */}
      <div className="flex-1 min-h-0 grid grid-cols-[260px_minmax(0,1fr)_360px] bg-bg-2">

        {/* Left: thread list (placeholder, single-user 本地版仅显示当前) */}
        <aside className="bg-bg-2 p-4 overflow-y-auto scroll-y">
          <div className="text-[11px] tracking-wider text-ink-3 mb-3 uppercase">Threads</div>
          {state.threadId ? (
            <div className="px-3 py-2 rounded-lg bg-clay-soft border border-[#ecd9cd]">
              <div className="text-[13px] truncate">当前会话</div>
              <div className="font-mono text-[10.5px] text-ink-3 mt-0.5">
                {state.threadId.slice(0, 12)}…
              </div>
              <div className="text-[10.5px] text-ink-3 mt-0.5">
                {state.messages.length} 条消息 · {state.toolCount} 次工具调用
              </div>
            </div>
          ) : (
            <div className="text-[12px] text-ink-3 px-3 py-2">尚未开始对话</div>
          )}
        </aside>

        {/* Center: conversation + composer (独立 flex column 让 composer 沉底) */}
        <main className="flex flex-col min-h-0 min-w-0 bg-bg">
          <Conversation
            messages={state.messages}
            onResolveInterrupt={onResolveInterrupt}
            error={state.error}
            running={state.running}
          />
          <Composer
            value={input}
            onChange={setInput}
            onSend={send}
            disabled={state.running}
          />
        </main>

        {/* Right: event timeline */}
        <EventTimeline
          events={state.events}
          onClear={clearThread}
          clearDisabled={state.running || !state.threadId}
        />
      </div>
    </div>
  );
}
