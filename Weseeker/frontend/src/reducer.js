// 全局状态 reducer
export const initialState = {
  messages: [],          // [{role, id?, text, reasoning?, tools?, interrupt?, interruptStatus?, interruptDecision?, interruptError?}]
  events: [],            // 原始事件时间线 [{type, ts, ...}]
  threadId: null,
  running: false,
  usage: null,           // {input, output, reasoning}
  toolCount: 0,
  error: null,
  hitlSubmittingToken: null,
  pendingInterrupt: null,
};

const findLastAssistantIndex = (msgs) => {
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') return i;
  }
  return -1;
};

export function reducer(state, ev) {
  // 所有事件都进时间线
  const events = ev.type === 'clear' || ev.type === 'new_thread' ? [] : [...state.events, ev];

  switch (ev.type) {
    case 'user_local': {
      return { ...state, events,
        messages: [...state.messages, { role: 'user', text: ev.text }] };
    }
    case 'run_started':
      return { ...state, events, threadId: ev.thread_id, running: true, error: null };
    case 'run_finished': {
      const pendingInterrupt = state.pendingInterrupt;
      if (!pendingInterrupt) {
        return { ...state, events, running: false };
      }

      const msgs = [...state.messages];
      const lastAssistantIndex = findLastAssistantIndex(msgs);
      if (lastAssistantIndex >= 0) {
        const last = msgs[lastAssistantIndex];
        msgs[lastAssistantIndex] = {
          ...last,
          interrupt: pendingInterrupt,
          interruptStatus: 'pending',
          interruptDecision: null,
          interruptError: null,
        };
      } else {
        msgs.push({
          role: 'assistant',
          reasoning: '',
          text: '',
          tools: [],
          interrupt: pendingInterrupt,
          interruptStatus: 'pending',
          interruptDecision: null,
          interruptError: null,
        });
      }

      return { ...state, events, running: false, messages: msgs, pendingInterrupt: null };
    }
    case 'message_start': {
      return { ...state, events,
        messages: [...state.messages, {
          role: 'assistant', id: ev.message_id,
          reasoning: '', text: '', tools: [], interrupt: null,
          interruptStatus: null, interruptDecision: null, interruptError: null,
        }] };
    }
    case 'reasoning_delta':
    case 'assistant_delta': {
      const msgs = [...state.messages];
      const lastIndex = findLastAssistantIndex(msgs);
      if (lastIndex >= 0) {
        const last = msgs[lastIndex];
        msgs[lastIndex] = {
          ...last,
          reasoning: ev.type === 'reasoning_delta'
            ? (last.reasoning || '') + ev.text
            : last.reasoning,
          text: ev.type === 'assistant_delta'
            ? (last.text || '') + ev.text
            : last.text,
        };
      }
      return { ...state, events, messages: msgs };
    }
    case 'tool_call': {
      const msgs = [...state.messages];
      const lastIndex = findLastAssistantIndex(msgs);
      if (lastIndex >= 0) {
        const last = msgs[lastIndex];
        msgs[lastIndex] = {
          ...last,
          tools: [...(last.tools || []), {
            id: ev.call_id, name: ev.name, args: ev.args,
            result: null, ok: null, summary: null,
          }],
        };
      }
      return { ...state, events, messages: msgs, toolCount: state.toolCount + 1 };
    }
    case 'tool_result': {
      const msgs = state.messages.map(m => {
        if (m.role !== 'assistant' || !m.tools) return m;
        return { ...m, tools: m.tools.map(t =>
          t.id === ev.call_id ? { ...t, result: ev.raw, ok: ev.ok, summary: ev.summary } : t
        ) };
      });
      return { ...state, events, messages: msgs };
    }
    case 'interrupt': {
      return { ...state, events, pendingInterrupt: ev };
    }
    case 'interrupt_submit_start': {
      const msgs = state.messages.map(m =>
        m.interrupt && m.interrupt.send_token === ev.send_token
          ? {
            ...m,
            interruptStatus: 'submitting',
            interruptDecision: ev.approved ? 'approved' : 'rejected',
            interruptError: null,
          }
          : m
      );
      return { ...state, events, messages: msgs, hitlSubmittingToken: ev.send_token };
    }
    case 'interrupt_submit_success': {
      const msgs = state.messages.map(m =>
        m.interrupt && m.interrupt.send_token === ev.send_token
          ? {
            ...m,
            interruptStatus: 'success',
            interruptDecision: ev.approved ? 'approved' : 'rejected',
            interruptError: null,
          }
          : m
      );
      return { ...state, events, messages: msgs, hitlSubmittingToken: null };
    }
    case 'interrupt_submit_error': {
      const msgs = state.messages.map(m =>
        m.interrupt && m.interrupt.send_token === ev.send_token
          ? {
            ...m,
            interruptStatus: 'error',
            interruptDecision: ev.approved ? 'approved' : 'rejected',
            interruptError: ev.message,
          }
          : m
      );
      return { ...state, events, messages: msgs, hitlSubmittingToken: null };
    }
    case 'usage':
      return { ...state, events,
        usage: { input: ev.input, output: ev.output, reasoning: ev.reasoning } };
    case 'error':
      return { ...state, events, error: ev.message };
    case 'thread_cleared':
      return { ...initialState, threadId: ev.thread_id };
    case 'new_thread':
      return { ...initialState, threadId: ev.thread_id };
    case 'clear':
      return { ...initialState };
    default:
      return { ...state, events };
  }
}
