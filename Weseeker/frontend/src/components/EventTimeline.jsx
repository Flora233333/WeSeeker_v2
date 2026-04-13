import { useMemo, useState } from 'react';

const TYPE_COLOR = {
  user_local:        'border-clay',
  message_start:     'border-ink-3',
  reasoning_delta:   'border-ink-3',
  assistant_delta:   'border-ink-3',
  tool_call:         'border-ok',
  tool_result:       'border-ok bg-ok',
  interrupt:         'border-warn bg-warn pulse-dot',
  interrupt_resolved:'border-clay bg-clay',
  usage:             'border-ink-3',
  error:             'border-err bg-err',
  run_started:       'border-clay bg-clay',
  run_finished:      'border-ink-3 bg-ink-3',
};

const TYPE_LABEL = {
  user_local: 'user_message',
  reasoning_delta: 'reasoning',
  assistant_delta: 'assistant',
  interrupt_resolved: 'interrupt_done',
};

// 把高频 delta 折叠成单行计数
function collapseDeltas(events) {
  const out = [];
  let buf = null;
  for (const ev of events) {
    if (ev.type === 'reasoning_delta' || ev.type === 'assistant_delta') {
      if (buf && buf.type === ev.type) {
        buf.count += 1;
        buf.chars += (ev.text || '').length;
        if (typeof ev.ts === 'number') {
          buf.ts = ev.ts;
          buf.endTs = ev.ts;
        }
        continue;
      }
      buf = {
        ...ev,
        count: 1,
        chars: (ev.text || '').length,
        startTs: typeof ev.ts === 'number' ? ev.ts : null,
        endTs: typeof ev.ts === 'number' ? ev.ts : null,
      };
      out.push(buf);
    } else {
      out.push(ev); buf = null;
    }
  }
  return out;
}

function fmtTime(ts) {
  if (typeof ts !== 'number' || Number.isNaN(ts)) return '--:--:--.---';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return '--:--:--.---';
  return d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3, '0');
}

function fmtDuration(ms) {
  if (typeof ms !== 'number' || Number.isNaN(ms)) return null;
  return `${Math.max(0, Math.round(ms))} ms`;
}

function addDurations(events) {
  const toolResultTsByCallId = new Map();

  for (const ev of events) {
    if (ev.type === 'tool_result' && ev.call_id && typeof ev.ts === 'number' && !toolResultTsByCallId.has(ev.call_id)) {
      toolResultTsByCallId.set(ev.call_id, ev.ts);
    }
  }

  return events.map((ev) => {
    if (ev.type === 'reasoning_delta' || ev.type === 'assistant_delta') {
      if (typeof ev.startTs === 'number' && typeof ev.endTs === 'number') {
        return { ...ev, durationMs: Math.max(0, ev.endTs - ev.startTs) };
      }
      return ev;
    }

    if (ev.type === 'tool_call') {
      const endTs = toolResultTsByCallId.get(ev.call_id);
      if (typeof endTs === 'number' && typeof ev.ts === 'number') {
        return { ...ev, durationMs: Math.max(0, endTs - ev.ts) };
      }
      return { ...ev, durationMs: null, durationLabel: 'running' };
    }

    return ev;
  });
}

function detail(ev) {
  switch (ev.type) {
    case 'user_local':       return ev.text;
    case 'reasoning_delta':  return `${ev.chars} chars · ×${ev.count}`;
    case 'assistant_delta':  return `${ev.chars} chars · ×${ev.count}`;
    case 'tool_call':        return `${ev.name}(${Object.keys(ev.args || {}).join(', ')})`;
    case 'tool_result':      return `${ev.name} · ${ev.summary || (ev.ok ? 'ok' : 'fail')}`;
    case 'interrupt':        return `${ev.action} · awaiting`;
    case 'interrupt_resolved':
      return `${ev.status}${ev.message ? ` · ${ev.message}` : ''}`;
    case 'usage':            return `in ${ev.input} · out ${ev.output} · r ${ev.reasoning || 0}`;
    case 'error':            return ev.message;
    default:                 return '';
  }
}

export default function EventTimeline({ events, onClear, clearDisabled = false }) {
  const [filter, setFilter] = useState('all');
  const view = useMemo(() => {
    const collapsed = addDurations(collapseDeltas(events));
    if (filter === 'all') return collapsed;
    if (filter === 'tools') return collapsed.filter(e => e.type === 'tool_call' || e.type === 'tool_result');
    if (filter === 'flow') return collapsed.filter(e => !['reasoning_delta', 'assistant_delta'].includes(e.type));
    return collapsed;
  }, [events, filter]);

  return (
    <aside className="bg-bg-2 overflow-y-auto scroll-y h-full flex flex-col">
      <div className="px-4 py-3 flex items-center justify-between sticky top-0 bg-bg-2/95 backdrop-blur z-10 flex-shrink-0">
        <div className="text-[11px] tracking-wider text-ink-3 uppercase">Event Stream</div>
        <div className="flex items-center gap-1.5">
          {['all', 'flow', 'tools'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`chip ${filter === f ? 'chip-warn' : ''}`}
            >{f}</button>
          ))}
          <button
            onClick={onClear}
            disabled={clearDisabled}
            className="chip hover:bg-clay-soft disabled:opacity-40 disabled:cursor-not-allowed"
          >
            clear thread
          </button>
        </div>
      </div>
      <ol className="relative py-3 flex-1">
        <div className="absolute left-[26px] top-3 bottom-3 w-px timeline-line" />
        {view.length === 0 && (
          <li className="px-4 py-8 text-center text-ink-3 text-[12px]">暂无事件</li>
        )}
        {view.map((ev, i) => {
          const cls = TYPE_COLOR[ev.type] || 'border-ink-3';
          const filled = cls.includes('bg-');
          return (
            <li key={i} className="pl-12 pr-4 py-2 row-hover relative">
              <div className={`absolute left-[22px] top-3 w-2.5 h-2.5 rounded-full border-2 ${cls} ${filled ? '' : 'bg-white'}`} />
              <div className="font-mono text-[10.5px] text-ink-3">{fmtTime(ev.ts)}</div>
              <div className="text-[12.5px] flex items-center justify-between gap-3">
                <span className="font-mono">{TYPE_LABEL[ev.type] || ev.type}</span>
                {(ev.durationLabel || fmtDuration(ev.durationMs)) && (
                  <span className="font-mono text-[10.5px] text-ink-3">
                    {ev.durationLabel || fmtDuration(ev.durationMs)}
                  </span>
                )}
              </div>
              <div className="text-[11px] text-ink-3 truncate">{detail(ev)}</div>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}
