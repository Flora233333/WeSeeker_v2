import Collapsible from './Collapsible.jsx';

export default function ToolBlock({ tool }) {
  const status = tool.ok === null ? 'running' : tool.ok ? 'success' : 'failed';
  const chipCls = status === 'success' ? 'chip-ok' : status === 'failed' ? 'chip-err' : 'chip-warn';
  const dotCls = status === 'success' ? 'bg-ok' : status === 'failed' ? 'bg-err' : 'bg-warn pulse-dot';

  const title = (
    <span className="font-mono text-[13px] text-ink flex items-center gap-2">
      <span className={`w-1.5 h-1.5 rounded-full ${dotCls}`} />
      {tool.name}
      {tool.summary && <span className="text-ink-3 text-[11px] font-sans">· {tool.summary}</span>}
    </span>
  );
  const badge = <span className={`chip ${chipCls} ml-2`}>{status}</span>;

  return (
    <Collapsible title={title} badge={badge} defaultOpen={false}>
      <div className="px-4 py-3 space-y-2">
        <div className="text-[10.5px] text-ink-3 font-mono uppercase tracking-wider">// args</div>
        <pre className="json-block">{JSON.stringify(tool.args, null, 2)}</pre>
        {tool.result !== null && (
          <>
            <div className="text-[10.5px] text-ink-3 font-mono uppercase tracking-wider mt-3">// result</div>
            <pre className="json-block">{JSON.stringify(tool.result, null, 2)}</pre>
          </>
        )}
      </div>
    </Collapsible>
  );
}
