function StatusLine({ status, decision, errorMessage }) {
  if (status === 'submitting') {
    return (
      <div className="flex items-center gap-2 text-[12px] text-ink-2">
        <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-[#d7c6ba] border-t-clay animate-spin" />
        <span>{decision === 'approved' ? '正在确认并发送…' : '正在提交拒绝…'}</span>
      </div>
    );
  }

  if (status === 'success') {
    return (
      <span className={`chip ${decision === 'approved' ? 'chip-ok' : 'chip-err'}`}>
        {decision === 'approved' ? '✓ 已发送' : '✓ 已拒绝'}
      </span>
    );
  }

  if (status === 'error') {
    return (
      <div className="text-[12px] text-err">
        {errorMessage || '提交失败，请重试'}
      </div>
    );
  }

  return <span className="text-[11px] text-ink-3">等待你的确认</span>;
}

export default function InterruptCard({ data, status, decision, errorMessage, onResolve }) {
  const isPending = status === 'pending' || status === 'error';
  const isSubmitting = status === 'submitting';

  const handle = async (approved) => {
    if (!isPending || isSubmitting) return;
    await onResolve(approved, data.send_token);
  };

  return (
    <div className="card mb-3 overflow-hidden border-[#f0d9b8] bg-[#fdf8ee]">
      <div className="px-4 py-2.5 border-b border-[#f0e2bf] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-warn">⏸</span>
          <span className="text-[13px] font-medium text-ink">发送前确认</span>
          <span className="chip chip-warn font-mono">prepare_send</span>
        </div>
        <span className="font-mono text-[10.5px] text-ink-3">
          token {(data.send_token || '').slice(0, 8)}…
        </span>
      </div>
      <div className="px-4 py-4">
        {(data.files || []).map((f, i) => (
          <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-white border border-line mb-2">
            <div className="w-9 h-9 rounded-md bg-clay-soft flex items-center justify-center text-clay text-[11px] font-semibold shrink-0">
              {(f.name || '').split('.').pop().toUpperCase().slice(0, 3)}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[13.5px] truncate">{f.name}</div>
              <div className="font-mono text-[11px] text-ink-3 truncate mt-0.5">
                {f.full_path}
              </div>
              <div className="text-[11px] text-ink-3 mt-1">
                {f.size_display} · 最后修改时间: {f.modified ?? 'null'}
              </div>
            </div>
          </div>
        ))}
        <div className="flex items-end justify-between gap-3 mt-3">
          <div className="min-h-[20px] flex items-center">
            <StatusLine status={status} decision={decision} errorMessage={errorMessage} />
          </div>
          {isPending ? (
            <>
              <div className="flex items-center gap-2">
                <button
                  disabled={isSubmitting}
                  onClick={() => handle(true)}
                  className="px-4 py-2 rounded-lg bg-clay text-white text-[13px] hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                  批准并发送
                </button>
                <button
                  disabled={isSubmitting}
                  onClick={() => handle(false)}
                  className="px-4 py-2 rounded-lg border border-line text-[13px] text-ink-2 hover:bg-bg disabled:opacity-50"
                >
                  拒绝
                </button>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
