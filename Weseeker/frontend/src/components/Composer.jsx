import { useRef, useEffect } from 'react';

export default function Composer({ value, onChange, onSend, disabled }) {
  const taRef = useRef(null);

  // autosize
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, [value]);

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!disabled) onSend();
    }
  };

  return (
    <div className="flex-shrink-0 bg-gradient-to-t from-bg via-bg to-transparent pt-4 pb-5">
      <div className="max-w-[760px] mx-auto px-8">
        <div className="card flex items-end gap-2 p-2 pl-4 shadow-sm">
          <textarea
            ref={taRef}
            rows={1}
            value={value}
            onChange={e => onChange(e.target.value)}
            onKeyDown={onKey}
            placeholder="输入指令 (e.g. 查找本周关于大模型的会议记录并预览)"
            className="flex-1 resize-none bg-transparent outline-none text-[14.5px] py-2 leading-[1.6] placeholder:text-ink-3 max-h-[200px]"
          />
          <button
            onClick={onSend}
            disabled={disabled || !value.trim()}
            className="w-9 h-9 rounded-lg bg-clay text-white flex items-center justify-center hover:opacity-90 disabled:opacity-30 disabled:cursor-not-allowed transition-opacity"
          >
            →
          </button>
        </div>
        <div className="text-[11px] text-ink-3 mt-2 px-1">
          Enter 发送 · Shift+Enter 换行
        </div>
      </div>
    </div>
  );
}
