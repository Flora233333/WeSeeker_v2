import { useState } from 'react';

export default function Collapsible({ title, badge, meta, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="card mb-3 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-2.5 flex items-center justify-between text-[12.5px] text-ink-2 hover:bg-[#fbfaf7] transition-colors"
      >
        <span className="flex items-center gap-2 min-w-0">
          <span className="text-clay shrink-0 w-3 inline-block">{open ? '▾' : '▸'}</span>
          <span className="truncate">{title}</span>
          {badge}
        </span>
        <span className="text-ink-3 text-[11px] shrink-0 ml-2">{meta}</span>
      </button>
      {open && <div className="border-t border-line">{children}</div>}
    </div>
  );
}
