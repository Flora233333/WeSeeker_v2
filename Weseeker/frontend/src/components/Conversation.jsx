import { useEffect, useLayoutEffect, useRef } from 'react';
import Collapsible from './Collapsible.jsx';
import Markdown from './Markdown.jsx';
import ToolBlock from './ToolBlock.jsx';
import InterruptCard from './InterruptCard.jsx';

const SCROLL_DURATION_MS = 680;
const BOTTOM_THRESHOLD_PX = 4;

function easeOutQuint(t) {
  return 1 - Math.pow(1 - t, 5);
}

function isAtBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_THRESHOLD_PX;
}

function stickToBottom(el, programmaticScrollRef) {
  programmaticScrollRef.current = true;
  el.scrollTop = el.scrollHeight;
  requestAnimationFrame(() => {
    programmaticScrollRef.current = false;
  });
}

function animateScrollToBottom(el, frameRef, programmaticScrollRef) {
  const startTop = el.scrollTop;
  const initialTargetTop = Math.max(el.scrollHeight - el.clientHeight, 0);
  if (initialTargetTop <= startTop) {
    stickToBottom(el, programmaticScrollRef);
    return;
  }

  if (frameRef.current) {
    cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
  }

  programmaticScrollRef.current = true;
  const startAt = performance.now();
  const tick = (now) => {
    const progress = Math.min((now - startAt) / SCROLL_DURATION_MS, 1);
    const eased = easeOutQuint(progress);
    const targetTop = Math.max(el.scrollHeight - el.clientHeight, 0);
    el.scrollTop = startTop + (targetTop - startTop) * eased;

    if (progress < 1) {
      frameRef.current = requestAnimationFrame(tick);
    } else {
      frameRef.current = null;
      programmaticScrollRef.current = false;
    }
  };

  frameRef.current = requestAnimationFrame(tick);
}

function UserBubble({ text }) {
  return (
    <div className="flex justify-end mb-8">
      <div className="max-w-[78%] bg-panel border border-line text-ink rounded-[16px] rounded-tr-[14px] rounded-br-[14px] rounded-bl-[4px] px-5 py-3 text-[15px] leading-relaxed whitespace-pre-wrap">
        {text}
      </div>
    </div>
  );
}

function isInterruptVisible(msg, running) {
  return Boolean(msg.interrupt) && !(running && msg.interruptStatus === 'pending');
}

function hasVisibleAssistantContent(msg, hideDebug, running) {
  const visibleReasoning = !hideDebug && Boolean(msg.reasoning);
  const visibleTools = !hideDebug && (msg.tools || []).length > 0;
  const visibleInterrupt = isInterruptVisible(msg, running);
  const visibleText = Boolean(msg.text);

  return visibleReasoning || visibleTools || visibleInterrupt || visibleText;
}

function AssistantTurn({ msg, onResolveInterrupt, hideDebug, showHeader, running }) {
  const reasoningTitle = (
    <span className="flex items-center gap-2">
      <span className="text-clay">◆</span>
      Agent 思考过程
    </span>
  );
  const visibleReasoning = !hideDebug && Boolean(msg.reasoning);
  const visibleTools = !hideDebug && (msg.tools || []).length > 0;
  const visibleInterrupt = isInterruptVisible(msg, running);
  const visibleText = Boolean(msg.text);

  if (!hasVisibleAssistantContent(msg, hideDebug, running)) {
    return null;
  }

  return (
    <div className="mb-3">
      {showHeader && (
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-md bg-clay-soft border border-[#ecd9cd] flex items-center justify-center text-clay text-[12px] font-semibold">
            W
          </div>
          <div className="text-[13px] text-ink-2">WeSeeker Agent</div>
          <span className="chip font-mono">v2.1</span>
        </div>
      )}

      {visibleReasoning && (
        <Collapsible
          title={reasoningTitle}
          meta={`${msg.reasoning.length} chars`}
          defaultOpen={false}
        >
          <div className="px-4 py-3 text-[13.5px] leading-[1.75] text-ink-2 whitespace-pre-wrap">
            {msg.reasoning}
          </div>
        </Collapsible>
      )}

      {visibleTools && (msg.tools || []).map(t => <ToolBlock key={t.id} tool={t} />)}

      {visibleText && (
        <div className="text-ink">
          <Markdown>{msg.text}</Markdown>
        </div>
      )}

      {visibleInterrupt && (
        <div className="mt-4">
          <InterruptCard
            data={msg.interrupt}
            status={msg.interruptStatus}
            decision={msg.interruptDecision}
            errorMessage={msg.interruptError}
            onResolve={onResolveInterrupt}
          />
        </div>
      )}
    </div>
  );
}

function buildHiddenDebugIndices(messages) {
  const hidden = new Set();
  let hideAfterInterrupt = false;

  messages.forEach((msg, index) => {
    if (msg.role === 'user') {
      hideAfterInterrupt = false;
      return;
    }

    if (hideAfterInterrupt) hidden.add(index);

    if (
      msg.interrupt &&
      ['submitting', 'success', 'error'].includes(msg.interruptStatus || '')
    ) {
      hideAfterInterrupt = true;
    }
  });

  return hidden;
}

export default function Conversation({ messages, onResolveInterrupt, error, running }) {
  const scrollRef = useRef(null);
  const contentRef = useRef(null);
  const frameRef = useRef(null);
  const followStreamRef = useRef(false);
  const programmaticScrollRef = useRef(false);
  const hiddenDebugIndices = buildHiddenDebugIndices(messages);
  const renderedMessages = [];
  let shouldShowAssistantHeader = true;

  messages.forEach((m, i) => {
    if (m.role === 'user') {
      shouldShowAssistantHeader = true;
      renderedMessages.push(<UserBubble key={i} text={m.text} />);
      return;
    }

    const hideDebug = hiddenDebugIndices.has(i);
    if (!hasVisibleAssistantContent(m, hideDebug, running)) {
      return;
    }

    renderedMessages.push(
      <AssistantTurn
        key={i}
        msg={m}
        onResolveInterrupt={onResolveInterrupt}
        hideDebug={hideDebug}
        showHeader={shouldShowAssistantHeader}
        running={running}
      />
    );
    shouldShowAssistantHeader = false;
  });

  const cancelAutoScroll = () => {
    followStreamRef.current = false;
    programmaticScrollRef.current = false;
    if (!frameRef.current) return;
    cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
  };

  useLayoutEffect(() => {
    const el = scrollRef.current;
    const lastMessage = messages[messages.length - 1];
    if (!el || !lastMessage) return;

    if (lastMessage.role === 'user') {
      followStreamRef.current = true;
      animateScrollToBottom(el, frameRef, programmaticScrollRef);
      return;
    }

    if (followStreamRef.current && !frameRef.current) {
      stickToBottom(el, programmaticScrollRef);
    }
  }, [messages]);

  useEffect(() => {
    const el = scrollRef.current;
    const contentEl = contentRef.current;
    if (!el || !contentEl || typeof ResizeObserver === 'undefined') return undefined;

    const observer = new ResizeObserver(() => {
      if (!followStreamRef.current || frameRef.current) return;
      stickToBottom(el, programmaticScrollRef);
    });

    observer.observe(contentEl);
    return () => observer.disconnect();
  }, []);

  useEffect(() => () => {
    cancelAutoScroll();
  }, []);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    if (programmaticScrollRef.current) {
      if (!frameRef.current) programmaticScrollRef.current = false;
      return;
    }
    followStreamRef.current = isAtBottom(el);
  };

  return (
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      onWheel={cancelAutoScroll}
      onTouchMove={cancelAutoScroll}
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) cancelAutoScroll();
      }}
      className="flex-1 min-h-0 overflow-y-auto scroll-y"
    >
      <div ref={contentRef} className="max-w-[760px] mx-auto px-8 py-10">
        {messages.length === 0 && (
          <div className="text-center text-ink-3 mt-20">
            <div className="font-serif text-[22px] mb-2">WeSeeker Debug</div>
            <div className="text-[13px]">输入指令开始一次新对话</div>
          </div>
        )}
        {renderedMessages}
        {error && (
          <div className="card p-3 text-[13px] text-err border-[#f0c8c8] bg-[#fbeaea]">
            ⚠ {error}
          </div>
        )}
      </div>
    </div>
  );
}
